"""M4 orchestration only. M5 segmentation and M6 objective/search stay external."""

import importlib
import inspect
import math

import torch

from src.contracts import GenerationResult, RegionPack, SearchResult
from src.models.fusion import fuse_region_features, validate_alphas


class IntegrationUnavailable(RuntimeError):
    pass


def load_adapter(spec: str):
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Adapter must be module.path:function")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise IntegrationUnavailable(
            f"Cannot import {spec}: missing {error.name}. "
            "Provide the real M5/M6 module; no mock search is substituted."
        ) from error
    function = getattr(module, function_name, None)
    if not callable(function):
        raise IntegrationUnavailable(f"Adapter callable not found: {spec}")
    return function


def integration_status(config: dict) -> tuple[bool, list[str]]:
    errors = []
    for key in ("region_adapter", "search_adapter"):
        try:
            load_adapter(config[key])
        except Exception as error:
            # Availability is a UI diagnostic boundary; actual run failures still propagate.
            errors.append(f"{type(error).__name__}: {error}")
    return not errors, errors


def validate_search_result(result, pair, prepared_regions, allowed):
    if not isinstance(result, SearchResult):
        raise ValueError("M6 must return contracts.SearchResult")
    if result.status not in {
        "improved",
        "no_improvement",
        "no_feasible",
        "budget_exhausted",
        "cancelled",
        "failed",
    }:
        raise ValueError(f"Unknown search status: {result.status}")
    validate_alphas(
        result.region_alphas, len(prepared_regions.region_ids), pair.content_features.device
    )
    if any(
        not any(math.isclose(value, choice, abs_tol=1e-6) for choice in allowed)
        for value in result.region_alphas
    ):
        raise ValueError("M6 returned alphas outside the formal search domain")
    image = result.best_image_rgb
    if (
        image.shape != pair.content_tensor.shape
        or image.dtype != torch.float32
        or image.device != pair.content_features.device
        or not torch.isfinite(image).all().item()
        or (image < 0).any().item()
        or (image > 1).any().item()
    ):
        raise ValueError("M6 best image violates the canonical image tensor contract")
    if set(("Q", "T", "S", "J")) - result.metrics.keys():
        raise ValueError("M6 must report Q/T/S/J, using None only for unmeasured values")
    for name, value in result.metrics.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"M6 returned nonfinite metric: {name}")
    for name in ("candidate_count", "decoder_calls", "state_cache_hits"):
        value = getattr(result, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"M6 {name} must be a nonnegative integer")
    if result.candidate_count < 1:
        raise ValueError("The initial/default state must count toward the search budget")
    if not isinstance(result.metadata.get("used_default"), bool):
        raise ValueError("M6 must explicitly report metadata['used_default']")
    if (
        result.status in {"no_improvement", "no_feasible", "failed"}
        and not result.metadata["used_default"]
    ):
        raise ValueError("This fallback status must explicitly identify use of the default result")
    if result.metadata["used_default"] and not all(
        math.isclose(value, 0.6, abs_tol=1e-6) for value in result.region_alphas
    ):
        raise ValueError("Default fallback must use the agreed all-0.6 state")
    if result.status == "improved" and (
        result.metadata["used_default"]
        or any(result.metrics[key] is None for key in ("Q", "T", "S", "J"))
    ):
        raise ValueError(
            "An improved result needs measured Q/T/S/J and cannot be a default fallback"
        )


def run_search(engine, pair, default, progress_callback=None):
    """Call real adapters; one fresh solver invocation owns its own state cache.

    Returns (RegionPack, GenerationResult, SearchResult). The default already exists
    and is passed into M6; it is not generated again here. Failures propagate.
    """
    build_regions = load_adapter(engine.config["region_adapter"])
    solve = load_adapter(engine.config["search_adapter"])
    if not math.isclose(engine.config["default_alpha"], 0.6):
        raise ValueError("The formal search contract requires an initial global alpha of 0.6")
    regions = build_regions(pair.content_rgb, feature_hw=pair.feature_hw, device=engine.device)
    if not isinstance(regions, RegionPack):
        raise ValueError("M5 must return contracts.RegionPack")
    prepared = engine.prepare_regions(pair, regions)
    allowed = engine.config["allowed_region_alphas"]

    def generate(region_alphas):
        alphas = tuple(float(value) for value in region_alphas)
        if any(
            not any(math.isclose(value, choice, abs_tol=1e-6) for choice in allowed)
            for value in alphas
        ):
            raise ValueError(f"Formal search alphas must belong to {allowed}")
        return engine.render_regions(pair, prepared, alphas)

    kwargs = dict(
        generate=generate,
        pair=pair,
        regions=regions,
        initial_result=default,
        search_config=dict(engine.config.get("search_config", {})),
    )
    parameters = inspect.signature(solve).parameters
    if "progress_callback" in parameters:
        kwargs["progress_callback"] = progress_callback
    result = solve(**kwargs)
    validate_search_result(result, pair, prepared, allowed)
    if result.metadata["used_default"] and not torch.equal(
        result.best_image_rgb, default.image_rgb
    ):
        raise ValueError("M6 marked a default fallback but returned a different image")
    alphas = validate_alphas(result.region_alphas, len(prepared.region_ids), engine.device)
    _, alpha_map = fuse_region_features(
        pair.content_features, pair.adain_features, prepared.soft_masks, alphas
    )
    best = GenerationResult(
        result.best_image_rgb,
        alpha_map,
        {
            "pair_id": pair.pair_id,
            "kind": "search_return",
            "region_alphas": list(result.region_alphas),
            "status": result.status,
            "used_default": result.metadata["used_default"],
        },
    )
    return regions, best, result
