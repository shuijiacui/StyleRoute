"""Measure real phases and fair K-state encoding-cache comparisons on one model."""

import argparse
import random
import statistics
import threading
import time
from collections import defaultdict

import numpy as np
import psutil
import torch

from src.integration import integration_status, run_search
from src.models.adain import AdaINEngine
from src.models.preprocessing import prepare_content, prepare_style
from src.postcard.render import render_postcard
from src.project import digest, load_config, project_path, utc_now, write_json


def sync(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, *, phase, repeat, device, records, method="m4", sampling_ms=10):
    """Synchronized wall time, endpoint RSS and an explicitly approximate sampled peak."""
    process = psutil.Process()
    before = process.memory_info().rss
    samples = [before]
    stop = threading.Event()

    def sample():
        while not stop.wait(sampling_ms / 1000):
            samples.append(process.memory_info().rss)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    sync(device)
    cuda = device is not None and device.type == "cuda"
    allocated = torch.cuda.memory_allocated(device) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    record = {
        "phase": phase,
        "repeat_index": repeat,
        "method": method,
        "rss_before_bytes": before,
        "rss_sample_interval_ms": sampling_ms,
        "cuda_allocated_before_bytes": allocated,
        "status": "success",
        "error_type": None,
        "error_message": None,
    }
    started = time.perf_counter()
    try:
        value = function()
        # Cold construction starts without pre-initializing CUDA. Synchronize after load.
        sync(value.device if isinstance(value, AdaINEngine) else device)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        stop.set()
        worker.join()
        after = process.memory_info().rss
        record.update(
            rss_after_bytes=after,
            rss_sample_peak_bytes=max(*samples, after),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
        )
        records.append(record)
    return value


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        if record["status"] == "success":
            grouped[(record["phase"], record["method"])].append(record["elapsed_ms"])
    return [
        {
            "phase": key[0],
            "method": key[1],
            "n": len(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for key, values in grouped.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--config", default="resources/configs/default.json")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--integrated", action="store_true", help="Requires actual M5/M6 adapters")
    args = parser.parse_args()
    if args.repeats < 5 or args.warmups < 1:
        parser.error("Use at least five repeats and one warmup")
    config = load_config(args.config)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    content, style = project_path(args.content), project_path(args.style)
    records, report = [], {"started_at": utc_now(), "records": []}
    device = None
    try:
        engine = measure(
            lambda: AdaINEngine(config),
            phase="model_load_cold_process",
            repeat=0,
            device=None,
            records=records,
        )
        device = engine.device
        report.update(
            runtime=engine.runtime,
            config=config,
            config_hash=digest(config),
            warmups=args.warmups,
            repeats=args.repeats,
            seed=0,
            measured_scope="M4 warm-model inference/export; no page refresh or candidate disk IO",
            cold_scope="one first construction in a fresh process; not OS disk-cache cold",
            memory_scope="approximate process RSS sampled every 10ms; CUDA allocated, not reserved",
        )

        def preprocess():
            return (
                prepare_content(content, tuple(config["content_hw"])),
                prepare_style(style, config["style_long_side"], config["minimum_style_short_side"]),
            )

        prepared_content, prepared_style = preprocess()
        pair = engine.prepare_preprocessed(prepared_content, prepared_style)
        states = (0.3, 0.6, 0.9)
        report.update(
            pair_id=pair.pair_id,
            input_hw=pair.input_hw,
            style_hw=pair.metadata["style_hw"],
            feature_hw=pair.feature_hw,
            model_hash=engine.model_hash,
            states=list(states),
            candidate_count_per_k_trial=len(states),
            integration=integration_status(config),
            inputs=pair.metadata,
        )
        # Check optimized and unoptimized outputs before publishing any speed comparison.
        maximum_error = 0.0
        for alpha in states:
            cached = engine.render_global(pair, alpha).image_rgb
            fresh = engine.render_global(
                engine.prepare_preprocessed(prepared_content, prepared_style), alpha
            ).image_rgb
            maximum_error = max(maximum_error, float((cached - fresh).abs().max()))
            torch.testing.assert_close(cached, fresh, rtol=1e-5, atol=1e-6)
        report["cache_correctness_max_abs_error"] = maximum_error
        font = project_path(config["font_path"])
        result = engine.render_global(pair, 0.6)

        def export(image=result.image_rgb):
            return render_postcard(
                image, "山海来信", "把眼前的风景，寄给远方的你。", "minimal", font
            )

        def k_states(cached):
            shared = (
                engine.prepare_preprocessed(prepared_content, prepared_style) if cached else None
            )
            for alpha in states:
                active = (
                    shared
                    if shared is not None
                    else engine.prepare_preprocessed(prepared_content, prepared_style)
                )
                engine.render_global(active, alpha)

        def m4_end_to_end():
            active = engine.prepare(content, style)
            return export(engine.render_global(active, 0.6).image_rgb)

        def integrated():
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            active = engine.prepare(content, style)
            initial = engine.render_global(active, config["default_alpha"])
            return run_search(engine, active, initial)

        if args.integrated and not integration_status(config)[0]:
            raise RuntimeError(
                "Actual M5/M6 adapters are missing; integrated timing cannot be fabricated"
            )
        for _ in range(args.warmups):
            preprocess()
            engine.prepare_preprocessed(prepared_content, prepared_style)
            engine.render_global(pair, 0.6)
            export()
            k_states(True)
            k_states(False)
            if args.integrated:
                integrated()
        for repeat in range(args.repeats):
            phases = [
                ("image_preparation", preprocess),
                (
                    "feature_preparation",
                    lambda: engine.prepare_preprocessed(prepared_content, prepared_style),
                ),
                ("single_global_candidate", lambda: engine.render_global(pair, 0.6)),
                ("postcard_export", export),
                ("end_to_end_m4_warm_model", m4_end_to_end),
            ]
            for phase, function in phases:
                measure(function, phase=phase, repeat=repeat, device=device, records=records)
            # Alternate order to reduce a fixed ordering bias. Both methods include their
            # own feature preparation; neither inherits the other's candidate result cache.
            for cached in (True, False) if repeat % 2 == 0 else (False, True):
                measure(
                    lambda flag=cached: k_states(flag),
                    phase="k_state_total",
                    repeat=repeat,
                    device=device,
                    records=records,
                    method="cached" if cached else "reencode",
                )
            if args.integrated:
                measure(
                    integrated,
                    phase="full_search_warm_model",
                    repeat=repeat,
                    device=device,
                    records=records,
                    method="real_m5_m6",
                )
        report.update(status="success", counters=dict(engine.counters))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        for record in records:
            record.update(
                pair_id=report.get("pair_id"),
                actual_device=str(device) if device else None,
                dtype="float32",
                seed=0,
            )
        report.update(records=records, summary=summarize(records), finished_at=utc_now())
        write_json(project_path(args.output), report)
    print(f"Actual measurements saved to {project_path(args.output)}")


if __name__ == "__main__":
    main()
