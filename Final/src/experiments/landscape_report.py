"""Re-score saved FP32 outputs with common A/U and optional final-only MUSIQ-AVA."""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from src.models.compensation import ContentGeometry
from src.project import file_sha256, project_path, write_json
from src.search.aesthetics import AestheticMetric
from src.search.landscape import UniformityMetric

from .artifacts import read_result


def musiq_metric(device):
    import pyiqa

    path = project_path("resources/weights/musiq_ava_ckpt-e8d3f067.pth")
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha256(path) != manifest["sha256"]:
        raise ValueError("MUSIQ checkpoint checksum mismatch")
    # num_class must be explicit when passing a local AVA checkpoint.
    model = pyiqa.create_metric(
        "musiq-ava", pretrained_model_path=str(path), num_class=10, device=device
    )
    return model, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", help="Path to run_matrix summary.json")
    parser.add_argument("--musiq", action="store_true")
    parser.add_argument("--out", help="Default: landscape_report.json next to summary")
    args = parser.parse_args()
    source = project_path(args.summary)
    saved = json.loads(source.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    aesthetic = AestheticMetric("cpu")
    independent, provenance = musiq_metric("cpu") if args.musiq else (None, None)
    pairs, rows = {}, []

    @torch.inference_mode()
    def measure(image, metric):
        return {
            "A": aesthetic(image),
            **metric(image),
            "MUSIQ_AVA": float(independent(image).item()) if independent else None,
        }

    for row in saved["runs"]:
        image, content, metadata = read_result(row)
        artifact = project_path(row["artifact_dir"])
        key = str(artifact)
        baseline_file = artifact / "default.npz"
        if file_sha256(baseline_file) != metadata.get("baseline_sha256"):
            raise ValueError("Missing or changed FP32 baseline; re-run in a fresh output root")
        if key not in pairs:
            with np.load(project_path(row["region_artifact"]), allow_pickle=False) as data:
                labels = data["labels"].copy()
            uniformity = UniformityMetric(ContentGeometry(content, labels))
            with np.load(baseline_file, allow_pickle=False) as data:
                baseline = torch.from_numpy(data["image_rgb"].copy())
            pairs[key] = (uniformity, measure(baseline, uniformity), baseline)
        uniformity, baseline_scores, baseline = pairs[key]
        measured = measure(image, uniformity)
        trajectory = project_path(row["trajectory"]).read_text(encoding="utf-8").splitlines()
        initial = json.loads(trajectory[0])
        output = {
            "pair_id": row["pair_id"],
            "method": row["method"],
            "seed": row["seed"],
            "baseline": {**baseline_scores, "Q": initial["Q"], "T": initial["T"]},
            "selected": {**measured, "Q": row["Q"], "T": row["T"]},
            "delta_A": measured["A"] - baseline_scores["A"],
            "delta_U": measured["U"] - baseline_scores["U"],
            "U_reduction_fraction": (baseline_scores["U"] - measured["U"])
            / max(baseline_scores["U"], 1e-8),
            "T_ratio": row["T"] / max(initial["T"], 1e-12),
            "delta_MUSIQ_AVA": measured["MUSIQ_AVA"] - baseline_scores["MUSIQ_AVA"]
            if independent
            else None,
            "rgb_mean_abs_delta": float((image - baseline).abs().mean()),
            "fallback": row["used_default"],
            "elapsed_s": row["elapsed_s"],
            "candidate_count": row["candidate_count"],
            "beta": row.get("beta", 0),
        }
        rows.append(output)
        print(
            f"[{len(rows)}/{len(saved['runs'])}] {row['run_id']} "
            f"dA={output['delta_A']:+.4f} dU={output['delta_U']:+.4f}",
            flush=True,
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    aggregate = {}
    for method, entries in grouped.items():
        aggregate[method] = {
            "count": len(entries),
            "fallback_count": sum(r["fallback"] for r in entries),
            "A_improved_count": sum(r["delta_A"] > 0.01 for r in entries),
            "A_degraded_count": sum(r["delta_A"] < -0.01 for r in entries),
            "U_improved_count": sum(r["delta_U"] < -0.01 for r in entries),
            "MUSIQ_improved_count": sum(r["delta_MUSIQ_AVA"] > 0.01 for r in entries)
            if independent
            else None,
            "MUSIQ_degraded_count": sum(r["delta_MUSIQ_AVA"] < -0.01 for r in entries)
            if independent
            else None,
            **{
                f"mean_{key}": float(np.mean([r[key] for r in entries]))
                for key in (
                    "delta_A",
                    "delta_U",
                    "U_reduction_fraction",
                    "elapsed_s",
                    "rgb_mean_abs_delta",
                    "T_ratio",
                )
            },
        }
    write_json(
        project_path(args.out) if args.out else source.with_name("landscape_report.json"),
        {
            "source": str(source),
            "source_sha256": file_sha256(source),
            "rows": rows,
            "aggregate": aggregate,
            "aesthetic": aesthetic.provenance,
            "independent": provenance,
            "scope": "Automatic proxies, not proof of human preference. J is method-specific.",
            "independent_note": "MUSIQ is excluded from search; both aesthetic models use AVA.",
        },
    )


if __name__ == "__main__":
    main()
