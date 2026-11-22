"""消融/敏感性矩阵的断点续跑器（eval_batch.py 的通用化版本）。

与 eval_batch 的差异：通过参数指定配置、方法和输出根目录，
用于主矩阵之外的补充实验（如 lam=0 关系约束消融）。

完成标记 = 每个 (pair, method) 存在 __seed{last}.jsonl。

用法（一般由 ablation_batch.bat / 计划任务调用）：
    python -m M6.experiments.ablation_batch \
        --config M6/configs/experiment_lam0.json \
        --out-root M6/outputs_ablation/lam0 \
        --methods risk plain --budget-seconds 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pairs() -> list[str]:
    manifest = json.loads(
        (_REPO_ROOT / "M4" / "resources" / "images" / "manifest.json").read_text(encoding="utf-8")
    )
    return [p["pair_id"] for p in manifest["evaluation_pairs"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--config", default=str(_REPO_ROOT / "M6" / "configs" / "experiment_lam0.json"))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--methods", nargs="*", default=None, help="默认取配置里的 methods")
    args = parser.parse_args()

    methods = args.methods or json.loads(Path(args.config).read_text(encoding="utf-8"))["methods"]
    out_root = _REPO_ROOT / args.out_root / args.split
    python = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    start = time.time()

    while time.time() - start < args.budget_seconds:
        todo = None
        for pair in _pairs():
            for method in methods:
                marker = out_root / pair / f"{method}__seed{args.seeds[-1]}.jsonl"
                if not marker.exists():
                    todo = (pair, method)
                    break
            if todo:
                break
        if todo is None:
            print("ABLATION MATRIX COMPLETE", flush=True)
            return
        pair, method = todo
        done = len(list(out_root.glob("*/*__seed*.jsonl")))
        print(f">>> [{time.strftime('%H:%M:%S')}] {pair} / {method} "
              f"(已完成 {done} 轨迹)", flush=True)
        subprocess.run(
            [str(python), "-m", "M6.experiments.run", "--split", args.split,
             "--config", args.config, "--out-root", args.out_root,
             "--pairs", pair, "--methods", method,
             "--seeds", *[str(s) for s in args.seeds]],
            cwd=_REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    print("batch window exhausted; next scheduled run will continue", flush=True)


if __name__ == "__main__":
    main()
