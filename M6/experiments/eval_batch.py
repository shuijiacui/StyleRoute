"""Windows 计划任务驱动的 evaluation 矩阵断点续跑器。

每次被调用时在时间预算内尽可能多跑 (pair, method) 批次；已完成批次
（存在 __seed4.jsonl）自动跳过。计划任务每 5 分钟触发一次，电脑休眠
时暂停、唤醒后继续，进度不丢。

用法（一般由 eval_batch.bat 调用，勿手动重复执行）：
    python -m M6.experiments.eval_batch [--budget-seconds 200]
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
    args = parser.parse_args()

    methods = ["risk", "plain", "random_states", "global_scan"]
    out_root = _REPO_ROOT / "M6" / "outputs" / "evaluation"
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
            print("EVALUATION MATRIX COMPLETE", flush=True)
            return
        pair, method = todo
        done = len(list(out_root.glob("*/*__seed*.jsonl")))
        print(f">>> [{time.strftime('%H:%M:%S')}] {pair} / {method} "
              f"(已完成 {done}/1440 轨迹)", flush=True)
        subprocess.run(
            [str(python), "-m", "M6.experiments.run", "--split", "evaluation",
             "--pairs", pair, "--methods", method,
             "--seeds", *[str(s) for s in args.seeds]],
            cwd=_REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    print("batch window exhausted; next scheduled run will continue", flush=True)


if __name__ == "__main__":
    main()
