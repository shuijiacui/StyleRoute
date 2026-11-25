"""Export side-by-side comparisons from exact saved results."""

import argparse
import json

from PIL import Image, ImageDraw

from src.experiments.artifacts import read_result
from src.models.preprocessing import tensor_to_pil
from src.project import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="pilot", choices=["pilot", "evaluation"])
    parser.add_argument("--out-root", default="outputs/experiments/integrated-v2")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    root = project_path(args.out_root) / args.split
    rows = json.loads((root / "summary.json").read_text(encoding="utf-8"))["runs"]
    rows = [r for r in rows if not args.pairs or r["pair_id"] in args.pairs]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    for row in rows[: args.limit]:
        image, content, _ = read_result(row)
        artifacts = project_path(row["artifact_dir"])
        with Image.open(artifacts / "default.png") as opened:
            default = opened.convert("RGB")
        panels = [tensor_to_pil(content), default, tensor_to_pil(image)]
        width, height = panels[0].size
        canvas = Image.new("RGB", (width * 3, height + 30), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(
            zip(panels, ["Content", "Default 0.6", f"{row['method']} seed {row['seed']}"])
        ):
            canvas.paste(panel, (i * width, 30))
            draw.text((i * width + 5, 5), label, fill="black")
        canvas.save(out / f"{row['pair_id']}__{row['method']}__seed{row['seed']}.png")
    print(f"Saved {min(len(rows), args.limit)} comparisons: {out}")


if __name__ == "__main__":
    main()
