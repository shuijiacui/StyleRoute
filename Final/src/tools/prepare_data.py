"""Validate curated data, explicitly download it, or register an existing Places365 release."""

import argparse
import json
import subprocess

from PIL import Image

from src.project import ROOT, file_sha256, project_path, utc_now, write_json


def validate_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    images = manifest["content_images"] + manifest["style_images"]
    fingerprints = set()
    for item in images:
        path = project_path(item["local_path"])
        actual = file_sha256(path)
        if actual != item["sha256"] or actual in fingerprints:
            raise ValueError(f"Wrong hash or duplicate image: {item['id']}")
        fingerprints.add(actual)
        with Image.open(path) as image:
            image.load()
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"Unexpected dimensions: {item['id']}")
    content = {item["id"]: item for item in manifest["content_images"]}
    styles = {item["id"]: item for item in manifest["style_images"]}
    seen_pairs = set()
    for key, usage in (("pilot_pairs", "pilot"), ("evaluation_pairs", "evaluation")):
        expected = {
            (item["id"], style_id)
            for item in content.values()
            if item["usage"] == usage
            for style_id in styles
        }
        actual = set()
        for pair in manifest[key]:
            if pair["pair_id"] in seen_pairs:
                raise ValueError(f"Duplicate pair ID: {pair['pair_id']}")
            seen_pairs.add(pair["pair_id"])
            actual.add((pair["content_id"], pair["style_id"]))
        if actual != expected:
            raise ValueError(f"{key} does not match the declared split")
    return {
        "status": "passed",
        "images": len(images),
        "pilot_pairs": len(manifest["pilot_pairs"]),
        "evaluation_pairs": len(manifest["evaluation_pairs"]),
    }


def register_places(root, output, per_class):
    """Use the supplied release's labels, not an assumed class-folder layout."""
    if per_class < 2:
        raise ValueError("At least two images per class are needed for separate pilot/evaluation")
    classes = root / "categories_places365.txt"
    labels = root / "places365_val.txt"
    for required in (classes, labels):
        if not required.is_file():
            raise FileNotFoundError(f"Missing release metadata: {required}")
    mapping = {
        line.split()[0].rsplit("/", 1)[-1]: int(line.split()[1])
        for line in classes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    expected_classes = {"beach": 48, "mountain": 232, "downtown": 125}
    if any(mapping.get(name) != index for name, index in expected_classes.items()):
        raise ValueError("Places365 class mapping differs from the agreed official IDs")
    samples = [
        line.split() for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    current = json.loads(project_path("resources/images/manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "dataset_id": "places365-val-small-registered-v1",
        "created_at": utc_now(),
        "content_source": "Places365 validation, supplied local release",
        "selection_policy": f"Sort release-relative filenames per class; first {per_class}; "
        "first pilot, remaining evaluation; fail on unreadable/missing files.",
        "release_metadata_sha256": {p.name: file_sha256(p) for p in (classes, labels)},
        "content_images": [],
        "style_images": current["style_images"],
        "evaluation_pairs": [],
        "pilot_pairs": [],
    }
    for name, index in expected_classes.items():
        chosen = sorted(filename for filename, label in samples if int(label) == index)[:per_class]
        if len(chosen) != per_class:
            raise ValueError(f"Not enough images for {name}")
        for order, filename in enumerate(chosen):
            relative = filename.lstrip("/")
            candidates = [root / "val_256" / relative, root / relative]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(f"Missing selected Places365 file: {relative}")
            with Image.open(source) as image:
                image.load()
                width, height = image.size
            item = {
                "id": f"places_{name}_{order + 1:02}",
                "source": "Places365",
                "source_relative_path": relative,
                "local_path": str(source.resolve()),
                "class_name": name,
                "class_id": index,
                "usage": "pilot" if order == 0 else "evaluation",
                "sha256": file_sha256(source),
                "width": width,
                "height": height,
                "rights_note": (
                    "Consult Places365 release/image terms; code license is not image ownership."
                ),
            }
            manifest["content_images"].append(item)
            for style in manifest["style_images"]:
                manifest["pilot_pairs" if order == 0 else "evaluation_pairs"].append(
                    {
                        "pair_id": f"{item['id']}__{style['id']}",
                        "content_id": item["id"],
                        "style_id": style["id"],
                    }
                )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a dataset manifest: {output}")
    write_json(output, manifest)
    return validate_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="Explicitly restore/download the curated image set"
    )
    parser.add_argument("--places-root", help="Register an already obtained Places365 release")
    parser.add_argument("--per-class", type=int, default=7)
    parser.add_argument("--output", help="New manifest path when registering Places365")
    args = parser.parse_args()
    if args.places_root:
        if args.download:
            parser.error("--download handles curated data; it cannot fetch Places365")
        output = project_path(args.output or "resources/images/manifest_places365.json")
        print(register_places(project_path(args.places_root), output, args.per_class))
    else:
        if args.download:
            subprocess.run(
                [
                    "node",
                    str(ROOT / "src/tools/download_images.mjs"),
                    "--restore"
                    if project_path("resources/images/manifest.json").exists()
                    else "--download",
                ],
                cwd=ROOT,
                check=True,
            )
        print(validate_manifest(project_path(args.output or "resources/images/manifest.json")))


if __name__ == "__main__":
    main()
