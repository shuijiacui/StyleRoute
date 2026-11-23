"""Package source and real evidence, excluding virtual environments and unrelated runs."""

import argparse
import json
import zipfile

from src.project import ROOT, file_sha256, project_path, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/packages/M4_organized_source_and_evidence.zip")
    parser.add_argument("--include-weights", action="store_true")
    args = parser.parse_args()
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new package filename: {output}")
    files = set()
    for name in (
        "app.py",
        "src/contracts.py",
        "src/project.py",
        "src/integration.py",
        "src/outputs.py",
        "README.md",
        "docs/THIRD_PARTY.md",
        "resources/env/requirements.txt",
        "resources/env/requirements-dev.txt",
        "resources/env/requirements-lock.txt",
        "pyproject.toml",
        "start.ps1",
        ".gitignore",
        "docs/M4代码实现详细说明_三人Pre方案.md",
    ):
        files.add(project_path(name))
    for directory in (
        "src",
        "resources/images",
        "resources/configs",
        "resources/env",
        "resources/fonts",
        "resources/upstream",
        "tests",
        "docs",
        ".streamlit",
        "outputs/runs/validation",
        "outputs/runs/profile",
        "outputs/runs/final_demo",
        "outputs/runs/quality_diagnosis",
        "outputs/runs/creative_demo",
    ):
        for file in project_path(directory).rglob("*"):
            if (
                file.is_file()
                and "__pycache__" not in file.parts
                and file.suffix not in {".pyc", ".part"}
                and file.name not in {"pip-freeze.txt", "secrets.toml"}
            ):
                files.add(file)
    files.add(project_path("resources/weights/manifest.json"))
    files.add(project_path("outputs/reorganization/layout.json"))
    if args.include_weights:
        files.update(project_path("resources/weights").glob("*.pth"))
    inventory = [
        {
            "path": file.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(file),
            "bytes": file.stat().st_size,
        }
        for file in sorted(files)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in inventory:
            archive.write(project_path(item["path"]), item["path"])
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(
                {
                    "created_at": utc_now(),
                    "include_weights": args.include_weights,
                    "venv_included": False,
                    "files": inventory,
                    "setup_note": "Install dependencies; run node src/tools/download_assets.mjs.",
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    print(
        f"Created and CRC-verified {output}: {len(inventory)} files, "
        f"{output.stat().st_size / 1048576:.2f} MiB; weights={args.include_weights}"
    )


if __name__ == "__main__":
    main()
