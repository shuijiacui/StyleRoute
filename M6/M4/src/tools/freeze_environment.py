"""Record actual installed project dependency closure; never modify the environment."""

import importlib.metadata as metadata
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from src.project import project_path, runtime_info, write_json


def main():
    roots = [
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "streamlit",
        "psutil",
        "pytest",
        "ruff",
        "fonttools",
    ]
    pending, versions = list(roots), {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in versions:
            continue
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        for line in distribution.requires or []:
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    lock = "# Verified Windows / Python 3.12 CPU versions. See README for PyTorch channels.\n"
    lock += "\n".join(f"{name}=={version}" for name, version in sorted(versions.items())) + "\n"
    project_path("resources/env/requirements-lock.txt").write_text(lock, encoding="utf-8")
    directory = project_path("outputs/runs/validation")
    directory.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True
    ).stdout
    (directory / "pip-freeze.txt").write_bytes(freeze)
    write_json(
        directory / "environment.json",
        {
            **runtime_info(),
            "project_dependency_versions": versions,
            "environment_note": "Read-only base NN packages; local venv app/test packages.",
        },
    )
    print(f"Recorded {len(versions)} installed project/transitive package versions.")


if __name__ == "__main__":
    main()
