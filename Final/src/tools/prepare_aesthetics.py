"""Download and verify the AVA NIMA checkpoint used by landscape search."""

import argparse
import json
import socket
import urllib.request

from src.project import file_sha256, project_path, write_json
from src.search.aesthetics import CHECKPOINT, URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=URL)
    parser.add_argument("--musiq", action="store_true", help="Also prepare final-only AVA metric")
    args = parser.parse_args()
    socket.setdefaulttimeout(30)
    path = project_path(CHECKPOINT)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = (
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if path.with_suffix(".json").exists()
        else {}
    )
    if not path.exists():
        temporary = path.with_suffix(".download")
        print(f"Downloading {path.name}", flush=True)
        urllib.request.urlretrieve(args.base_url.rstrip("/") + "/" + path.name, temporary)
        if not file_sha256(temporary).startswith("dc4e8265"):
            raise ValueError("Downloaded NIMA does not match published checksum prefix")
        temporary.replace(path)
    checksum = file_sha256(path)
    if not checksum.startswith("dc4e8265"):
        raise ValueError("NIMA checksum does not match published checkpoint")
    write_json(
        path.with_suffix(".json"),
        {
            "model": "nima-vgg16-ava",
            "dataset": "AVA",
            "range": [1, 10],
            "source": URL + path.name,
            "download_source": previous.get(
                "download_source", args.base_url.rstrip("/") + "/" + path.name
            ),
            "sha256": checksum,
            "implementation": "IQA-PyTorch reproduction; not original authors' weights",
        },
    )
    print(f"Verified {path.name}: {checksum}", flush=True)
    if args.musiq:
        path = project_path("resources/weights/musiq_ava_ckpt-e8d3f067.pth")
        if not path.exists():
            temporary = path.with_suffix(".download")
            print(f"Downloading {path.name}", flush=True)
            urllib.request.urlretrieve(args.base_url.rstrip("/") + "/" + path.name, temporary)
            if not file_sha256(temporary).startswith("e8d3f067"):
                raise ValueError("MUSIQ download checksum mismatch")
            temporary.replace(path)
        checksum = file_sha256(path)
        if not checksum.startswith("e8d3f067"):
            raise ValueError("MUSIQ checksum mismatch")
        write_json(
            path.with_suffix(".json"),
            {
                "model": "musiq-ava",
                "sha256": checksum,
                "source": URL + path.name,
                "download_source": args.base_url.rstrip("/") + "/" + path.name,
                "range": [1, 10],
                "use": "final evaluation only",
            },
        )
        print(f"Verified {path.name}: {checksum}", flush=True)


if __name__ == "__main__":
    main()
