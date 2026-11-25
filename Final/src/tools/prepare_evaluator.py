"""Explicitly restore verified ImageNet VGG19 evaluation features into this project."""

import argparse
import concurrent.futures
import time
import urllib.request
from pathlib import Path

import torch

from src.project import file_sha256, project_path, write_json

URL = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
SIZE = 574673361
EXPECTED_SHA256 = "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"


def download(directory, workers=8, url=URL):
    directory.mkdir(parents=True, exist_ok=True)
    size = 1024 * 1024
    starts = list(range(0, SIZE, size))

    def part(begin):
        end = min(begin + size, SIZE) - 1
        path = directory / f"{begin:012d}.part"
        if path.exists() and path.stat().st_size == end - begin + 1:
            return path
        for attempt in range(3):
            try:
                source_url = URL if attempt == 1 else url
                request = urllib.request.Request(
                    source_url, headers={"Range": f"bytes={begin}-{end}"}
                )
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=20) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{SIZE}"
                    ):
                        raise RuntimeError("Official server returned an unexpected byte range")
                    blocks = []
                    received = 0
                    while received < end - begin + 1:
                        if time.monotonic() - started > 45:
                            raise TimeoutError(
                                "Range exceeded 45 seconds; retry with cached progress"
                            )
                        block = response.read1(min(65536, end - begin + 1 - received))
                        if not block:
                            break
                        blocks.append(block)
                        received += len(block)
                    data = b"".join(blocks)
                if len(data) != end - begin + 1:
                    raise RuntimeError("Incomplete evaluator download range")
                path.write_bytes(data)
                return path
            except Exception:
                if attempt == 2:
                    raise
        raise RuntimeError("Download failed")

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(part, begin) for begin in starts]
        for count, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            try:
                job.result()
            except Exception as error:
                failures.append(str(error))
                print(f"Range failed; retained complete chunks for resume: {error}", flush=True)
            if count % 32 == 0 or count == len(jobs):
                print(f"Downloaded {count}/{len(jobs)} MiB ranges", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} ranges failed; rerun to resume verified-length chunks")
    target = directory / "vgg19-dcbb9e9d.pth"
    with target.open("wb") as output:
        for begin in starts:
            output.write((directory / f"{begin:012d}.part").read_bytes())
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Previously downloaded official full checkpoint"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="resources/weights/vgg19_features.pth")
    parser.add_argument(
        "--url", default=URL, help="Download mirror; the same full SHA256 is required"
    )
    args = parser.parse_args()
    source = args.source or download(
        project_path("outputs/cache/evaluator-download"), args.workers, args.url
    )
    digest = file_sha256(source)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Official VGG19 SHA256 mismatch: {digest}")
    state = torch.load(source, map_location="cpu", weights_only=True)
    features = {
        key.removeprefix("features."): value
        for key, value in state.items()
        if key.startswith("features.")
    }
    if not features:
        raise ValueError("Checkpoint has no VGG19 features")
    target = project_path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, target)
    write_json(
        target.with_suffix(".json"),
        {
            "weights": "IMAGENET1K_V1",
            "source": URL,
            "download_source": args.url if args.source is None else str(args.source),
            "source_sha256": digest,
            "sha256": file_sha256(target),
            "format": "torchvision.vgg19.features",
        },
    )
    print(f"Verified evaluator features saved: {target}")


if __name__ == "__main__":
    main()
