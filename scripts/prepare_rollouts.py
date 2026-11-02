"""Prepare NIKA lab and case images before timed policy comparisons."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from nika.firecracker.schedule import prebuild_images
from nika.firecracker.image import list_images, image_key, variant_for_row
from nika.firecracker.case import list_cases, case_key, wants_case_image


def main():
    tasks = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2])
    started = time.monotonic()
    prebuild_images(
        [SimpleNamespace(row=row["case"]) for split in tasks.values() for row in split],
        max_parallel=1,
    )
    images, cases = list_images(), list_cases()
    record = {
        "preparation_sec": time.monotonic() - started,
        "images": images,
        "cases": cases,
    }
    output.write_text(json.dumps(record, indent=2, default=str) + "\n")
    rows = [row["case"] for split in tasks.values() for row in split]
    required_images = {image_key(variant_for_row(row)) for row in rows}
    required_cases = {case_key(row) for row in rows if wants_case_image(row)}
    ready_images = {row["key"] for row in images if row["ready"] and row["current"]}
    ready_cases = {row["key"] for row in cases if row["ready"] and row["current"]}
    if not required_images <= ready_images or not required_cases <= ready_cases:
        raise RuntimeError("Some requested images were not prepared")


if __name__ == "__main__":
    main()
