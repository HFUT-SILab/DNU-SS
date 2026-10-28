"""Read-only image and protocol checks; never rename or rewrite the dataset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from PIL import Image
from .data import read_split, validate_protocol
from .resources import DATASETS, split_file


def inspect_dataset(dataset, protocol, data_root, splits_dir=None):
    report = dict(
        dataset=dataset,
        protocol=protocol,
        valid=False,
        errors=[],
        missing=[],
        corrupt=[],
        non_square=[],
        split_sha256={},
    )
    records = {}
    try:
        for phase in ("train", "test"):
            with split_file(dataset, protocol, phase, splits_dir) as path:
                report["split_sha256"][phase] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                records[phase] = read_split(path, data_root)
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        return report
    try:
        report["split_report"] = validate_protocol(
            records["train"], records["test"], protocol
        )
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
    sizes, modes = Counter(), Counter()
    unique = sorted({p for values in records.values() for p, _ in values})
    for path in unique:
        if not path.is_file():
            report["missing"].append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image.load()
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                if image.width != image.height:
                    report["non_square"].append(str(path))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            report["corrupt"].append(dict(path=str(path), error=str(error)))
    report.update(
        images_checked=len(unique), image_sizes=dict(sizes), image_modes=dict(modes)
    )
    if report["corrupt"]:
        report["errors"].append(f"{len(report['corrupt'])} undecodable images")
    if report["non_square"]:
        report["errors"].append(
            f"{len(report['non_square'])} non-square ROIs; square ROIs are required"
        )
    report["valid"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--protocol", choices=("open", "closed"), default="open")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON report, outside the dataset"
    )
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(args.data_root.resolve()):
            parser.error("The report output must be outside data-root")
        if output.exists():
            parser.error("Refusing to overwrite an existing report")
    report = inspect_dataset(
        args.dataset, args.protocol, args.data_root, args.splits_dir
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        key: value
        for key, value in report.items()
        if key not in ("missing", "corrupt", "non_square")
    }
    for key in ("missing", "corrupt", "non_square"):
        summary[key + "_count"] = len(report[key])
        summary[key + "_examples"] = report[key][:5]
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
