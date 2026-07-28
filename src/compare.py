"""Compare test_metrics.json from two training runs (e.g. before/after a
change to hyperparameters or architecture). --baseline/--improved are just
labels for "before" and "after" -- both should point at a resnet run's
test_metrics.json.

Usage:
    python src/compare.py \
        --baseline outputs/resnet_v1/test_metrics.json \
        --improved outputs/resnet_v2/test_metrics.json
"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two test_metrics.json files.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--output", default=None, help="Optional path to save the comparison as Markdown.")
    return parser.parse_args()


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    base = load(args.baseline)
    imp = load(args.improved)

    lines = []
    lines.append("| Metric | Baseline | Improved | Delta |")
    lines.append("|---|---|---|---|")
    for name in base["test_metrics"]:
        b = base["test_metrics"][name]
        i = imp["test_metrics"].get(name)
        if i is None:
            continue
        lines.append(f"| {name} | {b:.4f} | {i:.4f} | {i - b:+.4f} |")

    lines.append("")
    lines.append("| Digit | Baseline acc | Improved acc | Delta |")
    lines.append("|---|---|---|---|")
    for digit in sorted(base["per_position_accuracy"], key=int):
        b = base["per_position_accuracy"][digit]
        i = imp["per_position_accuracy"].get(digit)
        if i is None:
            continue
        lines.append(f"| {digit} | {b:.4f} | {i:.4f} | {i - b:+.4f} |")

    table = "\n".join(lines)
    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
        print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
