"""Collect finished evaluations into one table.

    python summarize_eval.py saved/eval/test_*/results.json

Reads the results.json files inference.py writes and prints a markdown table
plus the derived columns the write-up actually argues with: the night/day gap
at IoU 0.5 and at IoU 0.75, side by side.

Those two columns are the point. A single "gap" number invites the answer
"our mAP is around 0.5, that seems fine" - which is measured at the loosest
threshold, the one where the problem is least visible. Printed as a pair, the
gap widening from @50 to @75 is the finding itself, and it is arithmetic on
the journal's own columns rather than anything this project's analysis code
computed.
"""

import argparse
import json
from pathlib import Path

COLUMNS = [
    ("map", "mAP@50-95"),
    ("map_50", "mAP@50"),
    ("map_75", "mAP@75"),
    ("map_small", "small"),
]


def gap(day, night):
    """Relative shortfall of night against day, in percent."""
    return 100 * (day - night) / day if day else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--label", action="append", default=[],
        help="row label; repeat once per path, in order (default: folder name)",
    )
    args = parser.parse_args()

    rows = []
    for i, path in enumerate(args.paths):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        label = args.label[i] if i < len(args.label) else Path(path).parent.name
        rows.append((label, data))

    splits = {d.get("split", "?") for _, d in rows}
    print(f"split: {', '.join(sorted(splits))}")
    if len(splits) > 1:
        print("ВНИМАНИЕ: строки посчитаны на РАЗНЫХ сплитах, они не сравнимы")

    header = (
        ["прогон"]
        + [f"ночь {t}" for _, t in COLUMNS]
        + [f"день {t}" for _, t in COLUMNS]
        + ["разрыв@50", "разрыв@75"]
    )
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for label, data in rows:
        night, day = data["night"], data["day"]
        cells = (
            [label]
            + [f"{night[k]:.4f}" for k, _ in COLUMNS]
            + [f"{day[k]:.4f}" for k, _ in COLUMNS]
            + [
                f"{gap(day['map_50'], night['map_50']):.1f}%",
                f"**{gap(day['map_75'], night['map_75']):.1f}%**",
            ]
        )
        print("| " + " | ".join(cells) + " |")

    # Each row's night mAP against the first row's: what every step added.
    if len(rows) > 1:
        base = rows[0][1]["night"]["map"]
        print()
        print(f"прирост ночного mAP относительно «{rows[0][0]}» ({base:.4f}):")
        prev = base
        for label, data in rows[1:]:
            value = data["night"]["map"]
            print(
                f"  {label:28} {value:.4f}  "
                f"итого {value - base:+.4f}  шаг {value - prev:+.4f}"
            )
            prev = value


if __name__ == "__main__":
    main()
