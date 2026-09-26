"""Markdown rows for the README from predictions_samples.json: events per class per clip and the time budget.

    python tools/results_table.py                       # both tables from predictions_samples.json
    python tools/results_table.py --table results       # only "Clip | Length | Events we report"
    python tools/results_table.py --table budget        # only the time-budget table (needs the "log" section)
    python tools/results_table.py path/to/predictions.json

Paste the output over the corresponding rows in README.md after a re-run; nothing is written by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASSES  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def events_row(name: str, entry: dict, log: dict | None) -> str:
    counts = Counter(e[2] for e in entry.get("events", []) if len(e) >= 3)
    parts = [f"{c} {counts[c]}" for c in CLASSES if counts.get(c)] + sorted(f"{c} {n}" for c, n in counts.items() if c not in CLASSES)
    duration = (log or {}).get("duration")
    if duration is None:
        ends = [float(e[1]) for e in entry.get("events", [])] + [float(p[0]) for p in entry.get("risk", [])[-1:]]
        duration = max(ends) if ends else 0.0
    return f"| {name} | {duration:.0f} s | {', '.join(parts) if parts else 'none'} |"


def budget_row(name: str, log: dict) -> str:
    d = float(log.get("duration") or 0.0)
    a, b, total = log.get("part_a_sec"), log.get("part_b_sec"), log.get("total_sec")
    if d <= 0 or a is None or b is None or total is None:
        return f"| {name} | — | — | — | — | 3× |"
    return (f"| {name} | {d:.0f} s | {a:.0f} s ({a / d:.2f}×) | {b:.0f} s ({b / d:.2f}×) | "
            f"{total:.0f} s = {total / d:.2f}× | 3× |")


def main() -> int:
    ap = argparse.ArgumentParser(description="README table rows from a predictions file")
    ap.add_argument("pred", nargs="?", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--table", choices=["results", "budget", "all"], default="all")
    args = ap.parse_args()
    data = load(Path(args.pred))
    videos: dict = data.get("videos", {})
    logs: dict = data.get("log", {}) or {}
    if args.table in ("results", "all"):
        print("| Clip | Length | Events we report |")
        print("|---|---|---|")
        for name, entry in videos.items():
            print(events_row(name, entry, logs.get(name)))
        total = Counter(e[2] for entry in videos.values() for e in entry.get("events", []) if len(e) >= 3)
        print()
        print("Total: " + ", ".join(f"{c} {total[c]}" for c in CLASSES if total.get(c)))
    if args.table == "all":
        print()
    if args.table in ("budget", "all"):
        if not logs:
            print("(no \"log\" section in the predictions file: the time budget table needs the harness log)")
            return 0
        print("| Footage | Length | Part A | Part B | Total | Limit |")
        print("|---|---|---|---|---|---|")
        for name in videos:
            if name in logs:
                print(budget_row(name, logs[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
