"""Fail (exit 1) unless a small test run produced usable tool calls.

Used by the pipeline before each new model's full run, to catch chat-template
or output-format problems early.

  python scripts/smoke_check.py --run mistral-7b
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT                         # noqa: E402
from common.judge import parse_calls, strip_reasoning  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-frac", type=float, default=0.5)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(ROOT / "results" / "attack" / args.run / "responses.jsonl")]
    ok = sum(bool(parse_calls(r["response"])) for r in rows)
    for r in rows[:3]:
        print("---", r["setting"], "| final answer:", strip_reasoning(r["response"])[:300].replace("\n", " "))
    print(f"{ok}/{len(rows)} responses contain a parseable tool call")
    sys.exit(0 if ok >= args.min_frac * len(rows) else 1)


if __name__ == "__main__":
    main()
