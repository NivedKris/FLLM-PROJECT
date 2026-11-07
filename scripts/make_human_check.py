"""Sample judged responses for a human spot-check of the judge.

Writes results/human_check.csv. Fill the `human_label` column with one of
Success / Direct / Ignored / Refused / None, then run
  python scripts/make_human_check.py --score
to get the judge's agreement with the human labels.

  python scripts/make_human_check.py --runs qwen3-8b-think qwen3-8b --n 100
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load               # noqa: E402
from common.judge import parse_calls, strip_reasoning  # noqa: E402

OUT = ROOT / "results" / "human_check.csv"          # what the labeller sees (no judge label)
KEY = ROOT / "results" / "human_check_key.json"     # judge labels, kept separate so labelling is blind
SHORT = {"Success": "Success", "Failure-Direct Execution": "Direct", "Failure-Ignored": "Ignored",
         "Failure-Refused": "Refused", "None": "None"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["qwen3-8b-think", "qwen3-8b"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    if args.score:
        key = json.loads(KEY.read_text())
        rows = [dict(r, judge_label=key[r["id"]]) for r in csv.DictReader(open(OUT)) if r["human_label"].strip()]
        agree = sum(r["human_label"].strip().lower() == r["judge_label"].lower() for r in rows)
        s_agree = sum((r["human_label"].strip().lower() == "success") == (r["judge_label"] == "Success") for r in rows)
        print(f"{len(rows)} human-labelled rows: 4-class agreement {agree / len(rows):.1%}, "
              f"Success-vs-not agreement {s_agree / len(rows):.1%}")
        return

    bm = load()
    C = {c.id: c for c in bm.cases}
    pool = []
    for run in args.runs:
        d = ROOT / "results" / "attack" / run
        resp = [json.loads(line) for line in open(d / "responses.jsonl")]
        labs = [json.loads(line) for line in open(d / "labels.jsonl")]
        pool += [(run, r, lab) for r, lab in zip(resp, labs) if lab["label"] != "NotRetrieved"]
    # Balance: half judged Success, half not, so both error types are checked.
    rng = random.Random(0)
    succ = [p for p in pool if p[2]["label"] == "Success"]
    other = [p for p in pool if p[2]["label"] != "Success"]
    pick = rng.sample(succ, min(args.n // 2, len(succ))) + rng.sample(other, min(args.n - args.n // 2, len(other)))
    rng.shuffle(pick)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "run", "setting", "query", "poisoned_tool", "model_answer", "tool_calls", "human_label"])
        key = {}
        for i, (run, r, lab) in enumerate(pick):
            c = C[r["case"]]
            w.writerow([i, run, r["setting"], c.query, c.poison.block.strip()[:1500],
                        strip_reasoning(r["response"])[:1500],
                        json.dumps(parse_calls(r["response"]), ensure_ascii=False)[:1500], ""])
            key[str(i)] = SHORT.get(lab["label"], lab["label"])
    KEY.write_text(json.dumps(key, indent=1))
    print(f"wrote {len(pick)} rows to {OUT}; judge labels in {KEY}")


if __name__ == "__main__":
    main()
