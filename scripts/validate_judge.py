"""Check our attack judge against MCPTox's stored labels.

Runs the judge on MCPTox's stored responses for a few models and reports
how often our label matches theirs (4-class and Success-vs-not).

  python scripts/validate_judge.py --judge qwen3-8b
  python scripts/validate_judge.py --judge qwen3-32b-awq --think
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load            # noqa: E402
from common.judge import judge_many           # noqa: E402
from common.llm import LLM                    # noqa: E402
from common.refs import load_references, ref_key  # noqa: E402

# Offline-run labels (capitalised names) and online-run labels (lowercase names).
DEFAULT_MODELS = ["Qwen3-8b-Think", "Qwen3-8b-No-Think", "qwen3-8b_Think", "qwen3-32b_Think", "gpt-4o-mini"]
VALID = {"Success", "Failure-Direct Execution", "Failure-Ignored", "Failure-Refused"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3-8b")
    ap.add_argument("--think", action="store_true", help="let the judge reason (slower)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--refs", default="", help="reference tag in results/references/ (e.g. qwen3-32b-awq)")
    ap.add_argument("--limit", type=int, default=0, help="random cases per model (0 = all)")
    args = ap.parse_args()

    bm = load()
    items, meta = [], []
    for m in args.models:
        cases = [c for c in bm.cases if m in c.responses and c.labels.get(m) in VALID]
        if args.limit:  # random sample across all servers (fixed seed)
            cases = random.Random(0).sample(cases, min(args.limit, len(cases)))
        for c in cases:
            items.append((c, c.responses[m]))
            meta.append((m, c))
    print(f"{len(items)} labelled responses to judge")

    llm = LLM(args.judge, max_model_len=8192)
    refs = None
    if args.refs:
        R = load_references(args.refs)
        refs = [R[ref_key(c.server, c.query)]["calls"] for c, _ in items]
    ours = judge_many(bm, items, llm=llm, think=args.think, references=refs)

    out_dir = ROOT / "results" / "judge_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.judge}{'-think' if args.think else ''}{'-ref-' + args.refs if args.refs else ''}"
    rows = [{"model": m, "case": c.id, "paradigm": c.paradigm, "theirs": c.labels[m], "ours": o}
            for (m, c), o in zip(meta, ours)]
    (out_dir / f"{tag}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    print(f"\nJudge: {tag}")
    print(f"{'model':22s} {'n':>5} {'4-class':>8} {'succ-agree':>10} {'ASR theirs':>10} {'ASR ours':>9} {'kappa':>6}")
    for m in args.models + ["ALL"]:
        rs = [r for r in rows if m in ("ALL", r["model"])]
        if not rs:
            continue
        n = len(rs)
        acc4 = sum(r["theirs"] == r["ours"] for r in rs) / n
        t = [r["theirs"] == "Success" for r in rs]
        o = [r["ours"] == "Success" for r in rs]
        agree = sum(a == b for a, b in zip(t, o)) / n
        pt, po = sum(t) / n, sum(o) / n
        pe = pt * po + (1 - pt) * (1 - po)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
        print(f"{m:22s} {n:5d} {acc4:8.1%} {agree:10.1%} {pt:10.1%} {po:9.1%} {kappa:6.2f}")

    print("\nConfusion (rows = MCPTox label, cols = ours), all models:")
    conf = collections.Counter((r["theirs"], r["ours"]) for r in rows)
    cols = ["Success", "Failure-Direct Execution", "Failure-Ignored", "Failure-Refused", "None"]
    print(" " * 26 + "".join(f"{c.split('-')[-1][:9]:>10}" for c in cols))
    for a in cols[:4]:
        print(f"{a:26s}" + "".join(f"{conf[(a, b)]:10d}" for b in cols))


if __name__ == "__main__":
    main()
