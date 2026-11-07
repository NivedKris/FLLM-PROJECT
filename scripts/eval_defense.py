"""Step 4 (no LLM): how well does the filter separate poisoned from clean tools?

For every clean tool and every poisoned tool (original and any rewritten
variants), check the filter with the host server's tools as context.

  python scripts/eval_defense.py [--variants mimic llm]
  -> results/defense/filter_eval.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load               # noqa: E402
from common.defense import Defense               # noqa: E402
from common.poison_variants import apply_variant  # noqa: E402
from common.retrieval import Embedder            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-small")
    ap.add_argument("--variants", nargs="*", default=[])
    args = ap.parse_args()

    bm = load()
    emb = Embedder(args.embedder)
    configs = {
        "instructions>=1": dict(min_instruction_hits=1, dup_threshold=2.0, use_names=False),
        "instructions>=2": dict(min_instruction_hits=2, dup_threshold=2.0, use_names=False),
        "names+instr": dict(min_instruction_hits=99, dup_threshold=2.0, use_names=True),
        "near-copy>=0.93": dict(min_instruction_hits=99, dup_threshold=0.93, use_names=False),
        "near-copy>=0.85": dict(min_instruction_hits=99, dup_threshold=0.85, use_names=False),
        "full (default)": dict(),
    }
    sets = {"original": bm.cases}
    for v in args.variants:
        sets[v] = apply_variant(bm, v)

    out = {}
    print(f"{'filter':18s} {'clean FP':>9}" + "".join(f" {('det:' + s):>14}" for s in sets))
    for name, cfg in configs.items():
        d = Defense(emb, **cfg)
        d.precompute(bm.tools + [c.poison for cs in sets.values() for c in cs])
        fp = [t for t in bm.tools if d.reasons(t, bm.servers[t.server].tools)]
        row = {"clean_fp": len(fp) / len(bm.tools), "clean_fp_tools": [f"{t.server}/{t.name}" for t in fp]}
        line = f"{name:18s} {row['clean_fp']:9.1%}"
        for s, cs in sets.items():
            det = collections.Counter()
            for c in cs:
                hit = bool(d.reasons(c.poison, bm.servers[c.server].tools + [c.poison]))
                det["all"] += hit
                det[c.paradigm] += hit
            n = collections.Counter(c.paradigm for c in cs)
            row[f"detect_{s}"] = det["all"] / len(cs)
            row[f"detect_{s}_by_paradigm"] = {p: det[p] / n[p] for p in sorted(n)}
            line += f" {row[f'detect_{s}']:14.1%}"
        out[name] = row
        print(line)
    print("\nclean tools flagged by the default filter:", out["full (default)"]["clean_fp_tools"][:20])
    dd = ROOT / "results" / "defense"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "filter_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
