"""Retrieval-only statistics (no LLM needed).

1. Clean retrieval accuracy: on servers whose clean queries line up 1:1 with
   their tools, is the correct tool in the top-k?
2. Attack retrieval: for each MCPTox case, is the poisoned tool retrieved, and
   is it retrieved together with a legitimate tool that the poison names
   ("co-retrieval", needed for the attack to work)?

  python scripts/retrieval_stats.py --embedder bge-small
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load                     # noqa: E402
from common.retrieval import Embedder, ToolIndex, tool_text  # noqa: E402

KS = [1, 3, 5]
SETTINGS = [("tool", None), ("server", "inside"), ("server", "separate")]


def targets(bm, case):
    """Legitimate tools of the host server that the poison text mentions by name."""
    text = case.poison.block
    return {t.name for t in bm.servers[case.server].tools
            if re.search(r"(?<![\w-])" + re.escape(t.name) + r"(?![\w-])", text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-small")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()
    global KS
    KS = args.ks

    bm = load()
    idx = ToolIndex(Embedder(args.embedder), bm.tools)
    out = {"embedder": args.embedder}

    # 1. clean retrieval accuracy
    gold = [(q, t) for s in bm.servers.values() if s.queries_aligned for q, t in zip(s.clean_queries, s.tools)]
    qv = idx.emb.queries([q for q, _ in gold])
    print(f"Clean retrieval accuracy ({len(gold)} queries with known correct tool):")
    out["clean"] = {}
    for k in KS:
        tool_hits = sum(t in idx.retrieve(v, k).tools for v, (_, t) in zip(qv, gold)) / len(gold)
        srv_hits = sum(t in idx.retrieve(v, k, "server").tools for v, (_, t) in zip(qv, gold)) / len(gold)
        out["clean"][k] = {"tool": tool_hits, "server": srv_hits}
        print(f"  k={k}: tool-level {tool_hits:.1%}   server-level {srv_hits:.1%}")

    # 2. attack retrieval
    cases = bm.cases
    qv = idx.emb.queries([c.query for c in cases])
    pv = idx.emb.docs([tool_text(c.poison) for c in cases])
    tg = [targets(bm, c) for c in cases]
    print(f"\nPoison named target found for {sum(bool(t) for t in tg)}/{len(cases)} cases")
    print(f"\n{'setting':18s} {'k':>2} {'poison retrieved':>17} {'poison+target':>14}   by paradigm (poison+target)")
    out["attack"] = []
    for gran, place in SETTINGS:
        for k in KS:
            ret = co = 0
            byp = collections.defaultdict(lambda: [0, 0])
            for c, q, p, t in zip(cases, qv, pv, tg):
                hit = idx.retrieve(q, k, gran, poison=c.poison, pvec=p, placement=place or "inside",
                                   poison_index=c.poison_index)
                names = {x.name for x in hit.tools if not x.poisoned}
                r = c.poison in hit.tools
                both = r and (bool(t & names) if t else r)   # no named target: poison alone counts
                ret += r; co += both
                byp[c.paradigm][0] += both; byp[c.paradigm][1] += 1
            name = gran + (f"/{place}" if place else "")
            bp = "  ".join(f"{p[-1]}:{a / b:.0%}" for p, (a, b) in sorted(byp.items()))
            print(f"{name:18s} {k:2d} {ret / len(cases):17.1%} {co / len(cases):14.1%}   {bp}")
            out["attack"].append({"setting": name, "k": k, "poison_retrieved": ret / len(cases),
                                  "co_retrieved": co / len(cases)})

    d = ROOT / "results" / "retrieval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.embedder}{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
