"""Clean-prompt reference calls: what a model does for each attack query when
the poisoned tool is NOT in the prompt (host server's tools only).

Used by the judge: an identical call under attack = the poison was ignored.

  python scripts/make_references.py --model qwen3-32b-awq
  -> results/references/qwen3-32b-awq.json   {"<server>\\t<query>": {"raw": ..., "calls": [...]}}
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import load            # noqa: E402
from common.judge import parse_calls    # noqa: E402
from common.llm import LLM              # noqa: E402
from common.refs import REF_DIR, ref_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b-awq")
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    bm = load()
    keys = sorted({(c.server, c.query) for c in bm.cases})
    print(f"{len(keys)} unique (server, query) pairs")
    convs = [[{"role": "system", "content": bm.build_prompt(bm.servers[s].tools)},
              {"role": "user", "content": q}] for s, q in keys]
    llm = LLM(args.model, max_model_len=16384)
    outs = llm.chat(convs, think=args.think)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}{'-think' if args.think else ''}"
    data = {ref_key(s, q): {"raw": o, "calls": parse_calls(o)} for (s, q), o in zip(keys, outs)}
    (REF_DIR / f"{tag}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"saved {len(data)} references; {sum(bool(v['calls']) for v in data.values())} contain a tool call")


if __name__ == "__main__":
    main()
