"""Step 2-4: run MCPTox attacks through RAG-MCP and save raw model responses.

Settings (each case's query is run once per setting):
  original           - MCPTox's own prompt: host server's tools + poison (no retrieval)
  tool@k             - tool-level retrieval, poison is one more index entry
  server-inside@k    - server-level retrieval, poison is part of its host server
  server-separate@k  - server-level retrieval, poison is its own "server"

Options:
  --variant mimic|llm  use retriever-aware poisoned descriptions (Step 3)
  --defense            drop flagged tools after retrieval, before the LLM (Step 4)

Identical prompts are only sent to the model once.

  python scripts/run_attack.py --model qwen3-8b --think [--variant llm] [--defense]
  -> results/attack/<model>[-think][+variant][+defense]/responses.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load                           # noqa: E402
from common.poison_variants import apply_variant             # noqa: E402
from common.retrieval import Embedder, ToolIndex, tool_text  # noqa: E402

SETTINGS = ["original"] + [f"{s}@{k}" for s in ["tool", "server-inside", "server-separate"] for k in (1, 3, 5)]


def run_name(model: str, think: bool, variant: str = "", defense: bool = False,
             embedder: str = "bge-small", seed: int = 0) -> str:
    return (f"{model}{'-think' if think else ''}{'+' + variant if variant else ''}{'+defense' if defense else ''}"
            f"{'+' + embedder if embedder != 'bge-small' else ''}{f'+seed{seed}' if seed else ''}")


def build_jobs(bm, embedder: str, settings: list[str], limit: int = 0, variant: str = "",
               defense: bool = False):
    """[(case, setting, tool list, removed tool names)] for every case and setting."""
    cases = apply_variant(bm, variant)
    cases = cases[:limit] if limit else cases
    emb = Embedder(embedder)
    idx = ToolIndex(emb, bm.tools)
    qv = emb.queries([c.query for c in cases])
    pv = emb.docs([tool_text(c.poison) for c in cases])
    filt = None
    if defense:
        from common.defense import Defense
        filt = Defense(emb)
        filt.precompute(bm.tools + [c.poison for c in cases])
    jobs = []
    for c, q, p in zip(cases, qv, pv):
        for st in settings:
            if st == "original":
                tools = bm.case_tools(c)
            else:
                name, k = st.split("@")
                gran, _, place = name.partition("-")
                tools = idx.retrieve(q, int(k), gran, poison=c.poison, pvec=p,
                                     placement=place or "inside", poison_index=c.poison_index).tools
            removed = []
            if filt is not None:
                tools, rem = filt.filter(tools)
                removed = [t.name for t in rem]
            jobs.append((c, st, tools, removed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--embedder", default="bge-small")
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--variant", default="", help="poison variant (mimic / llm); default: MCPTox original")
    ap.add_argument("--defense", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (for testing)")
    ap.add_argument("--tag", default="", help="extra suffix for the run folder (e.g. smoke)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (matters for reasoning runs)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=32768, help="context length (longest prompt ~17.4k + 8k think)")
    args = ap.parse_args()

    bm = load()
    jobs = build_jobs(bm, args.embedder, args.settings, args.limit, args.variant, args.defense)

    convs, key_of = [], {}
    for c, st, tools, _ in jobs:
        key = (bm.build_prompt(tools), c.query)
        if key not in key_of:
            key_of[key] = len(convs)
            convs.append([{"role": "system", "content": key[0]}, {"role": "user", "content": c.query}])
    print(f"{len(jobs)} (case, setting) pairs -> {len(convs)} unique prompts")

    from common.llm import LLM
    llm = LLM(args.model, max_model_len=args.max_len, gpu_memory_utilization=args.gpu_mem)
    outs = llm.chat(convs, think=args.think, seed=args.seed)
    ntok = [llm.count_tokens(cv) for cv in convs]

    d = ROOT / "results" / "attack" / (run_name(args.model, args.think, args.variant, args.defense, args.embedder,
                                                args.seed) + (f"+{args.tag}" if args.tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(vars(args), indent=1))
    with open(d / "responses.jsonl", "w") as f:
        for c, st, tools, removed in jobs:
            i = key_of[(bm.build_prompt(tools), c.query)]
            f.write(json.dumps({
                "case": c.id, "setting": st, "paradigm": c.paradigm,
                "tools": [t.name for t in tools], "removed": removed,
                "poison_in_prompt": c.poison in tools,
                "prompt_tokens": ntok[i], "response": outs[i],
            }, ensure_ascii=False) + "\n")
    print(f"saved {len(jobs)} rows to {d / 'responses.jsonl'}")


if __name__ == "__main__":
    main()
