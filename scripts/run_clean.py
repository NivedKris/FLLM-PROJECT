"""Step 1: reproduce RAG-MCP - tool-selection accuracy and prompt size on clean queries.

Settings:
  server          the query's own server only (MCPTox's clean prompt; an oracle)
  all             all 362 tools in the prompt ("Blank Conditioning" in RAG-MCP)
  random-N        own server + N-1 random other servers (stress test, like the paper's)
  tool@k          RAG-MCP, tool-level top-k
  server@k        RAG-MCP, server-level top-k (the paper's setup at k=1)

Prompts longer than --long-tokens need Qwen3's YaRN context extension, so the
run is split in two passes (two processes, one vLLM engine each):
  python scripts/run_clean.py --model qwen3-8b [--think]          # normal prompts
  python scripts/run_clean.py --model qwen3-8b [--think] --long   # long prompts, YaRN on
  python scripts/run_clean.py --model qwen3-8b [--think] --summarize
  -> results/clean/<run>/{responses_short,responses_long}.jsonl, summary.json
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load                     # noqa: E402
from common.judge import parse_calls                   # noqa: E402
from common.retrieval import Embedder, ToolIndex       # noqa: E402

SETTINGS = (["server", "all", "random-5", "random-10", "random-20", "random-30"]
            + [f"{g}@{k}" for g in ("tool", "server") for k in (1, 3, 5)])
LONG_TOKENS = 24000



def run_dir(model, think, defense):
    return ROOT / "results" / "clean" / f"{model}{'-think' if think else ''}{'+defense' if defense else ''}"


def build_jobs(bm, gold, embedder, settings, defense=False):
    emb = Embedder(embedder)
    idx = ToolIndex(emb, bm.tools)
    qv = emb.queries([g["query"] for g in gold])
    names = list(bm.servers)
    filt = None
    if defense:
        from common.defense import Defense
        filt = Defense(emb)
        filt.precompute(bm.tools)
    jobs = []
    for gi, (g, q) in enumerate(zip(gold, qv)):
        for st in settings:
            if st == "server":
                tools = list(bm.servers[g["server"]].tools)
            elif st == "all":
                tools = bm.tools
            elif st.startswith("random-"):
                n = int(st.split("-")[1])
                rng = random.Random(f"{gi}-{n}")
                others = rng.sample([s for s in names if s != g["server"]], n - 1)
                chosen = others + [g["server"]]
                rng.shuffle(chosen)   # the right server lands at a random position
                tools = [t for s in chosen for t in bm.servers[s].tools]
            else:
                gran, k = st.split("@")
                tools = idx.retrieve(q, int(k), gran).tools
            if filt is not None:
                tools, _ = filt.filter(tools)
            jobs.append((gi, st, tools))
    return jobs


def summarize(d, gold):
    rows = []
    for f in ("responses_short.jsonl", "responses_long.jsonl"):
        if (d / f).exists():
            rows += [json.loads(line) for line in open(d / f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["setting"]].append(r)
    order = [s for s in SETTINGS if s in by] + [s for s in by if s not in SETTINGS]
    table = []
    print(f"{'setting':12s} {'n':>4} {'accuracy':>9} {'gold in prompt':>15} {'acc|gold in':>12} {'tools':>6} {'prompt tok':>11} {'compl tok':>10}")
    for st in order:
        rs = by[st]
        n = len(rs)
        acc = sum(r["correct"] for r in rs) / n
        gin = [r for r in rs if r["gold_in_prompt"]]
        t = {"setting": st, "n": n, "accuracy": acc, "gold_in_prompt": len(gin) / n,
             "acc_given_gold_in": sum(r["correct"] for r in gin) / len(gin) if gin else 0.0,
             "avg_tools": sum(r["n_tools"] for r in rs) / n,
             "avg_prompt_tokens": sum(r["prompt_tokens"] for r in rs) / n,
             "avg_completion_tokens": sum(r["completion_tokens"] for r in rs) / n}
        table.append(t)
        print(f"{st:12s} {n:4d} {acc:9.1%} {t['gold_in_prompt']:15.1%} {t['acc_given_gold_in']:12.1%} "
              f"{t['avg_tools']:6.1f} {t['avg_prompt_tokens']:11.0f} {t['avg_completion_tokens']:10.0f}")
    (d / "summary.json").write_text(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--defense", action="store_true")
    ap.add_argument("--embedder", default="bge-small")
    ap.add_argument("--settings", nargs="+", default=SETTINGS)
    ap.add_argument("--long", action="store_true", help="run only the long prompts, with YaRN")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=0, help="override context length (testing)")
    args = ap.parse_args()

    bm = load()
    gold = [g for g in json.loads((ROOT / "results" / "clean" / "gold.json").read_text()) if g["gold"]]
    if args.limit:
        gold = gold[:args.limit]
    d = run_dir(args.model, args.think, args.defense)
    d.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(d, gold)
        return

    from transformers import AutoTokenizer
    from common.llm import MODELS, count_for_model
    tok = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model))

    jobs = build_jobs(bm, gold, args.embedder, args.settings, args.defense)
    convs, key_of, ntok = [], {}, []
    for gi, st, tools in jobs:
        key = (bm.build_prompt(tools), gold[gi]["query"])
        if key not in key_of:
            cv = [{"role": "system", "content": key[0]}, {"role": "user", "content": key[1]}]
            n = count_for_model(args.model, tok, cv)
            key_of[key] = len(convs)
            convs.append(cv)
            ntok.append(n)
    want = [i for i, n in enumerate(ntok) if (n > LONG_TOKENS) == args.long]
    print(f"{len(jobs)} (query, setting) pairs -> {len(convs)} unique prompts; "
          f"{len(want)} in this pass ({'long, YaRN' if args.long else 'normal'})")
    if not want:
        return

    from common.llm import LLM, MODEL_CFG
    cfg = MODEL_CFG.get(args.model, {})
    if args.long:
        # Long prompts need >= 64k context: native (Llama 3.1) or YaRN (Qwen3).
        extra = {}
        if cfg.get("native_len", 0) < 65536:
            if not cfg.get("yarn"):
                print(f"{args.model} has a {cfg.get('native_len')} token context and no extension: "
                      f"skipping {len(want)} long prompts")
                return
            extra["hf_overrides"] = cfg["yarn"]
        llm = LLM(args.model, max_model_len=args.max_len or 65536, gpu_memory_utilization=args.gpu_mem, **extra)
    else:
        llm = LLM(args.model, max_model_len=args.max_len or min(32768, cfg.get("native_len", 32768)),
                  gpu_memory_utilization=args.gpu_mem)
    outs = dict(zip(want, llm.chat([convs[i] for i in want], think=args.think)))

    fname = "responses_long.jsonl" if args.long else "responses_short.jsonl"
    with open(d / fname, "w") as f:
        for gi, st, tools in jobs:
            i = key_of[(bm.build_prompt(tools), gold[gi]["query"])]
            if i not in outs:
                continue
            calls = parse_calls(outs[i])
            g = gold[gi]
            # Some servers share tool names (e.g. Commander and FileSystem both have
            # read_file), so a call only counts if the gold server's tool was in the prompt.
            gin = any(t.name == g["gold"] and t.server == g["server"] for t in tools)
            f.write(json.dumps({
                "query_idx": gi, "server": g["server"], "query": g["query"], "gold": g["gold"],
                "setting": st, "n_tools": len(tools), "gold_in_prompt": gin,
                "prompt_tokens": ntok[i], "completion_tokens": len(tok(outs[i], add_special_tokens=False)["input_ids"]),
                "called": calls[0]["tool"] if calls else None,
                "correct": gin and bool(calls) and calls[0]["tool"] == g["gold"],
                "response": outs[i],
            }, ensure_ascii=False) + "\n")
    print(f"saved {fname}")


if __name__ == "__main__":
    main()
