"""Step 1 data: the correct tool for each MCPTox clean query.

- Servers whose clean queries line up 1:1 with their tools: gold = that tool.
- Other servers: ask Qwen3-32B-AWQ (thinking) with the server's clean prompt
  and take the tool it calls.
We also run the LLM on the aligned servers and report how often it agrees
with the free labels, as a check on the LLM labels.

  python scripts/label_clean.py
  -> results/clean/gold.json  [{server, query, gold, source}]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data import ROOT, load          # noqa: E402
from common.judge import parse_calls        # noqa: E402
from common.llm import LLM                  # noqa: E402


def main():
    bm = load()
    items = [(s, q) for s in bm.servers.values() for q in s.clean_queries]
    convs = [[{"role": "system", "content": bm.build_prompt(s.tools)}, {"role": "user", "content": q}]
             for s, q in items]
    llm = LLM("qwen3-32b-awq", max_model_len=16384)
    outs = llm.chat(convs, think=True)

    rows, agree, n_al = [], 0, 0
    for (s, q), o in zip(items, outs):
        calls = parse_calls(o)
        llm_tool = calls[0]["tool"] if calls and calls[0]["tool"] in {t.name for t in s.tools} else None
        if s.queries_aligned:
            gold = s.tools[s.clean_queries.index(q)].name
            n_al += 1
            agree += llm_tool == gold
            rows.append({"server": s.name, "query": q, "gold": gold, "source": "aligned", "llm": llm_tool})
        else:
            rows.append({"server": s.name, "query": q, "gold": llm_tool, "source": "llm", "llm": llm_tool})
    d = ROOT / "results" / "clean"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gold.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    usable = sum(r["gold"] is not None for r in rows)
    print(f"{len(rows)} clean queries, {usable} with a gold tool")
    print(f"LLM label agrees with aligned gold on {agree}/{n_al} = {agree / max(n_al, 1):.1%}")


if __name__ == "__main__":
    main()
