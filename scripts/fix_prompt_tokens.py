"""Recount prompt_tokens in an attack run's responses.jsonl.

Runs made before the token-count fix stored a wrong value (2). Retrieval is
deterministic, so we rebuild each row's prompt and count it again.

  python scripts/fix_prompt_tokens.py --run qwen3-8b
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.data import ROOT, load              # noqa: E402
from common.llm import MODELS, count_for_model  # noqa: E402
from run_attack import build_jobs               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen3-8b", help="tokenizer to count with")
    args = ap.parse_args()

    d = ROOT / "results" / "attack" / args.run
    f = d / "responses.jsonl"
    rows = [json.loads(line) for line in open(f)]
    if all(r["prompt_tokens"] > 10 for r in rows):
        print("token counts already look right; nothing to do")
        return
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    bm = load()
    jobs = build_jobs(bm, meta.get("embedder", "bge-small"), sorted({r["setting"] for r in rows},
                      key=[r["setting"] for r in rows].index), meta.get("limit", 0),
                      meta.get("variant", ""), meta.get("defense", False))
    jm = {(c.id, st): tools for c, st, tools, _ in jobs}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model))
    query_of = {c.id: c.query for c in bm.cases}
    counted = {}
    for r in rows:
        tools = jm[(r["case"], r["setting"])]
        assert [t.name for t in tools] == r["tools"], f"tool list mismatch for {r['case']} {r['setting']}"
        key = (bm.build_prompt(tools), query_of[r["case"]])
        if key not in counted:
            counted[key] = count_for_model(args.model, tok, [{"role": "system", "content": key[0]},
                                                   {"role": "user", "content": key[1]}])
        r["prompt_tokens"] = counted[key]
    tmp = f.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(f)
    print(f"recounted prompt tokens for {len(rows)} rows in {f}")


if __name__ == "__main__":
    main()
