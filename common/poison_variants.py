"""Step 3: retriever-aware poisoned tools.

A variant rewrites each case's poisoned description so it ranks higher for
queries about the tool it attacks, while keeping the hidden instruction.
Variants are stored as {case_id: new description} in
results/poison_variants/<name>.json (made by scripts/make_poison_variants.py).

  mimic - prepend the target tool's own description to the poison (no LLM)
  llm   - an LLM rewrites the poison to read like the target tool's documentation
"""
from __future__ import annotations

import dataclasses
import json
import re

from common.data import ROOT, Benchmark, Case, Tool

VAR_DIR = ROOT / "results" / "poison_variants"


def named_targets(bm: Benchmark, case: Case) -> list[Tool]:
    """Host-server tools the poison mentions, in order of first mention."""
    text = case.poison.block
    found = []
    for t in bm.servers[case.server].tools:
        m = re.search(r"(?<![\w-])" + re.escape(t.name) + r"(?![\w-])", text)
        if m:
            found.append((m.start(), t))
    return [t for _, t in sorted(found, key=lambda x: x[0])]


def with_description(tool: Tool, description: str) -> Tool:
    """Same tool name and arguments, new description."""
    head, sep, rest = tool.block.partition("\nArguments:")
    new_block = f"Tool: {tool.name}\nDescription: {description.strip()}" + (sep + rest if sep else "\n\n\n")
    return dataclasses.replace(tool, block=new_block)


def apply_variant(bm: Benchmark, name: str) -> list[Case]:
    """Copies of bm.cases whose poison uses the variant's description."""
    if name in ("", "original"):
        return list(bm.cases)
    descs = json.loads((VAR_DIR / f"{name}.json").read_text())
    return [dataclasses.replace(c, poison=with_description(c.poison, descs[c.id])) if c.id in descs else c
            for c in bm.cases]
