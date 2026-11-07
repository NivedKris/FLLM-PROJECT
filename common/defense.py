"""Step 4: a simple filter that drops suspicious tools before they reach the LLM.

A retrieved tool is flagged if its description
  (a) contains instruction-like text aimed at the agent ("you MUST", "ignore
      previous instructions", "before using X, call Y", "<IMPORTANT>", ...), or
  (b) names another tool in the prompt and tells the agent what to do with it, or
  (c) is a near-copy of another tool in the prompt (embedding similarity)
      and also contains some instruction-like text.

Thresholds are chosen so that few of the 362 clean MCPTox tools are flagged
(see scripts/eval_defense.py, which reports the false-positive rate).
"""
from __future__ import annotations

import re

import numpy as np

from common.data import Tool

INSTRUCTION_PATTERNS = [
    r"\bignore (all |the )?(previous|prior|above) instructions?",
    r"\bpriority is higher than\b",
    r"<\s*important\s*>",
    r"\byou (must|should|need to|have to)\b",
    r"\b(must|always|never)\b[^.]{0,60}\b(call|use|invoke|run|change|modify|set|replace|append|add|inject|redirect)\b",
    r"\bbefore (using|calling|running|executing|invoking)\b",
    r"\bregardless of\b",
    r"\b(do not|don't) (tell|inform|mention|notify)\b",
    r"\bsystem (alert|policy|requirement)\b",
    r"\bmandatory\b",
]
_INSTR = [re.compile(p, re.I) for p in INSTRUCTION_PATTERNS]
_ACTION = re.compile(r"\b(call|use|invoke|run|change|modify|set|replace|append|add|redirect|before|after|first)\b", re.I)


def instruction_hits(text: str) -> list[str]:
    return [p.pattern for p in _INSTR if p.search(text)]


def names_other_tool(tool: Tool, others: list[Tool]) -> list[str]:
    """Other tools in the prompt that this tool's description names next to an action word."""
    text = tool.description
    out = []
    for o in others:
        if o.name == tool.name:
            continue
        for m in re.finditer(r"(?<![\w-])" + re.escape(o.name) + r"(?![\w-])", text):
            window = text[max(0, m.start() - 80): m.end() + 80]
            if _ACTION.search(window):
                out.append(o.name)
                break
    return out


class Defense:
    def __init__(self, embedder, min_instruction_hits: int = 2, dup_threshold: float = 0.93,
                 use_names: bool = True):
        self.emb = embedder
        self.min_hits = min_instruction_hits
        self.dup = dup_threshold
        self.use_names = use_names
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, t: Tool) -> np.ndarray:
        key = t.block
        if key not in self._cache:
            from common.retrieval import tool_text
            self._cache[key] = self.emb.docs([tool_text(t)])[0]
        return self._cache[key]

    def precompute(self, tools: list[Tool]):
        from common.retrieval import tool_text
        new = [t for t in {t.block: t for t in tools}.values() if t.block not in self._cache]
        if new:
            for t, v in zip(new, self.emb.docs([tool_text(t) for t in new])):
                self._cache[t.block] = v

    def reasons(self, tool: Tool, context: list[Tool]) -> list[str]:
        r = []
        hits = instruction_hits(tool.description)
        if len(hits) >= self.min_hits:
            r.append(f"instructions({len(hits)})")
        if self.use_names:
            named = names_other_tool(tool, context)
            if named and hits:          # names another tool AND sounds like an instruction
                r.append("names:" + ",".join(named[:3]))
        if hits:   # near-copies only count with instruction-like text (legit sibling tools look alike)
            v = self._vec(tool)
            for o in context:
                if o.name != tool.name and float(v @ self._vec(o)) >= self.dup:
                    r.append(f"near-copy:{o.name}")
                    break
        return r

    def filter(self, tools: list[Tool]) -> tuple[list[Tool], list[Tool]]:
        """(kept, removed) for one prompt's tool list."""
        kept, removed = [], []
        for t in tools:
            (removed if self.reasons(t, tools) else kept).append(t)
        return kept, removed
