"""Embedding retrieval over MCP tools (our RAG-MCP implementation).

Two granularities:
  tool   - every tool is one index entry; return the top-k tools.
  server - every server is one entry; return all tools of the top-k servers.
           A server's score is the best score of any of its tools ("max-sim"),
           because full servers are too long for a 512-token embedding model.

The poisoned tool for a case can be added in three ways (see `retrieve`):
  tool level            -> one extra entry
  server level, "inside"   -> part of its host server's entry (MCPTox setting)
  server level, "separate" -> its own one-tool "malicious server" entry
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.data import Tool

EMBEDDERS = {
    # name: (HF id, query prefix, document prefix)
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-large": ("BAAI/bge-large-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
}


def tool_text(t: Tool) -> str:
    """What we embed for a tool: its name and description."""
    return f"{t.name}: {t.description}"


class Embedder:
    def __init__(self, name: str = "bge-small", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        hf, self.qp, self.dp = EMBEDDERS[name]
        self.name = name
        self.model = SentenceTransformer(hf, device=device)

    def _enc(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)

    def queries(self, qs: list[str]) -> np.ndarray:
        return self._enc([self.qp + q for q in qs])

    def docs(self, ds: list[str]) -> np.ndarray:
        return self._enc([self.dp + d for d in ds])


@dataclass
class Hit:
    tools: list[Tool]        # tools that go into the prompt, in rank order
    unit_scores: list[float]  # score of each retrieved entry (tool or server)


class ToolIndex:
    """Index over the clean tool pool; poisoned tools are added per query."""

    def __init__(self, embedder: Embedder, tools: list[Tool]):
        self.emb = embedder
        self.tools = tools
        self.vecs = embedder.docs([tool_text(t) for t in tools])
        self.servers = sorted({t.server for t in tools}, key=[t.server for t in tools].index)
        self.server_of = np.array([self.servers.index(t.server) for t in tools])

    def retrieve(self, qvec: np.ndarray, k: int, granularity: str = "tool",
                 poison: Tool | None = None, pvec: np.ndarray | None = None,
                 placement: str = "inside", poison_index: int | None = None) -> Hit:
        """Top-k retrieval for one query vector.

        poison/pvec: optional poisoned tool and its embedding.
        placement: for server level, "inside" (poison belongs to its host
                   server's entry) or "separate" (poison is its own entry).
        poison_index: position of the poison within its host server's tools
                      for "inside" (default: last), so prompts match MCPTox.
        """
        scores = self.vecs @ qvec
        pscore = float(pvec @ qvec) if poison is not None else None

        if granularity == "tool":
            cand = [(float(s), [t]) for s, t in zip(scores, self.tools)]
            if poison is not None:
                cand.append((pscore, [poison]))
        elif granularity == "server":
            cand = []
            for si, sname in enumerate(self.servers):
                idx = np.where(self.server_of == si)[0]
                best = float(scores[idx].max())
                tools = [self.tools[i] for i in idx]
                if poison is not None and placement == "inside" and poison.server == sname:
                    best = max(best, pscore)
                    pos = len(tools) if poison_index is None else poison_index
                    tools = tools[:pos] + [poison] + tools[pos:]
                cand.append((best, tools))
            if poison is not None and placement == "separate":
                cand.append((pscore, [poison]))
        else:
            raise ValueError(granularity)

        cand.sort(key=lambda x: -x[0])
        top = cand[:k]
        return Hit([t for _, ts in top for t in ts], [s for s, _ in top])
