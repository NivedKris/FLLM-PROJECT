"""Clean-prompt reference calls (made by scripts/make_references.py)."""
import json

from common.data import ROOT

REF_DIR = ROOT / "results" / "references"


def ref_key(server: str, query: str) -> str:
    return f"{server}\t{query}"


def load_references(tag: str) -> dict:
    """{"<server>\\t<query>": {"raw": ..., "calls": [...]}}"""
    return json.loads((REF_DIR / f"{tag}.json").read_text())
