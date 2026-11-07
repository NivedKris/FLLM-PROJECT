"""Sanity checks for retrieval (runs the small embedder on CPU)."""
import pytest
from common.data import load
from common.retrieval import Embedder, ToolIndex

@pytest.fixture(scope="module")
def setup():
    bm = load()
    return bm, ToolIndex(Embedder("bge-small"), bm.tools)

def test_tool_level_k(setup):
    bm, idx = setup
    q = idx.emb.queries(["Read the content of '/data/project/documents/report.txt'."])[0]
    hit = idx.retrieve(q, k=3)
    assert len(hit.tools) == 3 and "read_file" in [t.name for t in hit.tools]

def test_server_inside_reproduces_mcptox_prompt(setup):
    bm, idx = setup
    c = bm.cases[0]
    q = idx.emb.queries([c.query])[0]; p = idx.emb.docs([c.poison.name + ": " + c.poison.description])[0]
    hit = idx.retrieve(q, k=1, granularity="server", poison=c.poison, pvec=p, placement="inside", poison_index=c.poison_index)
    assert hit.tools == bm.case_tools(c)   # top-1 server = MCPTox's original prompt

def test_server_separate(setup):
    bm, idx = setup
    c = bm.cases[0]
    q = idx.emb.queries([c.query])[0]; p = idx.emb.docs(["x"])[0]
    hit = idx.retrieve(q, k=2, granularity="server", poison=c.poison, pvec=p, placement="separate")
    assert len(hit.unit_scores) == 2
