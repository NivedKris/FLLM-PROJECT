"""Check that prompts we rebuild match MCPTox's original prompts exactly."""
import json
from common.data import DEFAULT_PATH, load

def test_rebuild_matches_original():
    # 362 tool blocks: MCPTox says 353 because Apify's tool_names lists 7 of its 16 prompt tools.
    bm = load()
    raw = json.loads(DEFAULT_PATH.read_text())["servers"]
    assert len(bm.servers) == 45 and len(bm.tools) == 362 and len(bm.cases) == 1348
    for name, s in bm.servers.items():
        assert bm.build_prompt(s.tools) == raw[name]["clean_system_promot"]
    originals = {}
    for name, s in raw.items():
        for i, inst in enumerate(s["malicious_instance"]):
            for d in inst["datas"]:
                originals[f"{name}/{i}/{d['id']}"] = d["system"]
    assert len(originals) == 1348, "case ids must be unique"
    for c in bm.cases:
        assert bm.build_prompt(bm.case_tools(c)) == originals[c.id], c.id
