"""Load the MCPTox benchmark into simple Python objects.

Every MCPTox attack prompt is the server's clean prompt with one extra
(poisoned) tool block inserted. We split prompts into tool blocks so we can
rebuild any prompt we want: all tools, a retrieved subset, with or without
the poisoned tool.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "MCPTox-Benchmark" / "response_all.json"

HEADER = "You are a helpful assistant with access to these tools:\n\n\n"
FOOTER_START = "Choose the appropriate tool based on the user's question"


@dataclass
class Tool:
    server: str
    name: str
    block: str            # full "Tool: ...\nDescription: ...\nArguments: ..." text
    poisoned: bool = False

    @property
    def description(self) -> str:
        """Text after 'Description:' and before 'Arguments:' (used for retrieval)."""
        m = re.search(r"Description:(.*?)(?:\nArguments:|$)", self.block, re.S)
        return (m.group(1) if m else self.block).strip()


@dataclass
class Server:
    name: str
    tools: list[Tool]
    clean_queries: list[str]

    @property
    def queries_aligned(self) -> bool:
        """True when there is exactly one clean query per tool, in tool order."""
        return len(self.clean_queries) == len(self.tools)


@dataclass
class Case:
    """One MCPTox poisoned test case."""
    id: str                # "<server>/<instance>/<data id>", unique
    server: str
    query: str
    poison: Tool
    poison_index: int      # where the poison sat in the original tool list
    paradigm: str          # Template-1 / Template-2 / Template-3
    risk: str
    risk_description: str  # short note from MCPTox (often in Chinese)
    labels: dict[str, str] = field(default_factory=dict)     # model -> stored MCPTox label
    responses: dict[str, str] = field(default_factory=dict)  # model -> stored raw response


@dataclass
class Benchmark:
    header: str
    footer: str
    servers: dict[str, Server]
    cases: list[Case]

    @property
    def tools(self) -> list[Tool]:
        return [t for s in self.servers.values() for t in s.tools]

    def build_prompt(self, tools: list[Tool]) -> str:
        """System prompt in MCPTox's exact format for the given tool list."""
        return self.header + "".join(t.block for t in tools) + self.footer

    def case_tools(self, case: Case) -> list[Tool]:
        """MCPTox's original setting: the host server's tools + poison at its original spot."""
        tools = list(self.servers[case.server].tools)
        tools.insert(case.poison_index, case.poison)
        return tools


def split_tool_blocks(prompt: str) -> tuple[str, list[str], str]:
    """Split a system prompt into (header, tool blocks, footer).

    Each block keeps its trailing blank lines so that
    header + ''.join(blocks) + footer == prompt exactly.
    """
    start = prompt.find("Tool: ")
    end = prompt.find(FOOTER_START)
    header, body, footer = prompt[:start], prompt[start:end], prompt[end:]
    starts = [m.start() for m in re.finditer(r"^Tool: ", body, re.M)]
    blocks = [body[a:b] for a, b in zip(starts, starts[1:] + [len(body)])]
    return header, blocks, footer


def _tool_name(block: str) -> str:
    return block.split("\n", 1)[0][len("Tool: "):].strip()


def load(path: str | Path = DEFAULT_PATH) -> Benchmark:
    raw = json.loads(Path(path).read_text())
    servers: dict[str, Server] = {}
    cases: list[Case] = []
    header = footer = None

    for sname, s in raw["servers"].items():
        h, blocks, f = split_tool_blocks(s["clean_system_promot"])
        header, footer = header or h, footer or f
        assert h == header and f == footer, f"unexpected prompt format in {sname}"
        tools = [Tool(sname, _tool_name(b), b) for b in blocks]
        servers[sname] = Server(sname, tools, list(s["clean_querys"]))
        clean_names = [t.name for t in tools]

        for inst_i, inst in enumerate(s["malicious_instance"]):
            for d in inst["datas"]:
                _, cblocks, _ = split_tool_blocks(d["system"])
                names = [_tool_name(b) for b in cblocks]
                # The poison is the one block whose name is not a clean tool
                # (fall back to position if a poison reuses a clean name).
                extra = [i for i, n in enumerate(names) if n not in clean_names]
                if len(extra) != 1:
                    extra = [i for i, (a, b) in enumerate(zip(names, clean_names + [None])) if a != b][:1]
                pi = extra[0]
                assert len(cblocks) == len(tools) + 1
                poison = Tool(sname, names[pi], cblocks[pi], poisoned=True)

                labels, responses = dict(d.get("label") or {}), dict(d.get("response") or {})
                online = d.get("online_result") or {}
                for lab in online.get("labeled_model_results") or []:
                    labels.update(lab)
                for resp in online.get("backend_model_responses") or []:
                    responses.update(resp)

                cases.append(Case(
                    id=f"{sname}/{inst_i}/{d['id']}",
                    server=sname,
                    query=d["query"],
                    poison=poison,
                    poison_index=pi,
                    paradigm=inst["metadata"]["paradigm"],
                    risk=inst["metadata"]["security risk"],
                    risk_description=inst.get("security_risk_description", ""),
                    labels={k: v for k, v in labels.items() if v is not None},
                    responses={k: v for k, v in responses.items() if isinstance(v, str)},
                ))
    return Benchmark(header, footer, servers, cases)


if __name__ == "__main__":
    import collections
    bm = load()
    print(f"servers={len(bm.servers)} tools={len(bm.tools)} cases={len(bm.cases)}")
    print("paradigms:", collections.Counter(c.paradigm for c in bm.cases))
    print("servers with aligned clean queries:", sum(s.queries_aligned for s in bm.servers.values()))
