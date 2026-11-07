"""Thin wrapper around vLLM offline batch inference."""
from __future__ import annotations

import os
import sys

# vLLM compiles kernels with `ninja`, which lives in the venv's bin/.
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
# Scripts load the embedding model first, which initialises CUDA; a forked
# vLLM engine process then fails with "CUDA error: initialization error".
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# name -> settings.
#   think_mode: "switch" = Qwen3 enable_thinking on/off; "always" = model always
#               reasons (<think>...</think>); "none" = no reasoning mode.
#   think / plain: sampling for reasoning / non-reasoning generation.
#   native_len: context length without extensions.
#   yarn: hf_overrides that extend the context with YaRN (Qwen3 only).
_QWEN_YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768,
                                  "rope_theta": 1000000},
              "max_position_embeddings": 65536}
_QWEN_THINK = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192)
_GREEDY = dict(temperature=0.0, max_tokens=1024)

MODEL_CFG = {
    "qwen3-0.6b": dict(hf="Qwen/Qwen3-0.6B", think_mode="switch", think=_QWEN_THINK, plain=_GREEDY,
                       native_len=40960, yarn=_QWEN_YARN),
    "qwen3-8b": dict(hf="Qwen/Qwen3-8B", think_mode="switch", think=_QWEN_THINK, plain=_GREEDY,
                     native_len=40960, yarn=_QWEN_YARN),
    "qwen3-32b-awq": dict(hf="Qwen/Qwen3-32B-AWQ", think_mode="switch", think=_QWEN_THINK, plain=_GREEDY,
                          native_len=40960, yarn=_QWEN_YARN),
    "mistral-7b": dict(hf="mistralai/Mistral-7B-Instruct-v0.3", think_mode="none", plain=_GREEDY,
                       native_len=32768, kw=dict(tokenizer_mode="hf")),
    "llama-3.1-8b": dict(hf="meta-llama/Llama-3.1-8B-Instruct", think_mode="none", plain=_GREEDY,
                         native_len=131072),
    # Microsoft's recommended sampling for Phi-4-reasoning. Its chat template
    # always uses its own system prompt and DROPS ours, so our system prompt
    # (the tool list) is moved into the user message instead.
    # FP8 weights: in bf16 the 14B model leaves only ~57k tokens of KV cache on the
    # A40, so long reasoning outputs run a few at a time (~58 h per attack run).
    # max_tokens 4096 (not 12000): normal answers finish in < 1.5k tokens, while
    # ~1 in 8 outputs loops until the limit and never answers; at 12000 those loops
    # were ~70% of all generated tokens.
    "phi-4-reasoning": dict(hf="microsoft/Phi-4-reasoning", think_mode="always",
                            think=dict(temperature=0.8, top_p=0.95, top_k=50, max_tokens=4096),
                            native_len=32768, system_to_user=True, kw=dict(quantization="fp8")),
}
MODELS = {k: v["hf"] for k, v in MODEL_CFG.items()}


class LLM:
    def __init__(self, model: str = "qwen3-8b", max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.90, **kw):
        from vllm import LLM as _VLLM
        self.name = model
        self.cfg = MODEL_CFG.get(model, dict(hf=model, think_mode="none", plain=_GREEDY, native_len=32768))
        self.llm = _VLLM(self.cfg["hf"], max_model_len=max_model_len,
                         gpu_memory_utilization=gpu_memory_utilization, seed=0,
                         **{**self.cfg.get("kw", {}), **kw})
        self.tokenizer = self.llm.get_tokenizer()

    def chat(self, conversations: list[list[dict]], think: bool = False,
             max_tokens: int | None = None, temperature: float | None = None, seed: int = 0) -> list[str]:
        """Run a batch of chats; returns the raw generated text (incl. <think> if any).

        Non-reasoning generation is greedy. Reasoning uses each model's
        recommended sampling (greedy decoding makes Qwen3 thinking loop).
        `think` is ignored for models without a switch: "always" models
        always reason, "none" models never do.
        """
        from vllm import SamplingParams
        mode = self.cfg["think_mode"]
        reasoning = mode == "always" or (mode == "switch" and think)
        params = dict(self.cfg["think"] if reasoning else self.cfg["plain"])
        if max_tokens:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        sp = SamplingParams(seed=seed, **params)

        if self.cfg.get("system_to_user"):
            conversations = [system_into_user(cv) for cv in conversations]
        # One prompt longer than the context window would crash the whole batch:
        # skip those (empty response) and say so.
        limit = self.llm.llm_engine.model_config.max_model_len
        ok = [i for i, cv in enumerate(conversations) if count_chat_tokens(self.tokenizer, cv) < limit]
        if len(ok) < len(conversations):
            print(f"WARNING: skipping {len(conversations) - len(ok)} prompts longer than {limit} tokens", flush=True)
        kwargs = {"chat_template_kwargs": {"enable_thinking": think}} if mode == "switch" else {}
        outs = self.llm.chat([conversations[i] for i in ok], sp, use_tqdm=True, **kwargs)

        texts = [""] * len(conversations)
        for i, o in zip(ok, outs):
            t = o.outputs[0].text
            # Qwen3's template opens <think> inside the prompt, so its output starts
            # mid-thought; put the tag back so strip_reasoning() sees an unfinished
            # thought (= no answer) when generation was cut off.
            if mode == "switch" and think and "<think>" not in t:
                t = "<think>" + t
            elif mode == "always" and "</think>" in t and "<think>" not in t:
                t = "<think>" + t
            texts[i] = t
        return texts

    def count_tokens(self, conversation: list[dict]) -> int:
        if self.cfg.get("system_to_user"):
            conversation = system_into_user(conversation)
        return count_chat_tokens(self.tokenizer, conversation)


def system_into_user(conversation: list[dict]) -> list[dict]:
    """Put a system message's text at the start of the first user message."""
    sys_txt = "\n\n".join(m["content"] for m in conversation if m["role"] == "system")
    rest = [dict(m) for m in conversation if m["role"] != "system"]
    if sys_txt and rest and rest[0]["role"] == "user":
        rest[0]["content"] = sys_txt + "\n\n" + rest[0]["content"]
    return rest


def count_for_model(model: str, tokenizer, conversation: list[dict]) -> int:
    """Prompt tokens as `model` will see them (handles system_to_user models)."""
    if MODEL_CFG.get(model, {}).get("system_to_user"):
        conversation = system_into_user(conversation)
    return count_chat_tokens(tokenizer, conversation)


def count_chat_tokens(tokenizer, conversation: list[dict]) -> int:
    """Prompt length in tokens (chat template applied, as the model sees it)."""
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])
