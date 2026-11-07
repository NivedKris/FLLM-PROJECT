#!/usr/bin/env bash
# One entry point for the whole project: "Is RAG-MCP Safe Against Tool Poisoning?"
#
#   ./run.sh --help
#
# Typical first run on a fresh machine (Linux, NVIDIA GPU with >= 40 GB, CUDA drivers):
#   ./run.sh --setup        # Python 3.12 venv + packages
#   ./run.sh --data         # download the MCPTox benchmark into data/
#   ./run.sh --all --detach # full experiment in the background (~5 days on one A40)
set -u -o pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
MCPTOX_URL=https://github.com/zhiqiangwang4/MCPTox-Benchmark.git

usage() {
  cat <<'EOF'
Usage: ./run.sh <option> [arguments]

Setup
  --setup                 Create .venv (Python 3.12) and install requirements.txt
  --data                  Download the MCPTox benchmark into data/MCPTox-Benchmark
  --test                  Run the unit tests (data loader, retriever)

Full experiment (resumable: finished stages are skipped on re-run)
  --all [--detach]        Run every stage in order; --detach keeps it running after you
                          log out (output: logs/pipeline_main.log)
  --list                  List all stages and whether each one is done
  --stage NAME [--force]  Run a single stage; --force redoes it even if it is done
  --status                Show the last lines of the pipeline log and the current stage

Individual experiments
  --attack MODEL [opts]   One attack run, then judging and a sanity check.
                          opts: --think  --variant mimic|llm  --defense
                                --embedder bge-small|e5-base|bge-large  --seed N
                                --settings S [S ...]  --limit N
  --clean MODEL [opts]    Clean tool-selection accuracy. opts: --think --defense
  --retrieval [EMBEDDER]  Retrieval-only statistics, no LLM (default: bge-small)
  --defense-eval          Evaluate the description filter on clean tools and poisons (no LLM)
  --judge-validation      Check the LLM judge against MCPTox's stored labels
  --report                Rebuild results/REPORT.md from all finished runs
  --human-check           Sample 100 responses for a blind human check (results/human_check.csv)
  --human-score           Score the judge against the filled-in human labels

MODEL is one of: qwen3-8b, qwen3-32b-awq, mistral-7b, llama-3.1-8b, phi-4-reasoning
(Llama-3.1-8B is gated: accept its licence on Hugging Face and run `hf auth login` first.)

Examples
  ./run.sh --attack qwen3-8b --think --settings original tool@1 tool@5
  ./run.sh --attack phi-4-reasoning --variant llm --defense
  ./run.sh --clean mistral-7b --defense
  ./run.sh --stage jdg_q32 --force
EOF
}

need_venv() {
  [ -x "$PY" ] || { echo "No .venv found. Run ./run.sh --setup first." >&2; exit 1; }
}
need_data() {
  [ -f data/MCPTox-Benchmark/response_all.json ] || {
    echo "MCPTox data not found. Run ./run.sh --data first." >&2; exit 1; }
}

setup() {
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python "$PY" -r requirements.txt
  else
    local py=""
    for c in python3.12 python3; do command -v "$c" >/dev/null 2>&1 && { py=$c; break; }; done
    [ -n "$py" ] || { echo "Need uv or Python 3.12." >&2; exit 1; }
    "$py" -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
  fi
  echo "Setup done. Next: ./run.sh --data"
}

data() {
  mkdir -p data
  if [ -d data/MCPTox-Benchmark/.git ]; then
    git -C data/MCPTox-Benchmark pull --ff-only
  else
    git clone "$MCPTOX_URL" data/MCPTox-Benchmark
  fi
  echo "MCPTox data is in data/MCPTox-Benchmark"
}

pipeline() {   # pipeline [extra env assignments...]
  need_venv; need_data
  mkdir -p logs
  env "$@" bash scripts/pipeline.sh 2>&1 | tee -a logs/pipeline_main.log
}

attack() {
  need_venv; need_data
  local model=$1; shift
  local think="" variant="" defense="" emb="bge-small" seed=0 args=()
  while [ $# -gt 0 ]; do
    case $1 in
      --think)    think=1; args+=("$1") ;;
      --defense)  defense=1; args+=("$1") ;;
      --variant)  variant=$2; args+=("$1" "$2"); shift ;;
      --embedder) emb=$2; args+=("$1" "$2"); shift ;;
      --seed)     seed=$2; args+=("$1" "$2"); shift ;;
      *)          args+=("$1") ;;
    esac
    shift
  done
  # Same naming as scripts/run_attack.py:run_name
  local run=$model
  [ -n "$think" ] && run+="-think"
  [ -n "$variant" ] && run+="+$variant"
  [ -n "$defense" ] && run+="+defense"
  [ "$emb" != "bge-small" ] && run+="+$emb"
  [ "$seed" != "0" ] && run+="+seed$seed"
  if [ -n "$variant" ] && [ ! -f "results/poison_variants/$variant.json" ]; then
    echo "Poison variant '$variant' not built yet: building it first"
    $PY scripts/make_poison_variants.py --variant "$variant" || exit 1
  fi
  if [ ! -f results/references/qwen3-32b-awq.json ]; then
    echo "Judge reference calls not built yet: building them first"
    $PY scripts/make_references.py --model qwen3-32b-awq || exit 1
  fi
  $PY scripts/run_attack.py --model "$model" "${args[@]}" && \
  $PY scripts/judge_attack.py --run "$run" && \
  $PY scripts/sanity_check.py --run "$run"
}

clean() {
  need_venv; need_data
  local model=$1; shift
  local name=$model
  [[ " $* " == *" --think "* ]] && name+="-think"
  [[ " $* " == *" --defense "* ]] && name+="+defense"
  if [ ! -f results/clean/gold.json ]; then
    echo "Clean-query labels not built yet: building them first"
    $PY scripts/label_clean.py || exit 1
  fi
  $PY scripts/run_clean.py --model "$model" "$@" && \
  $PY scripts/run_clean.py --model "$model" "$@" --long && \
  $PY scripts/run_clean.py --model "$model" "$@" --summarize && \
  $PY scripts/sanity_check.py --clean "$name"
}

list_stages() {
  need_venv
  DRY=1 bash scripts/pipeline.sh 2>/dev/null | while IFS= read -r line; do
    name=${line%%:*}
    case $line in \[*|"  "*) continue ;; *:*) ;; *) continue ;; esac
    if [ -f "results/.pipeline/$name" ]; then mark="done"; else mark="    "; fi
    printf '%-5s %s\n' "$mark" "$line"
  done
}

[ $# -eq 0 ] && { usage; exit 0; }
cmd=$1; shift
case $cmd in
  -h|--help)        usage ;;
  --setup)          setup ;;
  --data)           data ;;
  --test)           need_venv; need_data; $PY -m pytest -q tests ;;
  --all)
    if [ "${1:-}" = "--detach" ]; then
      need_venv; need_data; mkdir -p logs
      echo "[$(date '+%m-%d %H:%M')] pipeline started (detached)" >> logs/pipeline_main.log
      setsid nohup bash -c 'bash scripts/pipeline.sh >> logs/pipeline_main.log 2>&1' \
        > /dev/null 2>&1 < /dev/null &
      echo "Running in the background. Follow it with: tail -f logs/pipeline_main.log"
    else
      pipeline
    fi ;;
  --list)           list_stages ;;
  --stage)
    [ -n "${1:-}" ] || { echo "Usage: ./run.sh --stage NAME [--force]" >&2; exit 1; }
    if [ "${2:-}" = "--force" ]; then pipeline ONLY="$1" FORCE=1; else pipeline ONLY="$1"; fi ;;
  --status)
    tail -n 15 logs/pipeline_main.log 2>/dev/null || echo "No pipeline log yet." ;;
  --attack)
    [ -n "${1:-}" ] || { echo "Usage: ./run.sh --attack MODEL [opts]" >&2; exit 1; }
    attack "$@" ;;
  --clean)
    [ -n "${1:-}" ] || { echo "Usage: ./run.sh --clean MODEL [--think] [--defense]" >&2; exit 1; }
    clean "$@" ;;
  --retrieval)      need_venv; need_data; $PY scripts/retrieval_stats.py --embedder "${1:-bge-small}" ;;
  --defense-eval)
    need_venv; need_data
    variants=()
    for v in mimic llm; do [ -f "results/poison_variants/$v.json" ] && variants+=("$v"); done
    $PY scripts/eval_defense.py --variants "${variants[@]}" ;;
  --judge-validation)
    need_venv; need_data
    $PY scripts/validate_judge.py --judge qwen3-32b-awq --think --refs qwen3-32b-awq --limit 300 ;;
  --report)         need_venv; $PY scripts/report.py ;;
  --human-check)    need_venv; need_data; $PY scripts/make_human_check.py --runs qwen3-8b-think qwen3-8b --n 100 ;;
  --human-score)    need_venv; need_data; $PY scripts/make_human_check.py --score ;;
  *)                echo "Unknown option: $cmd" >&2; usage; exit 1 ;;
esac
