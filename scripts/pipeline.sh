#!/usr/bin/env bash
# Full experiment pipeline (about 5 days on one A40). Usually started via ./run.sh.
#
# - Each stage writes logs/pipeline/<stage>.log.
# - A finished stage leaves a marker in results/.pipeline/ and is skipped on re-run,
#   so after fixing a failure just start this script again.
# - The first failing stage stops the pipeline (exit 1).
#
#   bash scripts/pipeline.sh [PID_TO_WAIT_FOR]
#
# Environment variables:
#   DRY=1           only list the stages and their commands
#   ONLY=<stage>    run just this one stage (skips all others)
#   FORCE=1         with ONLY: run the stage again even if it is marked done
#
# Models: q8 = Qwen3-8B, q32 = Qwen3-32B-AWQ (T = Think mode), mis = Mistral-7B-Instruct-v0.3,
#         phi = Phi-4-reasoning (always reasons), llama = Llama-3.1-8B-Instruct (needs HF access).
set -u -o pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DONE=results/.pipeline
LOG=logs/pipeline
mkdir -p "$DONE" "$LOG"

# The 7 cheaper settings used for slow models and for variant / defense runs.
RED="original tool@1 tool@3 tool@5 server-inside@1 server-separate@3 server-separate@5"

stage() {
  local name=$1; shift
  if [ -n "${DRY:-}" ]; then echo "$name: $*"; return 0; fi
  if [ -n "${ONLY:-}" ] && [ "$name" != "$ONLY" ]; then return 0; fi
  if [ -n "${ONLY:-}" ] && [ -n "${FORCE:-}" ]; then rm -f "$DONE/$name"; fi
  if [ -f "$DONE/$name" ]; then echo "[$(date '+%m-%d %H:%M')] skip  $name (done)"; return 0; fi
  echo "[$(date '+%m-%d %H:%M')] START $name"
  if "$@" > "$LOG/$name.log" 2>&1; then
    touch "$DONE/$name"; echo "[$(date '+%m-%d %H:%M')] DONE  $name"
  else
    echo "[$(date '+%m-%d %H:%M')] STAGE FAILED: $name (see $LOG/$name.log)"; exit 1
  fi
}
atk()   { $PY scripts/run_attack.py "$@"; }
# judge a run, then sanity-check it (stops the pipeline if the run looks broken)
jdg()   { $PY scripts/judge_attack.py --run "$@" && $PY scripts/sanity_check.py --run "$1"; }
# clean accuracy: normal pass, long pass (YaRN where possible), summary, sanity check
clean() {
  local name="" a prev=""
  for a in "$@"; do [ "$prev" = "--model" ] && name=$a; prev=$a; done
  [[ " $* " == *" --think "* ]] && name+="-think"
  [[ " $* " == *" --defense "* ]] && name+="+defense"
  $PY scripts/run_clean.py "$@" && $PY scripts/run_clean.py "$@" --long && \
  $PY scripts/run_clean.py "$@" --summarize && $PY scripts/sanity_check.py --clean "$name"
}
smoke() { local run=$1; shift; atk "$@" --limit 4 --settings tool@3 server-inside@1 --tag smoke && \
          $PY scripts/smoke_check.py --run "$run+smoke"; }

if [ -n "${1:-}" ] && [ -z "${DRY:-}" ]; then
  echo "[$(date '+%m-%d %H:%M')] waiting for PID $1 to finish"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

# ---------------------------------------------------------------- P. preparation (no attack runs yet)
stage tests              $PY -m pytest -q tests
stage references         $PY scripts/make_references.py --model qwen3-32b-awq
stage validate_judge     $PY scripts/validate_judge.py --judge qwen3-32b-awq --think --refs qwen3-32b-awq --limit 300
stage retr_bge_small     $PY scripts/retrieval_stats.py --embedder bge-small
stage retr_e5            $PY scripts/retrieval_stats.py --embedder e5-base
stage variant_mimic      $PY scripts/make_poison_variants.py --variant mimic

# ---------------------------------------------------------------- 0. Qwen3-8B, full grid
stage atk_q8             atk --model qwen3-8b
stage jdg_q8             jdg qwen3-8b
stage atk_q8T            atk --model qwen3-8b --think
stage jdg_q8T            jdg qwen3-8b-think
stage chain_check        bash -c 'test -f results/attack/qwen3-8b/summary.json && test -f results/attack/qwen3-8b-think/summary.json'
stage fix_tokens_think   $PY scripts/fix_prompt_tokens.py --run qwen3-8b-think
stage resum_8b           jdg qwen3-8b --resummarize
stage resum_8b_think     jdg qwen3-8b-think --resummarize

# ---------------------------------------------------------------- 1. core table
stage label_clean        $PY scripts/label_clean.py
stage clean_q8           clean --model qwen3-8b
stage human_check        $PY scripts/make_human_check.py --runs qwen3-8b-think qwen3-8b --n 100

stage smoke_q32          smoke qwen3-32b-awq --model qwen3-32b-awq
stage atk_q32            atk --model qwen3-32b-awq
stage jdg_q32            jdg qwen3-32b-awq
stage smoke_mis          smoke mistral-7b --model mistral-7b
stage atk_mis            atk --model mistral-7b
stage jdg_mis            jdg mistral-7b
stage atk_q32T           atk --model qwen3-32b-awq --think --settings $RED
stage jdg_q32T           jdg qwen3-32b-awq-think
stage smoke_phi          smoke phi-4-reasoning --model phi-4-reasoning
stage atk_phi            atk --model phi-4-reasoning --settings $RED
stage jdg_phi            jdg phi-4-reasoning
stage report1            $PY scripts/report.py

# ---------------------------------------------------------------- 2. adaptive attack + defense
stage variant_llm        $PY scripts/make_poison_variants.py --variant llm
stage eval_defense       $PY scripts/eval_defense.py --variants mimic llm
stage atk_q8_mimic       atk --model qwen3-8b --variant mimic --settings $RED
stage jdg_q8_mimic       jdg qwen3-8b+mimic
for spec in "q8|qwen3-8b|" "mis|mistral-7b|" "q32|qwen3-32b-awq|" "q8T|qwen3-8b|--think" \
            "q32T|qwen3-32b-awq|--think"; do
  IFS='|' read -r short model flag <<< "$spec"
  run=$model${flag:+-think}
  stage atk_${short}_llm      atk --model $model $flag --variant llm --settings $RED
  stage jdg_${short}_llm      jdg $run+llm
  stage atk_${short}_def      atk --model $model $flag --defense --settings $RED
  stage jdg_${short}_def      jdg $run+defense
  stage atk_${short}_llm_def  atk --model $model $flag --variant llm --defense --settings $RED
  stage jdg_${short}_llm_def  jdg $run+llm+defense
done
stage report2            $PY scripts/report.py

# ---------------------------------------------------------------- 3. clean accuracy for all + checks
stage clean_q8_def       clean --model qwen3-8b --defense
stage clean_mis          clean --model mistral-7b
stage clean_mis_def      clean --model mistral-7b --defense
stage clean_q32          clean --model qwen3-32b-awq
stage clean_q32_def      clean --model qwen3-32b-awq --defense
stage clean_q8T          clean --model qwen3-8b --think
stage clean_q8T_def      clean --model qwen3-8b --think --defense
stage clean_q32T         clean --model qwen3-32b-awq --think
stage clean_q32T_def     clean --model qwen3-32b-awq --think --defense
stage retr_bge_large     $PY scripts/retrieval_stats.py --embedder bge-large
stage retr_ksweep        $PY scripts/retrieval_stats.py --embedder bge-small --ks 1 2 3 4 5 6 7 8 9 10 --tag=-ksweep
stage atk_q8_e5          atk --model qwen3-8b --embedder e5-base --settings $RED
stage jdg_q8_e5          jdg qwen3-8b+e5-base
stage atk_q8T_seed1      atk --model qwen3-8b --think --seed 1 --settings $RED
stage jdg_q8T_seed1      jdg qwen3-8b-think+seed1
stage report3            $PY scripts/report.py

# ---------------------------------------------------------------- 4. Llama (only if HF access was granted)
if curl -sf -o /dev/null -H "Authorization: Bearer $(cat ~/.cache/huggingface/token 2>/dev/null)" \
     https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json; then
  stage dl_llama           .venv/bin/hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
  stage smoke_llama        smoke llama-3.1-8b --model llama-3.1-8b
  stage atk_llama          atk --model llama-3.1-8b
  stage jdg_llama          jdg llama-3.1-8b
  stage atk_llama_llm      atk --model llama-3.1-8b --variant llm --settings $RED
  stage jdg_llama_llm      jdg llama-3.1-8b+llm
  stage atk_llama_def      atk --model llama-3.1-8b --defense --settings $RED
  stage jdg_llama_def      jdg llama-3.1-8b+defense
  stage atk_llama_llm_def  atk --model llama-3.1-8b --variant llm --defense --settings $RED
  stage jdg_llama_llm_def  jdg llama-3.1-8b+llm+defense
  stage clean_llama        clean --model llama-3.1-8b
  stage clean_llama_def    clean --model llama-3.1-8b --defense
  stage report4            $PY scripts/report.py
else
  echo "[$(date '+%m-%d %H:%M')] Llama-3.1-8B access not granted (yet): skipping Llama stages."
  echo "  Re-run this script after access is approved to do them."
fi

# ---------------------------------------------------------------- 5. Phi-4 extras (slowest model, so last)
stage atk_phi_llm        atk --model phi-4-reasoning --variant llm --settings $RED
stage jdg_phi_llm        jdg phi-4-reasoning+llm
stage atk_phi_def        atk --model phi-4-reasoning --defense --settings $RED
stage jdg_phi_def        jdg phi-4-reasoning+defense
stage atk_phi_llm_def    atk --model phi-4-reasoning --variant llm --defense --settings $RED
stage jdg_phi_llm_def    jdg phi-4-reasoning+llm+defense
stage clean_phi          clean --model phi-4-reasoning
stage clean_phi_def      clean --model phi-4-reasoning --defense
stage report5            $PY scripts/report.py

echo "[$(date '+%m-%d %H:%M')] ALL STAGES DONE"
