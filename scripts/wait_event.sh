#!/usr/bin/env bash
# Exit as soon as the pipeline log shows a new milestone (a judged / clean / report
# stage finished, a stage failed, or everything is done). Used to wake up Claude.
LOG=${1:-logs/pipeline_main.log}
PAT='DONE  (jdg_|clean_|resum_|report|label_|variant_|eval_)|STAGE FAILED|ALL STAGES DONE'
start=$(grep -cE "$PAT" "$LOG")
while [ "$(grep -cE "$PAT" "$LOG")" -le "$start" ]; do sleep 60; done
grep -E "$PAT" "$LOG" | tail -n +$((start + 1))
