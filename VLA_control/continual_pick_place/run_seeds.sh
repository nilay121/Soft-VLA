#!/bin/bash
# =============================================================================
# Continual pick-and-place: train the model with several seeds and evaluate
# each, saving every log, metric, checkpoint and evaluation.
#
#   systemd-inhibit --what=handle-lid-switch:sleep:idle --why=training \
#       bash VLA_control/continual_pick_place/run_seeds.sh
#
# Model: MiniLM + position loss (arch minilm_aux) WITHOUT FiLM (the default
# since 2026-10-03; TRAIN_ARGS=--with-film for the earlier model), continual cube -> apple ->
# can, RAW replay with CAPACITY samples per finished object (2 000), spread
# layout (CONTINUAL_LAYOUT=spread), demonstrations in dataset_clutter_spread/.
# Every seed trains its OWN stage 1 (--fresh-stage1), so the seeds are
# independent from the first epoch.
#
# Runs the code of this folder in place; every output goes to seeds/
# (RUN_DIR: checkpoints/, results/, logs/), through CONTINUAL_OUT, which
# train_continual.py and combined_deploy.py read. Re-running resumes.
#
# Evaluation (standard protocol: arm re-homed + bowl emptied between
# instructions), per model:
#   stage 1 checkpoint: cube instruction          x 3 layout sets x 10 episodes
#   stage 2 checkpoint: cube + apple              x 3 layout sets
#   final checkpoint:   all three, random order   x 3 layout sets (+ videos)
#   final checkpoint:   each object alone          x 1 layout set
#
# Settings (environment variables, defaults in brackets)
#   PY        python interpreter                  [python from the active env]
#   RUN_DIR   output folder                       [continual_pick_place/seeds]
#   SEEDS     training seeds                      ["0 1 2"]
#   CAPACITY  raw replay samples per object       [2000]
#   PARALLEL  train the seeds at the same time    [1]  (~2.5 GB GPU each)
#   JOBS      evaluations at the same time        [8]
#   VIDEOS    videos per model (final, layout 5000) [2]
#   EPISODES  episodes per evaluation             [10]
#   REGEN_DATA  record new demonstrations into dataset_clutter_spread/ if it
#               is missing (needed on another machine: rendering depends on
#               the GPU driver)                  [0]
#   TRAIN_ARGS  extra train_continual.py arguments                [""]
# =============================================================================
set -euo pipefail

PY=${PY:-$(command -v python)}
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # continual_pick_place
RUN_DIR=$(realpath -m "${RUN_DIR:-$SRC/seeds}")
SEEDS=${SEEDS:-"0 1 2"}
CAPACITY=${CAPACITY:-2000}
PARALLEL=${PARALLEL:-1}
JOBS=${JOBS:-8}
VIDEOS=${VIDEOS:-2}
EPISODES=${EPISODES:-10}
REGEN_DATA=${REGEN_DATA:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}
export MUJOCO_GL=${MUJOCO_GL:-egl} CONTINUAL_LAYOUT=spread \
       HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} CONTINUAL_OUT="$RUN_DIR"
DATA=dataset_clutter_spread

LOG=$RUN_DIR/logs
mkdir -p "$LOG" "$RUN_DIR/results" "$RUN_DIR/checkpoints"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/pipeline.log"; }
run() {
  local name=$1; shift
  say "START $name: $*"
  if "$@" > "$LOG/$name.log" 2>&1; then say "DONE  $name"
  else say "FAILED $name (see $LOG/$name.log)"; return 1; fi
}

say "=== continual pick-and-place seeds: RUN_DIR=$RUN_DIR SEEDS=[$SEEDS] raw replay $CAPACITY ==="

# relative checkpoint / result paths below are inside RUN_DIR
cd "$RUN_DIR"

# ---- 0. environment -------------------------------------------------------------
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "python: $PY"
  "$PY" - <<'EOF'
import platform, sys
print("python", sys.version.split()[0], platform.platform())
import numpy, mujoco, torch, torchvision, transformers
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__, "| torch",
      torch.__version__, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__)
print("cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
  echo "git: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo n/a), $(git -C "$SRC" status --short 2>/dev/null | wc -l) uncommitted"
  echo "settings: EPISODES=$EPISODES SEEDS=[$SEEDS] CAPACITY=$CAPACITY PARALLEL=$PARALLEL JOBS=$JOBS VIDEOS=$VIDEOS REGEN_DATA=$REGEN_DATA TRAIN_ARGS=[$TRAIN_ARGS] CONTINUAL_LAYOUT=$CONTINUAL_LAYOUT"
} > "$LOG/environment.txt" 2>&1
cat "$LOG/environment.txt"
"$PY" -c "import torch; assert torch.cuda.is_available()" \
  || { say "no CUDA GPU visible; aborting"; exit 1; }

# ---- 1. demonstrations -------------------------------------------------------------
if [ -e "$SRC/$DATA/index.json" ]; then say "demonstrations: $SRC/$DATA present"
elif [ "$REGEN_DATA" = "1" ]; then
  run 1_generate_demos "$PY" "$SRC/generate_demos.py" --clutter --episodes 170 --workers 10
else
  say "no demonstrations in $SRC/$DATA (set REGEN_DATA=1 to record them)"; exit 1
fi
"$PY" - <<EOF | tee -a "$LOG/pipeline.log"
import json; d=json.load(open("$SRC/$DATA/index.json"))
print("  demos kept:", {k: v["episodes_kept"] for k, v in d["generated"].items()})
EOF

# ---- 2. training ------------------------------------------------------------------
train_one() {
  local s=$1
  if [ -f "checkpoints/seed$s/stage3_can.pt" ] && [ -f "results/train_seed$s.json" ]; then
    say "SKIP training seed $s"; return 0; fi
  # shellcheck disable=SC2086
  run "2_train_seed$s" "$PY" "$SRC/train_continual.py" --arch minilm_aux --replay raw \
      --capacity "$CAPACITY" --aux-weight 1e-4 --data-dir "$DATA" \
      --label "seed$s" --select last --seed "$s" --fresh-stage1 --workers 4 $TRAIN_ARGS
}
fail=0
if [ "$PARALLEL" = "1" ]; then
  pids=(); for s in $SEEDS; do train_one "$s" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
  for s in $SEEDS; do train_one "$s" || fail=1; done
fi
[ "$fail" = 0 ] || { say "a training run failed - see logs/"; exit 1; }

# ---- 3. evaluation ----------------------------------------------------------------
say "START 3_evaluation ($JOBS at a time)"
{
  for s in $SEEDS; do
    c=checkpoints/seed$s
    for L in 5000 6000 7000; do
      v=0; [ "$L" = 5000 ] && v=$VIDEOS
      echo "--ckpt $c/stage1_cube.pt --tasks cube --episodes $EPISODES --seed $L --tag s${s}_stage1_L$L"
      echo "--ckpt $c/stage2_apple.pt --tasks cube apple --episodes $EPISODES --seed $L --tag s${s}_stage2_L$L"
      echo "--ckpt $c/stage3_can.pt --episodes $EPISODES --seed $L --videos $v --tag s${s}_final_L$L"
    done
    for o in cube apple can; do
      echo "--ckpt $c/stage3_can.pt --only $o --episodes $EPISODES --seed 5000 --tag s${s}_alone_$o"
    done
  done
} | while read -r line; do
  tag=${line##*--tag }
  [ -f "results/$tag.json" ] || echo "$line"
done | xargs -r -P "$JOBS" -L 1 sh -c \
  'tag=$(echo "$@" | sed "s/.*--tag //"); '"$PY"' "'"$SRC"'/combined_deploy.py" "$@" > "'"$LOG"'/3_eval_$tag.log" 2>&1 || echo "FAILED $tag"' _ \
  | tee -a "$LOG/pipeline.log"
say "DONE  3_evaluation"

# ---- 4. summary -------------------------------------------------------------------
run 4_summary "$PY" "$SRC/summarize_seeds.py" "$RUN_DIR" $SEEDS
cat "$LOG/4_summary.log"
say "=== finished. Summary: $RUN_DIR/results/seeds_summary.{json,csv,png} ==="
