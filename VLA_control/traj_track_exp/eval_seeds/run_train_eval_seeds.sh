#!/bin/bash
# =============================================================================
# Trajectory tracking: train the best model (future_only, ResNet18) with
# several seeds and evaluate each one, saving every log, metric, checkpoint
# and per-step prediction.
#
#   bash VLA_control/traj_track_exp/eval_seeds/run_train_eval_seeds.sh
#
# The code of traj_track_exp runs in place; every input it records and every
# output goes to traj_track_exp/seeds/ (RUN_DIR: dataset/, goals/,
# checkpoints/, results/, logs/) through TRACK_OUT, which generate_demos.py,
# record_goal_trajectory.py, train.py and deploy.py read - so the single-run
# dataset/, goals/, checkpoints/ and results/ are never touched. Re-running
# resumes: finished steps are skipped.
#
# Steps
#   0. environment report (versions, GPU, driver, git commit)
#   1. expert demonstrations (200 random paths)       generate_demos.py
#   2. taught goal trajectories (6 held-out shapes)   record_goal_trajectory.py
#   3. training, one model per seed                   train.py --seed S
#   4. evaluation of each model: clean + 3 noise seeds, all six shapes,
#      every prediction saved                         eval_seeds.py
#      + videos of the first shapes                   deploy.py
#   5. summary over the training seeds                summarize_seeds.py
#
# Steps 1-2 RE-RECORD the demonstrations and goals on this machine on
# purpose: MuJoCo's rendered pixels depend on the GPU driver, and a model
# trained or taught on another machine's renders sees a shifted input
# (measured here: 1.54 -> 2.32 mm after a driver update). Set
# REUSE_DATA=1 to link traj_track_exp/dataset and goals instead.
#
# Settings (environment variables, defaults in brackets)
#   PY          python interpreter        [python from the active env]
#   RUN_DIR     output folder             [traj_track_exp/seeds]
#   SEEDS       training seeds            ["0 1 2"]
#   WORKERS     demo-generation processes [10]
#   PARALLEL    train seeds at the same time (needs GPU memory) [0]
#   VIDEOS      videos per model (first N shapes)  [2]
#   REUSE_DATA  link the existing dataset/goals instead of re-recording [0]
#   TRAIN_ARGS  extra train.py arguments, e.g. "--epochs 80"   [""]
# =============================================================================
set -euo pipefail

PY=${PY:-$(command -v python)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../traj_track_exp/eval_seeds
SRC="$(dirname "$HERE")"                                        # .../traj_track_exp
RUN_DIR=$(realpath -m "${RUN_DIR:-$SRC/seeds}")
SEEDS=${SEEDS:-"0 1 2"}
WORKERS=${WORKERS:-10}
PARALLEL=${PARALLEL:-0}
VIDEOS=${VIDEOS:-2}
REUSE_DATA=${REUSE_DATA:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}
export MUJOCO_GL=${MUJOCO_GL:-egl} TRACK_OUT="$RUN_DIR"

LOG=$RUN_DIR/logs
mkdir -p "$RUN_DIR" "$LOG" "$RUN_DIR/checkpoints" "$RUN_DIR/results"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/pipeline.log"; }
run() {  # run <log name> <command...>: output to its own log and the screen
  local name=$1; shift
  say "START $name: $*"
  if "$@" 2>&1 | tee "$LOG/$name.log"; then say "DONE  $name"
  else say "FAILED $name (see $LOG/$name.log)"; exit 1; fi
}

say "=== tracking seeds pipeline: RUN_DIR=$RUN_DIR SEEDS=[$SEEDS] ==="

# relative data / checkpoint / result paths below are inside RUN_DIR
cd "$RUN_DIR"

# ---- 0. environment -------------------------------------------------------
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "python: $PY"
  "$PY" - <<'EOF'
import platform, sys
print("python", sys.version.split()[0], platform.platform())
import numpy, mujoco, torch, torchvision
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__,
      "| torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || echo "nvidia-smi: n/a"
  echo "git: $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo n/a) $(git -C "$SRC" status --short 2>/dev/null | wc -l) uncommitted file(s)"
  echo "settings: SEEDS=[$SEEDS] WORKERS=$WORKERS PARALLEL=$PARALLEL VIDEOS=$VIDEOS REUSE_DATA=$REUSE_DATA TRAIN_ARGS=[$TRAIN_ARGS] MUJOCO_GL=$MUJOCO_GL"
} | tee "$LOG/environment.txt"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'" \
  || { say "no CUDA GPU visible - training would be very slow; aborting"; exit 1; }

# ---- 1-2. data --------------------------------------------------------------
if [ "$REUSE_DATA" = "1" ]; then
  [ -e dataset ] || ln -s "$SRC/dataset" dataset
  [ -e goals ] || ln -s "$SRC/goals" goals
  say "reusing $SRC/dataset and $SRC/goals (linked)"
else
  if [ -f dataset/metadata.json ]; then say "SKIP demos (dataset/metadata.json exists)"
  else run 1_generate_demos "$PY" "$SRC/generate_demos.py" --workers "$WORKERS"; fi
  if [ -f goals/manifest.json ]; then say "SKIP goals (goals/manifest.json exists)"
  else run 2_record_goals "$PY" "$SRC/record_goal_trajectory.py"; fi
fi

# ---- 3. training ------------------------------------------------------------
train_one() {
  local s=$1 ck=checkpoints/track_vla_future_only_seed$1.pt
  if [ -f "checkpoints/track_vla_future_only_seed${s}_report.json" ]; then
    say "SKIP training seed $s (report exists)"; return; fi
  # shellcheck disable=SC2086
  run "3_train_seed$s" "$PY" "$SRC/train.py" --conditioning future_only --seed "$s" \
      --out "$ck" $TRAIN_ARGS
}
if [ "$PARALLEL" = "1" ]; then
  for s in $SEEDS; do train_one "$s" & done; wait
else
  for s in $SEEDS; do train_one "$s"; done
fi

# ---- 4. evaluation ----------------------------------------------------------
for s in $SEEDS; do
  ck=checkpoints/track_vla_future_only_seed$s.pt
  out=results/train_seed$s
  if [ -f "$out/aggregate.json" ]; then say "SKIP evaluation of seed $s"; else
    # clean + three noise seeds, in parallel (small GPU load)
    for c in clean seed0 seed1 seed2; do
      "$PY" "$HERE/eval_seeds.py" --ckpt "$ck" --results-dir "$out" \
          --conditions "$c" > "$LOG/4_eval_train${s}_$c.log" 2>&1 &
    done
    wait
    run "4_eval_train${s}_aggregate" "$PY" "$HERE/eval_seeds.py" --ckpt "$ck" \
        --results-dir "$out" --aggregate-only
  fi
  if [ "$VIDEOS" -gt 0 ] && ! ls results/seed${s}_future_only_*.mp4 >/dev/null 2>&1; then
    run "4_videos_train$s" "$PY" "$SRC/deploy.py" --ckpt "$ck" --tag "seed$s" --videos "$VIDEOS"
  fi
done

# ---- 5. summary -------------------------------------------------------------
run 5_summary "$PY" "$HERE/summarize_seeds.py" "$RUN_DIR" $SEEDS
say "=== finished. Results: $RUN_DIR/results/training_seeds_summary.{json,csv,png} ==="
