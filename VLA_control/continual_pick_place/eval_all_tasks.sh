#!/bin/bash
# =============================================================================
# Continual-learning matrix: the stage-1 (cube), stage-2 (+ apple) and
# stage-3 (+ can) checkpoints of every seed, each evaluated on EVERY task -
# learned, current and still to come - one task at a time: all three objects
# on the table, ONE instruction per episode (so a wrong pick on one task
# cannot remove the object of another), 3 layout sets x 10 episodes, standard
# settings otherwise. Nothing is trained.
#
#   bash continual_pick_place/eval_all_tasks.sh
#
# Reads seeds/checkpoints/ (run_seeds.sh), writes
# seeds/results/s<S>_stage<K>_task<T>_L<L>.json (combined_deploy.py's
# format). Stage 1 / cube is the same test as s<S>_stage1_L<L>.json (run
# by run_seeds.sh with --tasks cube) and is copied from it.
# =============================================================================
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=$(realpath -m "${RUN_DIR:-$SRC/seeds}")
cd "$RUN_DIR"                     # relative paths below are inside RUN_DIR
PY=${PY:-$(command -v python)}
SEEDS=${SEEDS:-"0 1 2"}
JOBS=${JOBS:-6}
EPISODES=${EPISODES:-10}
export MUJOCO_GL=${MUJOCO_GL:-egl} CONTINUAL_LAYOUT=spread \
       HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} CONTINUAL_OUT="$RUN_DIR"
mkdir -p logs
CKPT=(x stage1_cube stage2_apple stage3_can)
for s in $SEEDS; do for L in 5000 6000 7000; do
  [ -f "results/s${s}_stage1_taskcube_L$L.json" ] || \
    cp "results/s${s}_stage1_L$L.json" "results/s${s}_stage1_taskcube_L$L.json"
done; done
{
  for s in $SEEDS; do
    for k in 1 2 3; do
      for t in cube apple can; do
        for L in 5000 6000 7000; do
          echo "--ckpt checkpoints/seed$s/${CKPT[$k]}.pt --tasks $t --episodes $EPISODES --seed $L --tag s${s}_stage${k}_task${t}_L$L"
        done
      done
    done
  done
} | while read -r line; do
  tag=${line##*--tag }
  [ -f "results/$tag.json" ] || echo "$line"
done | xargs -r -P "$JOBS" -L 1 sh -c \
  'tag=$(echo "$@" | sed "s/.*--tag //"); '"$PY"' "'"$SRC"'/combined_deploy.py" "$@" > "logs/5_eval_$tag.log" 2>&1 && echo "done $tag" || echo "FAILED $tag"' _
echo "all done"
