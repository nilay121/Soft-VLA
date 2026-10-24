#!/bin/bash
# The four-object version: cube, apple, can, cup - all on the table in every
# demonstration and every test episode, learned in that order.
#   ./run4.sh           demos -> training -> evaluation
#   ./run4.sh eval      evaluation only
set -e -o pipefail
PY=${PY:-$(command -v python)}
export CONTINUAL_TASKS=cube,apple,can,cup HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} MUJOCO_GL=egl
cd "$(dirname "$0")"
LABEL=${LABEL:-minilm_aux_raw_clutter_cup}
DATA=dataset_clutter_cup
C=checkpoints/$LABEL/stage4_cup.pt; E=results/eval_$LABEL
WORKERS=${WORKERS:-14}

if [ "$1" != "eval" ]; then
  if [ ! -f $DATA/index.json ]; then
    echo "=== demos, all four objects on the table $(date +%H:%M) ==="
    $PY generate_demos.py --clutter --episodes 200 --workers $WORKERS \
        | grep -E "task [0-9]|kept|min ->|wall clock"
  fi
  echo "=== training $LABEL $(date +%H:%M) ==="
  $PY train_continual.py --arch minilm_aux --replay raw --aux-weight 1e-4 \
      --data-dir $DATA --label "$LABEL" --select last --workers 6
fi

echo "=== evaluation, in parallel $(date +%H:%M) ==="
# main result: all four objects on the table, four instructions per episode
# in random order, arm back home after each
$PY combined_deploy.py --ckpt $C --episodes 10 --rehome 3 --videos 3 \
    --tag rehome_$LABEL > ${E}_rehome.log 2>&1 &
for t in cube apple can cup; do
  $PY combined_deploy.py --ckpt $C --episodes 10 --only $t \
      --tag ${LABEL}_final_only_$t > ${E}_only_$t.log 2>&1 &
done
wait
for f in rehome only_cube only_apple only_can only_cup; do
  grep -E "checkpoint :|instructions:  |^    (cube|apple|can|cup) " ${E}_$f.log
done
echo "=== $LABEL done $(date +%H:%M) ==="
