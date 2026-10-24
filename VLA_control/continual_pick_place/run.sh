#!/bin/bash
# MiniLM + position (grounding) loss, continual cube -> apple -> can with raw
# replay, evaluated back-to-home. See README.md.
#
#   ./run.sh                 train + evaluate (generates demos if missing)
#   ./run.sh eval            evaluate the existing checkpoint only
#   LABEL=myrun ./run.sh     train under another checkpoint folder
set -e -o pipefail
PY=${PY:-$(command -v python)}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} MUJOCO_GL=egl
cd "$(dirname "$0")"
LABEL=${LABEL:-minilm_aux_raw_clutter}
C=checkpoints/$LABEL/stage3_can.pt; E=results/eval_$LABEL

if [ "$1" != "eval" ]; then
  if [ ! -f dataset_clutter/index.json ]; then
    echo "=== demonstrations, all three objects on the table $(date +%H:%M) ==="
    $PY generate_demos.py --clutter --episodes 170 --workers 9
  fi
  echo "=== training $LABEL $(date +%H:%M) ==="
  $PY train_continual.py --arch minilm_aux --replay raw --aux-weight 1e-4 \
      --data-dir dataset_clutter --label "$LABEL" --select last --workers 6
fi

echo "=== evaluation, in parallel $(date +%H:%M) ==="
# the main result, standard protocol: arm home + bowl emptied between instructions
$PY combined_deploy.py --ckpt $C --episodes 10 --rehome 3 --videos 1 \
    --tag rehome_$LABEL > ${E}_rehome.log 2>&1 &
# the original protocol (no re-homing, bowl kept), for comparison
$PY combined_deploy.py --ckpt $C --episodes 10 --videos 2 --rehome 0 --keep-bowl \
    > ${E}_stage3.log 2>&1 &
# each object alone, same layouts
for t in cube apple can; do
  $PY combined_deploy.py --ckpt $C --episodes 10 --only $t \
      --tag ${LABEL}_final_only_$t > ${E}_only_$t.log 2>&1 &
done
wait
for f in rehome stage3 only_cube only_apple only_can; do
  grep -E "checkpoint :|instructions:  |^    (cube|apple|can) " ${E}_$f.log
done
echo "=== $LABEL done $(date +%H:%M) ==="
