#!/bin/bash
# Three objects kept as far apart as the reach allows (CONTINUAL_LAYOUT=
# spread, >= 40 mm edge-to-edge where possible), then three replay arms
# trained in parallel from one shared stage 1:
#   raw replay 2000 / raw replay 8000 per task / latent replay 2000 per task
#   ./run_spread.sh          demos -> stage 1 -> 3 arms -> evaluation
#   ./run_spread.sh eval     evaluation only
set -e -o pipefail
PY=${PY:-$(command -v python)}
export CONTINUAL_LAYOUT=spread HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} MUJOCO_GL=egl
cd "$(dirname "$0")"
DATA=dataset_clutter_spread
COMMON="--arch minilm_aux --aux-weight 1e-4 --data-dir $DATA --select last"
ARMS="raw:2000 raw:8000 latent:2000"
label() { echo "minilm_aux_${1}${2}_spread"; }

if [ "$1" != "eval" ]; then
  if [ ! -f $DATA/index.json ]; then
    echo "=== demos, three objects spread apart $(date +%H:%M) ==="
    $PY generate_demos.py --clutter --episodes 170 --workers ${WORKERS:-14} \
        | grep -E "task [0-9]|kept|min ->|wall clock"
  fi
  if [ ! -f checkpoints/stage1_cube_minilm_aux_${DATA}.pt ]; then
    echo "=== stage 1 (cube), shared by every arm $(date +%H:%M) ==="
    $PY train_continual.py $COMMON --replay none --upto 1 \
        --label spread_stage1 --workers 6 | grep -E "STAGE|epoch 0(01|45)/|checkpoints in"
  fi
  echo "=== three replay arms in parallel $(date +%H:%M) ==="
  for arm in $ARMS; do
    kind=${arm%:*}; cap=${arm#*:}; L=$(label $kind $cap)
    $PY train_continual.py $COMMON --replay $kind --capacity $cap \
        --label $L --workers 4 > results/run_$L.log 2>&1 &
  done
  wait
  grep -hE "arm=|after (cube|apple|can)" results/run_minilm_aux_*_spread.log
fi

echo "=== evaluation, 8 at a time $(date +%H:%M) ==="
S1=checkpoints/stage1_cube_minilm_aux_${DATA}.pt
{
  echo "--ckpt $S1 --tasks cube --episodes 10 --tag spread_stage1_cube"
  for arm in $ARMS; do
    L=$(label ${arm%:*} ${arm#*:})
    echo "--ckpt checkpoints/$L/stage2_apple.pt --tasks cube apple --episodes 10 --rehome 3 --tag rehome_${L}_stage2"
    echo "--ckpt checkpoints/$L/stage3_can.pt --episodes 10 --rehome 3 --videos 2 --seed 5000 --tag rehome_${L}"
    echo "--ckpt checkpoints/$L/stage3_can.pt --episodes 10 --rehome 3 --seed 6000 --tag rehome_${L}_seed6000"
    echo "--ckpt checkpoints/$L/stage3_can.pt --episodes 10 --rehome 3 --seed 7000 --tag rehome_${L}_seed7000"
    for t in cube apple can; do
      echo "--ckpt checkpoints/$L/stage3_can.pt --episodes 10 --only $t --tag ${L}_final_only_$t"
    done
  done
} | xargs -P 8 -L 1 sh -c 'tag=$(echo "$@" | sed "s/.*--tag //"); '"$PY"' combined_deploy.py "$@" > results/eval_$tag.log 2>&1' _
echo "=== spread done $(date +%H:%M) ==="
