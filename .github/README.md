# Soft VLA for I-Support Robot Control

Simulation and learning code for controlling a two-module pneumatic soft arm
with a soft gripper. The project includes MuJoCo scenes, expert demonstration
generation, visual policy training, trajectory tracking, and continual
instruction-conditioned pick-and-place.

## Project structure

```
simulator/                        MuJoCo model of the arm and gripper
├── configs/                      robot parameters (robot.yaml)
├── sim/                          kinematics, pressure model, MuJoCo environment, meshes
├── viz/                          rendering and video utilities
├── build_*.py                    scene builders, each writes one scene_*.xml
├── scene_*.xml                   generated scenes (committed, ready to load)
└── robot.py, hanging_robot.py, gripper.py, *_scene.py    controllers and scene wrappers
VLA_control/
├── traj_track_exp/               trajectory tracking from recorded goal images
│   └── eval_seeds/               multi-seed training and evaluation pipeline
└── continual_pick_place/         sequential, instruction-conditioned pick-and-place with replay
```

Every script adds `simulator/` to `sys.path` itself, so all commands below are
run from the repository root without installing the project as a package.

## Setup

Run these commands from the repository root. Python 3.10 and MuJoCo 3.x are
used by the project. Install PyTorch and torchvision for your machine's CUDA
or CPU environment before installing the remaining dependencies.

```bash
git clone https://github.com/NYUAD-REACH-LAB/Soft-VLA-for-ISupport-Robot-Control.git
cd Soft-VLA-for-ISupport-Robot-Control
conda create -y -n soft_vla python=3.10
conda activate soft_vla
```

The CUDA installation command recorded in `requirements.txt` is:

```bash
pip install torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Use a PyTorch build compatible with your GPU driver. For headless rendering
on an EGL-capable machine:

```bash
export MUJOCO_GL=egl
```

The MiniLM-based continual learning pipeline also requires `transformers`:

```bash
pip install transformers
```

Pretrained encoders are downloaded on first use. Allow downloads with
`HF_HUB_OFFLINE=0` when running a pipeline before its models have been cached.

## Simulator

The scene XML files are committed. Rebuild one only after changing its
builder or `simulator/configs/robot.yaml`:

```bash
python simulator/build_track_scene.py        # scene_tracking.xml
python simulator/build_bowl_scene.py         # scene_bowl_drop.xml
python simulator/build_continual_scene.py    # scene_continual.xml
```

## Trajectory tracking

The tracking pipeline records an expert trajectory as goal images. During
deployment, the policy combines live observations with those recorded frames
to follow the taught motion.

Run the expert controller on one shape (no learned model):

```bash
python VLA_control/traj_track_exp/heuristic_track.py --shape circle
```

Generate demonstrations, record the goal trajectories, train, and deploy.
`deploy.py` needs both the recorded goals and a trained checkpoint:

```bash
python VLA_control/traj_track_exp/generate_demos.py --episodes 200 --workers 10
python VLA_control/traj_track_exp/record_goal_trajectory.py
python VLA_control/traj_track_exp/train.py --conditioning future_only
python VLA_control/traj_track_exp/deploy.py --conditioning future_only --videos 2
```

These write to `dataset/`, `goals/`, `checkpoints/`, and `results/` inside
`VLA_control/traj_track_exp/`.

For training and evaluation across multiple seeds (a CUDA GPU is required):

```bash
bash VLA_control/traj_track_exp/eval_seeds/run_train_eval_seeds.sh
```

The pipeline records its own demonstrations and goals, trains one model per
seed, evaluates each, and writes everything to
`VLA_control/traj_track_exp/seeds/`. Re-running resumes from the last finished
step. `SEEDS`, `WORKERS`, `PARALLEL`, `VIDEOS`, `REUSE_DATA`, `TRAIN_ARGS`, and
`RUN_DIR` configure it, for example:

```bash
SEEDS="0 1" TRAIN_ARGS="--epochs 80" \
    bash VLA_control/traj_track_exp/eval_seeds/run_train_eval_seeds.sh
```

## Continual instruction-conditioned control

The continual learning pipeline introduces the cube, apple, and can tasks
sequentially. It combines visual observations, language embeddings, previous
pressure commands, and replay of earlier tasks.

Run the expert controller on one object (no learned model):

```bash
python VLA_control/continual_pick_place/heuristic.py --task cube
```

On a CUDA machine, generate missing demonstrations and run the seeded
pipeline with:

```bash
HF_HUB_OFFLINE=0 REGEN_DATA=1 \
    bash VLA_control/continual_pick_place/run_seeds.sh
```

It records demonstrations into `dataset_clutter_spread/`, trains one model per
seed through the three stages, evaluates every stage, and writes everything to
`VLA_control/continual_pick_place/seeds/`. Re-running resumes. `SEEDS`,
`CAPACITY`, `PARALLEL`, `JOBS`, `VIDEOS`, `EPISODES`, `TRAIN_ARGS`, and
`RUN_DIR` configure the seed list, replay buffer, concurrency, and output
location, for example `TRAIN_ARGS=--with-film`.

After `run_seeds.sh` has finished, evaluate every stage checkpoint on every
task:

```bash
bash VLA_control/continual_pick_place/eval_all_tasks.sh
```

To evaluate a single checkpoint directly:

```bash
cd VLA_control/continual_pick_place
CONTINUAL_LAYOUT=spread CONTINUAL_OUT=seeds python combined_deploy.py \
    --ckpt seeds/checkpoints/seed0/stage3_can.pt --episodes 10 --videos 2
```

Single-run variants of the same pipeline (demonstrations, training,
evaluation; pass `eval` to evaluate an existing checkpoint only):

```bash
bash VLA_control/continual_pick_place/run.sh           # three objects, raw replay
bash VLA_control/continual_pick_place/run_spread.sh    # spread layout, three replay arms
bash VLA_control/continual_pick_place/run4.sh          # four objects (adds the cup)
```
