# Image-only DiT baseline: clear-piece insertion

A Diffusion Transformer policy that sees only the wrist camera, trained on the 45 teleop demos in
`/data/teleop_demos/clear-piece` on `coral` (the 3 failed takes are in its `failures/` subfolder,
which the converter ignores). It can also be trained and
evaluated in the MuJoCo benchmark in `../../sim` (DeviceAssembly/sim, next to this repo).

**Poses in the dataset are nominal-kinematics poses.** `convert_rosbags.py` (default `--pose-source
nominal_fk`) computes `tcp_pose.npy` from `/joint_states` with `geometry.ur5e_fk`, the same nominal
UR5e model teleop commanded through (`quest_teleop/utils/core.py`) and the robot node executes with.
Older datasets used the driver's `/tcp_pose_broadcaster/pose`, which is the controller's factory-calibrated
model (~1.2 mm / 0.3 deg off on the lab arm): mixed with nominal commands, "hold still" became a
constant small action the robot then executed every plan (steady creep and tilt). One model end to end
makes a still arm a zero action on any UR5e. The dataset's `meta.json` records `pose_source`;
`policy_node --anchor auto` reads it from the deployed policy and plans from the matching pose.

| Path | What |
|---|---|
| `dit_policy/convert_rosbags.py` | rosbags → flat dataset (`images.npy`, poses, gripper, `episode_ends.npy`, `meta.json`); pure Python, no ROS |
| `dit_policy/model.py` | ResNet-18 (GroupNorm, ImageNet init) image tokens → DiT with adaLN-Zero + cross-attention, DDPM ε-prediction, DDIM sampling |
| `dit_policy/dataset.py` | Samples: 2 wrist frames in, 16-step action chunk out |
| `dit_policy/train.py` | Training: EMA, cosine LR, bf16, resumable `latest.pt`, held-out-episode action error |
| `dit_policy/policy.py` | Inference wrapper: image + current tool0 pose → tool0 targets + gripper opening |
| `configs/clear_piece_dit.yaml` | The baseline's settings |
| `docker/Dockerfile`, `scripts/build_image.sh` | One x86-64 image for training, sim demos and sim eval |
| `scripts/gautschi/*.sbatch` | Slurm jobs for Gautschi (Apptainer, H100) |

## What the policy predicts

- **Input**: the last 2 wrist RGB frames (192×256, random-cropped to 176×240 in training). No joint
  angles, poses or gripper state.
- **Output**: 16 future tool0 targets at 10 Hz (`/teleop/target_pose`, what the operator commanded),
  each as xyz + 6D rotation **in the current tool0 frame** (`action_space: rel_ee`), plus the
  commanded finger opening. The first 8 are executed, then the policy replans.

The current tool0 pose is needed to *execute* a relative chunk (target = current @ relative), but it
is never an input to the network. `action_space: abs_ee` (targets in `base`) is the alternative;
with one wrist camera it has to infer where the arm is from the image alone.

## Data (on coral)

Converted 2026-11-14 to `coral:~/dit_policy_scratch/out/clear_piece_v2` (45 episodes at 10 Hz). Use
this one: the older `out/clear_piece` has the orientation bug described below. To redo it:

```bash
docker run --rm -v /data:/data:ro -v ~/dit_policy_scratch:/w -w /w/src python:3.11-slim bash -c \
  "pip install -q rosbags opencv-python-headless scipy numpy && \
   python -m dit_policy.convert_rosbags /data/teleop_demos/clear-piece /w/out/clear_piece_v2"
```

The idle head and tail of each take (arm not moving) are trimmed.

**Orientation fix.** teleop's `_quat` (`teleop_node.py`) takes each quaternion sign from 4·w·q_i,
which is ~0 when the gripper points down, so `/teleop/target_pose`'s orientation flips by up to 180°
(37% of samples were >20° off the measured tool0). The robot was unaffected (it follows joint
commands); the recording wasn't. The converter re-picks the ambiguous signs to match the measured
orientation (after: 99% within 1.4°), and counts the fixes per bag in `meta.json`.
`python -m dit_policy.analyze_jitter <bags>` measures shake and orientation agreement. Camera intrinsics, scaled to the
stored size, are in `meta.json`.

## Gautschi

```bash
# 1. Image (on the Mac, from DeviceAssembly/): ~4 GB, x86-64
./teleop/policy/scripts/build_image.sh   # from DeviceAssembly/ (the image also packs ../sim)
ssh <you>@gautschi.rcac.purdue.edu 'mkdir -p $SCRATCH/dit/data'
scp dit-policy.tar.gz <you>@gautschi.rcac.purdue.edu:/scratch/gautschi/<you>/dit/
# on Gautschi (no root needed):
apptainer build $SCRATCH/dit/dit-policy.sif docker-archive://$SCRATCH/dit/dit-policy.tar.gz

# 2. Data: coral -> Mac -> Gautschi (coral has no Gautschi credentials)
rsync -a coral:dit_policy_scratch/out/clear_piece_v2 /tmp/ && \
  rsync -a /tmp/clear_piece_v2 <you>@gautschi.rcac.purdue.edu:/scratch/gautschi/<you>/dit/data/

# 3. Everything at once (real and sim training on separate H100s, in parallel; the first job to
#    start builds the .sif): copy scripts/gautschi to $SCRATCH/dit/scripts/gautschi, then
ACCOUNT=<account> bash $SCRATCH/dit/scripts/gautschi/submit_all.sh

# 3b. Or one at a time: train on real demos (from a checkout of this directory on Gautschi, or copy scripts/gautschi)
sbatch -A <account> scripts/gautschi/train.sbatch $SCRATCH/dit/data/clear_piece_v2 $SCRATCH/dit/runs/real_v1
tail -f dit-clear-piece-<jobid>.out      # JSON lines; also $RUN/log.jsonl
```

`<account>` is your group's RCAC allocation (from `slist`; GPU-hour balance on the `ai`
partition). That account has QoS `normal`, `preemptible`, `standby` and `burst`; the scripts use
`ai` (8× H100 per node) with `normal`. The partition allows up to 4 days per job. The train script
asks for 8 h, which should cover 100k steps at batch 128 on one H100; if it doesn't, it saves
`latest.pt` and requeues itself, and the same applies if a `preemptible` job is killed.

## Sim benchmark

The image-only DiT is trained on sim demos from the scripted expert (the same format and config as
the real data) and scored by insertion success in sim:

```bash
sbatch -A <account> scripts/gautschi/sim.sbatch collect $SCRATCH/dit/data/sim_clear_piece 45
sbatch -A <account> scripts/gautschi/train.sbatch $SCRATCH/dit/data/sim_clear_piece $SCRATCH/dit/runs/sim_v1
sbatch -A <account> scripts/gautschi/sim.sbatch eval $SCRATCH/dit/runs/sim_v1/final.pt 100
```

`eval` writes `summary.json` (success rate with a 95% interval, per-seed errors) and mp4s of the
first episodes. See `../sim/README.md` for what the sim does and does not model.

## Local (Mac)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python torch torchvision -r requirements.txt
.venv/bin/python -m dit_policy.train --config configs/clear_piece_dit.yaml --data <dataset> --out runs/dev \
  --set train.steps=2000 train.batch_size=32 train.num_workers=2    # runs on MPS
```
