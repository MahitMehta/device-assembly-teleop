#!/bin/bash
# Trains the DiT policy on one Gautschi H100, inside the Apptainer image.
#
#   sbatch -A <account> scripts/gautschi/train.sbatch <dataset_dir> <run_dir> [extra --set overrides...]
#   sbatch -A <account> scripts/gautschi/train.sbatch $SCRATCH/dit/data/clear_piece_v2 $SCRATCH/dit/runs/real_v1
#
# CODE=<dir with dit_policy/ and configs/> runs that code instead of the copy baked into the image
# (bind-mounted over it), so a code change needs an rsync, not an image rebuild:
#   rsync -a --exclude __pycache__ teleop/policy/dit_policy teleop/policy/configs gautschi:$SCRATCH/dit/code/   (from DeviceAssembly/)
#   CODE=$SCRATCH/dit/code sbatch -A <account> scripts/gautschi/train.sbatch ...
#
# Resumable: Slurm sends SIGUSR1 five minutes before the time limit; the trainer saves latest.pt and
# the job requeues itself, picking up where it stopped. Resubmitting the same command also resumes.
#
# Partition/QoS: RCAC lists `ai` (8x H100 per node, 14 CPU cores per GPU) with QoS `normal` or
# `preemptible`; check yours with `slist` / `sacctmgr show assoc user=$USER`, and override any
# of the lines below on the sbatch command line (e.g. -q preemptible, -t 12:00:00).
#SBATCH --job-name=dit-clear-piece
#SBATCH --partition=ai
#SBATCH --qos=preemptible
# preemptible, not normal: the group's normal-QoS GPUs are capped (our group: gres/hp_gpu=2, shared with
# the whole group), preemptible isn't (80 H100s/user) and is billed at 0.25x. A preempted job is
# requeued and the trainer resumes from latest.pt (saved every 1000 steps). Override with -q normal.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -euo pipefail

DATA=${1:?dataset dir (convert_rosbags.py or collect_demos.py output)}
RUN=${2:?run dir}
shift 2
# $SCRATCH is set only in login shells: fall back for jobs submitted over plain ssh.
SCRATCH=${SCRATCH:-${RCAC_SCRATCH:-/scratch/gautschi/$USER}}
SIF=${SIF:-$SCRATCH/dit/dit-policy.sif}
CONFIG=${CONFIG:-/opt/device_assembly/policy/configs/clear_piece_dit.yaml}
mkdir -p "$RUN"

module load apptainer 2>/dev/null || true   # on PATH by default on some RCAC clusters
# Build the .sif from the docker-archive if no job has yet (a few minutes; done here rather than in a
# separate CPU job because the CPU queue can be slower than the GPU one). Concurrent jobs each build
# to their own temp file and atomically rename, so a race only wastes a build.
if [ ! -f "$SIF" ]; then
  TAR=${TAR:-$(dirname "$SIF")/dit-policy.tar.gz}
  export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/apt_$SLURM_JOB_ID APPTAINER_CACHEDIR=${TMPDIR:-/tmp}/aptc_$SLURM_JOB_ID
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
  SRC=$TAR
  if [[ $TAR == *.gz ]]; then  # apptainer reads only an uncompressed docker-archive
    SRC=$APPTAINER_TMPDIR/image.tar; gzip -dc "$TAR" > "$SRC"
  fi
  apptainer build "$SIF.$SLURM_JOB_ID.tmp" "docker-archive://$SRC"
  mv -f "$SIF.$SLURM_JOB_ID.tmp" "$SIF"
  rm -rf "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
fi

# Stage the dataset to node-local disk: random image reads from the parallel filesystem are slow.
LOCAL=${TMPDIR:-/tmp}/dit_data_$SLURM_JOB_ID
mkdir -p "$LOCAL" && cp -r "$DATA"/. "$LOCAL"/

BINDS="$LOCAL,$RUN"
if [ -n "${CODE:-}" ]; then
  BINDS="$BINDS,$CODE/dit_policy:/opt/device_assembly/policy/dit_policy,$CODE/configs:/opt/device_assembly/policy/configs"
  echo "using the policy code in $CODE"
fi
apptainer exec --nv --bind "$BINDS" "$SIF" \
  python -m dit_policy.train --config "$CONFIG" --data "$LOCAL" --out "$RUN" \
  --set train.num_workers=$((SLURM_CPUS_PER_TASK - 2)) "$@" &
child=$!

on_usr1() {
  echo "time limit near: checkpointing and requeueing"
  kill -USR1 "$child" 2>/dev/null || true
  wait "$child" || true
  if [ ! -f "$RUN/final.pt" ]; then scontrol requeue "$SLURM_JOB_ID"; fi
  exit 0
}
trap on_usr1 USR1
wait "$child"
rm -rf "$LOCAL"
