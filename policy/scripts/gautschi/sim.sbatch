#!/bin/bash
# Sim jobs in the same image: collect scripted demos, or evaluate a checkpoint (EGL rendering on the GPU).
#
#   sbatch -A <account> scripts/gautschi/sim.sbatch collect $SCRATCH/dit/data/sim_clear_piece 45
#   sbatch -A <account> scripts/gautschi/sim.sbatch eval    $SCRATCH/dit/runs/sim_v1/final.pt 100
#
#SBATCH --job-name=dit-sim
#SBATCH --partition=ai
#SBATCH --qos=preemptible
# preemptible, not normal: the group's normal-QoS GPUs are capped (our group: gres/hp_gpu=2, shared with
# the whole group), preemptible isn't (80 H100s/user) and is billed at 0.25x. A preempted job is
# requeued and the trainer resumes from latest.pt (saved every 1000 steps). Override with -q normal.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -euo pipefail

MODE=${1:?collect | eval}
TARGET=${2:?output dir (collect) or checkpoint (eval)}
N=${3:-45}
# $SCRATCH is set only in login shells: fall back for jobs submitted over plain ssh.
SCRATCH=${SCRATCH:-${RCAC_SCRATCH:-/scratch/gautschi/$USER}}
SIF=${SIF:-$SCRATCH/dit/dit-policy.sif}
module load apptainer 2>/dev/null || true
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
cd /tmp

case "$MODE" in
  collect)
    mkdir -p "$(dirname "$TARGET")"
    apptainer exec --nv --bind "$(dirname "$TARGET")" "$SIF" \
      python /opt/device_assembly/sim/tools/collect_demos.py "$TARGET" --episodes "$N" --workers "$SLURM_CPUS_PER_TASK" ;;
  eval)
    RUN=$(dirname "$TARGET")
    apptainer exec --nv --bind "$RUN" "$SIF" \
      python /opt/device_assembly/sim/tools/eval_policy.py "$TARGET" --episodes "$N" --out "$RUN/eval_$(basename "$TARGET" .pt)" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
