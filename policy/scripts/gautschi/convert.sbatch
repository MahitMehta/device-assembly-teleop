#!/bin/bash
# Converts rosbag recordings into a training dataset on Gautschi, inside the Apptainer image
# (it has rosbags, OpenCV and SciPy), so the conversion doesn't need disk space on coral.
#
#   sbatch -A <account> scripts/gautschi/convert.sbatch <bags_dir> <dataset_dir> [convert_rosbags args...]
#   CODE=$SCRATCH/dit/code sbatch -A <account> scripts/gautschi/convert.sbatch \
#       $SCRATCH/dit/bags/clear-piece $SCRATCH/dit/data/clear_piece_v4
#
# Copy the recordings first (coral has no Gautschi credentials, so through the Mac; nothing lands there):
#   ssh coral 'tar -C /data/teleop_demos --exclude=clear-piece/failures -cf - clear-piece' \
#     | ssh gautschi 'mkdir -p $SCRATCH/dit/bags && tar -C $SCRATCH/dit/bags -xf -'
# The default --pose-source is nominal_fk (poses from /joint_states through the nominal UR5e model).
# Our account has no CPU-partition hours, so this runs on `ai` with one GPU, which it doesn't use.
#SBATCH --job-name=dit-convert
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -euo pipefail

BAGS=${1:?bags dir}
OUT=${2:?dataset dir}
shift 2
SCRATCH=${SCRATCH:-${RCAC_SCRATCH:-/scratch/gautschi/$USER}}
SIF=${SIF:-$SCRATCH/dit/dit-policy.sif}
module load apptainer 2>/dev/null || true

BINDS="$BAGS,$(dirname "$OUT")"
if [ -n "${CODE:-}" ]; then
  BINDS="$BINDS,$CODE/dit_policy:/opt/device_assembly/policy/dit_policy"
  echo "using the policy code in $CODE"
fi
rm -rf "$OUT.tmp"
apptainer exec --bind "$BINDS" --pwd /opt/device_assembly/policy "$SIF" \
  python -m dit_policy.convert_rosbags "$BAGS" "$OUT.tmp" "$@"
rm -rf "$OUT" && mv "$OUT.tmp" "$OUT"   # only a finished dataset appears under its name
echo "dataset ready: $OUT"
