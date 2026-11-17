#!/bin/bash
# Builds the Apptainer image from the docker-archive made by scripts/build_image.sh. A CPU job
# (too heavy for a login node) on the uncharged `standby` QoS.
#
#   sbatch -A <account> scripts/gautschi/build_sif.sbatch $SCRATCH/dit/dit-policy.tar.gz $SCRATCH/dit/dit-policy.sif
#SBATCH --job-name=dit-build-sif
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
TAR=${1:?docker-archive .tar.gz}
SIF=${2:?output .sif}
module load apptainer 2>/dev/null || true
W=${TMPDIR:-/tmp}/apptainer_$SLURM_JOB_ID
export APPTAINER_TMPDIR=$W/tmp APPTAINER_CACHEDIR=$W/cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
apptainer build --force "$SIF.tmp" "docker-archive://$TAR"
mv "$SIF.tmp" "$SIF"
# Smoke test: everything the training and sim jobs import.
apptainer exec "$SIF" python -c "import torch, mujoco, dit_policy.train, coral_sim.env; print('ok', torch.__version__, mujoco.__version__)"
rm -rf "$W"
