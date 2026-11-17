#!/bin/bash
# Submits the whole baseline as dependent Slurm jobs, on Gautschi, from this directory's parent
# (policy/). Real and sim training each get their own H100 and run in parallel:
#
#   train real (H100)                                   } in parallel
#   collect sim demos (H100) ─> train sim (H100)        }
#
# Whichever job starts first builds $SCRATCH/dit/dit-policy.sif from dit-policy.tar.gz (see train.sbatch).
#   ACCOUNT=<account> bash scripts/gautschi/submit_all.sh
#   MAIL=you@example.com ...  emails on job start / end / failure / requeue
#   BUILD_SIF=1 ...  instead builds it first in a CPU standby job (build_sif.sbatch)
set -euo pipefail
cd "$(dirname "$0")"
: "${ACCOUNT:?set ACCOUNT (see slist)}"
D=$SCRATCH/dit
SIF=$D/dit-policy.sif
mkdir -p "$D/logs" "$D/runs"
# kill-on-invalid-dep: if a job fails, its dependents are cancelled rather than left pending forever.
mail=(); [ -n "${MAIL:-}" ] && mail=(--mail-type=BEGIN,END,FAIL,REQUEUE --mail-user="$MAIL")
sub() { sbatch --parsable --kill-on-invalid-dep=yes -A "$ACCOUNT" -D "$D/logs" "${mail[@]}" "$@"; }

dep=()
if [ -n "${BUILD_SIF:-}" ]; then
  b=$(sub build_sif.sbatch "$D/dit-policy.tar.gz" "$SIF")
  dep=(--dependency=afterok:$b); echo "build-sif   $b"
fi
r=$(sub "${dep[@]}" train.sbatch "$D/data/clear_piece_v2" "$D/runs/real_v1");                 echo "train real  $r"
c=$(sub "${dep[@]}" sim.sbatch collect "$D/data/sim_clear_piece" 45);                         echo "collect sim $c"
s=$(sub --dependency=afterok:$c --job-name=dit-sim-train train.sbatch "$D/data/sim_clear_piece" "$D/runs/sim_v1"); echo "train sim   $s"
echo "logs: $D/logs   runs: $D/runs/{real_v1,sim_v1}"
