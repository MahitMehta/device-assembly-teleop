#!/usr/bin/env bash
# Run on the Mac: copies a trained policy from Gautschi to coral (coral can't reach Gautschi) as a
# self-contained directory, ~/policies/<name> on coral, which the `policy` service sees as
# /policies/<name>:
#   final.pt (or --checkpoint), config.yaml, log.jsonl   from the Gautschi run (checkpoint stripped to its EMA weights)
#   meta.json                                            from the training dataset (image size)
#   src/dit_policy                                       the policy code, from this Mac
#
#   teleop/scripts/deploy_policy.sh real_v1                      # $SCRATCH/dit/runs/real_v1 -> coral:~/policies/real_v1
#   teleop/scripts/deploy_policy.sh real_v1 --checkpoint step_0080000.pt --data clear_piece_v2 --name real_v1_80k
#
# The code is copied from ../policy/dit_policy next to this repo (DeviceAssembly/policy), so deploy from
# the same version the run was trained with.
set -euo pipefail
# Needs a Python with torch on the Mac for stripping the checkpoint: ../policy/.venv (POLICY_PY overrides).
RUN=${1:?run name under \$SCRATCH/dit/runs on Gautschi (e.g. real_v1)}
shift
CKPT=final.pt; DATA=clear_piece_v2; NAME=$RUN
GAUTSCHI=${GAUTSCHI:-gautschi}; CORAL=${CORAL:-coral}
while [ $# -gt 0 ]; do
  case $1 in
    --checkpoint) CKPT=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done
here=$(cd "$(dirname "$0")/.." && pwd)
src=${POLICY_SRC:-$here/../policy/dit_policy}
[ -d "$src" ] || { echo "no policy code at $src (set POLICY_SRC)" >&2; exit 1; }

remote_scratch=$(ssh "$GAUTSCHI" 'echo $RCAC_SCRATCH' 2>/dev/null | tail -1)
[ -n "$remote_scratch" ] || remote_scratch=/scratch/gautschi/$(ssh "$GAUTSCHI" whoami 2>/dev/null | tail -1)
run=$remote_scratch/dit/runs/$RUN

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
echo "Fetching $GAUTSCHI:$run/$CKPT ..."
rsync -a "$GAUTSCHI:$run/$CKPT" "$tmp/"
rsync -a "$GAUTSCHI:$run/config.yaml" "$GAUTSCHI:$run/log.jsonl" "$tmp/" 2>/dev/null || true
rsync -a "$GAUTSCHI:$remote_scratch/dit/data/$DATA/meta.json" "$tmp/" 2>/dev/null \
  || echo "no meta.json for $DATA: the node will assume 192x256 images" >&2
mkdir -p "$tmp/src"
rsync -a --exclude __pycache__ "$src" "$tmp/src/"
# Keep only what inference needs: the EMA weights, config and action normalizer (~170 MB). The
# training weights and optimizer state (the other ~500 MB) stay in the Gautschi run.
PY=${POLICY_PY:-$here/../policy/.venv/bin/python}
"$PY" - "$tmp/$CKPT" "$tmp/final.pt.tmp" <<'PYEOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
torch.save({k: ck[k] for k in ("ema", "cfg", "normalizer", "step")}, sys.argv[2])
print(f"checkpoint at step {ck['step']}: kept the EMA weights only")
PYEOF
rm "$tmp/$CKPT"; mv "$tmp/final.pt.tmp" "$tmp/final.pt"   # the node loads final.pt by default
echo "$RUN/$CKPT, deployed $(date -u +%Y-%m-%dT%H:%MZ)" > "$tmp/SOURCE"

ssh "$CORAL" "mkdir -p ~/policies/$NAME"
rsync -a --delete "$tmp/" "$CORAL:policies/$NAME/"
echo "Deployed to $CORAL:~/policies/$NAME. On coral: docker compose run --rm policy /policies/$NAME"
