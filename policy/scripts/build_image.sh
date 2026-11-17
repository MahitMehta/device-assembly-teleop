#!/usr/bin/env bash
# Builds the training/sim image for x86-64 (Gautschi) on the Mac and packs it for Apptainer.
#
#   ./teleop/policy/scripts/build_image.sh     # from DeviceAssembly/ (needs ../sim too), writes dit-policy.tar.gz
#   scp dit-policy.tar.gz <you>@gautschi.rcac.purdue.edu:/scratch/gautschi/<you>/dit/
#   # on Gautschi (no root needed):
#   apptainer build $SCRATCH/dit/dit-policy.sif docker-archive://$SCRATCH/dit/dit-policy.tar.gz
set -euo pipefail
cd "$(dirname "$0")/../../.."   # DeviceAssembly/ (the image also takes sim/, next to teleop/)
docker buildx build --platform linux/amd64 -f teleop/policy/docker/Dockerfile -t dit-policy:latest --load .
docker save dit-policy:latest | gzip -1 > dit-policy.tar.gz
ls -lh dit-policy.tar.gz
