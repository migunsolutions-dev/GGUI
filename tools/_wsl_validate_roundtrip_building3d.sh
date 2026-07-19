#!/usr/bin/env bash
# Developer-only: validate round-trip case through mesh + short blastFoam start.
set -eo pipefail
CASE="${1:-/mnt/c/Users/migun/Desktop/GGUI/_audit_building3d_roundtrip/roundtrip_from_loader}"
# OpenFOAM bashrc runs pipelines that can trip pipefail/errexit if sourced strictly.
set +e +o pipefail
set +u
source /opt/openfoam9/etc/bashrc
set -e -o pipefail
cd "$CASE" || exit 1

[ -x ./Allclean ] && ./Allclean || true
rm -rf 0
cp -r 0.orig 0
touch case.foam

echo "=== surfaceFeatures ==="
surfaceFeatures > log.surfaceFeatures 2>&1
tail -3 log.surfaceFeatures

echo "=== blockMesh ==="
blockMesh > log.blockMesh 2>&1
tail -3 log.blockMesh

echo "=== decomposePar ==="
decomposePar > log.decomposePar 2>&1
tail -2 log.decomposePar

NP=$(grep numberOfSubdomains system/decomposeParDict | awk '{print $2}' | tr -d ';')
echo "=== snappyHexMesh np=$NP ==="
mpirun -np "$NP" snappyHexMesh -parallel -overwrite > log.snappyHexMesh 2>&1
tail -8 log.snappyHexMesh

echo "=== reconstructParMesh ==="
reconstructParMesh -constant > log.reconstructParMesh 2>&1
tail -3 log.reconstructParMesh

addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1 || true
rm -rf processor* 0
cp -r 0.orig 0
changeDictionary > log.changeDictionary 2>&1
tail -3 log.changeDictionary

echo "=== setRefinedFields ==="
setRefinedFields > log.setRefinedFields 2>&1
tail -8 log.setRefinedFields

echo "=== blastFoam (60s cap) ==="
decomposePar -force > log.decomposeParFinal 2>&1
set +e
timeout 60 mpirun -np "$NP" blastFoam -parallel > log.blastFoam_head 2>&1
set -e
tail -45 log.blastFoam_head

echo "=== grep dictionary/cellLevel (informational) ==="
grep -E "cellLevel|errorEstimator|Refin|Unrefin|FOAM FATAL|FOAM exiting" log.blastFoam_head | head -30 || true
