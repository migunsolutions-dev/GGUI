#!/usr/bin/env bash
# Direct WSL run of building3D (parallel, 4 cores)
# - Copies original to /tmp to avoid touching reference
# - Renames 0/*.orig -> 0/*  (the source files have a misleading .orig suffix)
# - Sets endTime = 0.0006 (matches the GUI run's progress: ~0.000598 s)
# - Times each stage
set +e
SRC=/home/naor/OpenFOAM/naor-9/run/Work/building3D
DST=/tmp/building3D_direct
END_TIME=0.0006

source /opt/openfoam9/etc/bashrc

echo "==== Preparing fresh copy ===="
rm -rf "$DST"
mkdir -p "$DST"
cp -r "$SRC"/0 "$DST"/0
cp -r "$SRC"/constant "$DST"/constant
cp -r "$SRC"/system "$DST"/system
cp "$SRC"/Allrun "$SRC"/Allclean "$DST"/ 2>/dev/null

# Strip Zone.Identifier files (Windows-side artifacts)
find "$DST" -name "*Zone.Identifier*" -delete

# Strip the .orig suffix on field files (they should be named T, U, etc.)
cd "$DST/0"
for f in *.orig; do
    [ -e "$f" ] || continue
    base="${f%.orig}"
    if [ ! -e "$base" ]; then
        mv -- "$f" "$base"
        echo "Renamed 0/$f -> 0/$base"
    fi
done
cd "$DST"

# Patch endTime
sed -i "s|^endTime\s\+.*|endTime         $END_TIME;|" "$DST/system/controlDict"

echo "==== Files in 0/ now ===="
ls 0/

echo "==== Running parallel pipeline (4 cores) ===="
echo "Start: $(date +%T)"
GLOBAL_START=$(date +%s.%N)

T0=$(date +%s.%N)
surfaceFeatures > log.surfaceFeatures 2>&1
T1=$(date +%s.%N)
echo "  surfaceFeatures: $(echo "$T1-$T0" | bc) s"

blockMesh > log.blockMesh 2>&1
T2=$(date +%s.%N)
echo "  blockMesh:       $(echo "$T2-$T1" | bc) s"

decomposePar -copyZero > log.decomposePar 2>&1
T3=$(date +%s.%N)
echo "  decomposePar:    $(echo "$T3-$T2" | bc) s"

mpirun -np 4 snappyHexMesh -overwrite -parallel > log.snappyHexMesh 2>&1
T4=$(date +%s.%N)
echo "  snappyHexMesh:   $(echo "$T4-$T3" | bc) s"

mpirun -np 4 addEmptyPatch internalPatch internal -overwrite -parallel > log.addEmptyPatch 2>&1
T5=$(date +%s.%N)
echo "  addEmptyPatch:   $(echo "$T5-$T4" | bc) s"

mpirun -np 4 setRefinedFields -parallel > log.setRefinedFields 2>&1
T6=$(date +%s.%N)
echo "  setRefinedFields:$(echo "$T6-$T5" | bc) s"

# Solver
echo "==== blastFoam (parallel, 4 cores) ===="
SOLVER_START=$(date +%s.%N)
mpirun -np 4 blastFoam -parallel > log.blastFoam 2>&1
SOLVER_END=$(date +%s.%N)
SOLVER_DT=$(echo "$SOLVER_END-$SOLVER_START" | bc)
echo "  blastFoam:       $SOLVER_DT s"

GLOBAL_END=$(date +%s.%N)
TOTAL=$(echo "$GLOBAL_END-$GLOBAL_START" | bc)
echo "==== Total wall (no meshing-overhead extras): $TOTAL s ===="

echo
echo "==== blastFoam log tail ===="
tail -25 log.blastFoam

echo
echo "==== Final markers ===="
grep -E "^Time = |^ExecutionTime|nProcs" log.blastFoam | tail -10

echo
echo "==== Mesh sizes (initial after snappy + setRefinedFields) ===="
grep -E "Mesh size|nCells" log.snappyHexMesh log.setRefinedFields 2>/dev/null | head
ls -d processor*/0 2>/dev/null | wc -l
echo "Final cell count per processor (last time step):"
ls processor0/ 2>/dev/null
