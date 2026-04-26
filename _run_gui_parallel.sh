#!/usr/bin/env bash
# Take the GUI-generated case, force numberOfSubdomains=4, run parallel,
# and time the solver. This isolates "serial-vs-parallel" from "GUI-extras".
set +e
SRC=/home/naor/OpenFOAM/naor-9/run/Work/Case_3D_20260426_223337
DST=/tmp/gui_case_parallel
END_TIME=0.0006

source /opt/openfoam9/etc/bashrc

echo "==== Cloning GUI case ===="
rm -rf "$DST"
mkdir -p "$DST"
# Copy 0.orig (template fields) + constant + system + helpers
cp -r "$SRC"/0.orig "$DST"/0.orig
cp -r "$SRC"/constant "$DST"/constant
cp -r "$SRC"/system "$DST"/system
cp "$SRC"/check_alpha_c4.sh "$SRC"/check_charge_region.py "$SRC"/check_internal_patch.sh "$DST"/ 2>/dev/null

# Patch numberOfSubdomains 1 -> 4
sed -i "s|^numberOfSubdomains.*|numberOfSubdomains 4;|" "$DST/system/decomposeParDict"
# Patch endTime to 0.0006
sed -i "s|^endTime\s\+.*|endTime         $END_TIME;|" "$DST/system/controlDict"

echo "==== decomposeParDict after patch ===="
grep -E "numberOfSubdomains|method" "$DST/system/decomposeParDict"

cd "$DST"
mkdir -p 0
cp -r 0.orig/* 0/  # restore from template

# Run pipeline (serial setup, parallel solver — same as direct)
echo "==== Pipeline ===="
GLOBAL_START=$(date +%s.%N)

T0=$(date +%s.%N)
surfaceFeatures > log.surfaceFeatures 2>&1
T1=$(date +%s.%N)
echo "  surfaceFeatures: $(echo "$T1-$T0" | bc) s"

blockMesh > log.blockMesh 2>&1
T2=$(date +%s.%N)
echo "  blockMesh:       $(echo "$T2-$T1" | bc) s"

snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
T3=$(date +%s.%N)
echo "  snappyHexMesh:   $(echo "$T3-$T2" | bc) s"

addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1
T4=$(date +%s.%N)
echo "  addEmptyPatch:   $(echo "$T4-$T3" | bc) s"

# Restore 0/
rm -rf 0
cp -r 0.orig 0
changeDictionary > log.changeDictionary 2>&1
checkMesh > log.checkMesh 2>&1

setRefinedFields > log.setRefinedFields 2>&1
T5=$(date +%s.%N)
echo "  setRefinedFields:$(echo "$T5-$T4" | bc) s (incl. changeDict + checkMesh)"

# Decompose for parallel
decomposePar -force > log.decomposePar 2>&1
T6=$(date +%s.%N)
echo "  decomposePar:    $(echo "$T6-$T5" | bc) s"

echo "==== blastFoam (parallel, 4 cores, GUI-generated dicts) ===="
SOLVER_START=$(date +%s.%N)
mpirun -np 4 blastFoam -parallel > log.blastFoam 2>&1
SOLVER_END=$(date +%s.%N)
SOLVER_DT=$(echo "$SOLVER_END-$SOLVER_START" | bc)
echo "  blastFoam:       $SOLVER_DT s"

GLOBAL_END=$(date +%s.%N)
TOTAL=$(echo "$GLOBAL_END-$GLOBAL_START" | bc)
echo "==== Total: $TOTAL s ===="

echo
echo "==== blastFoam log final markers ===="
grep -E "^Time = |^ExecutionTime|nProcs" log.blastFoam | tail -10
