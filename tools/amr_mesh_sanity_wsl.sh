#!/usr/bin/env bash
# Optional debug hook: short OpenFOAM pipeline sanity check (run inside WSL from case root).
# Usage: bash tools/amr_mesh_sanity_wsl.sh [/path/to/case]
# Does not modify the repo; intended for manual / CI-on-Linux debugging only.
#
# Note: blastFoam here expects time directories under 0/ (fields). GUI cases normally use
# ``bash Allrun`` (cp 0.orig → 0, setRefinedFields/setFields, then blastFoam). This script
# only runs blockMesh + snappy + blastFoam, so blastFoam may fail with missing 0/U unless
# you prepare 0/ first. Use Allrun for end-to-end solver validation.
set -uo pipefail
CASE="${1:-.}"
cd "$CASE" || exit 1
echo "=== amr_mesh_sanity: CASE=$(pwd) ==="
if ! command -v blockMesh >/dev/null 2>&1; then
  echo "ERROR: blockMesh not in PATH (source OpenFOAM/blastFoam bashrc)."
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "--- $name ---"
  if "$@" >"log.${name}" 2>&1; then
    echo "OK  $name (see log.${name})"
    return 0
  else
    local ec=$?
    echo "FAIL $name exit=$ec (see log.${name})"
    return "$ec"
  fi
}

run_log blockMesh blockMesh
BM=$?

SN=0
if [[ -f system/snappyHexMeshDict ]] && grep -q "castellatedMesh on" system/snappyHexMeshDict; then
  if grep -q "geometry" system/snappyHexMeshDict && grep -E "triSurfaceMesh|searchableSphere|searchableCylinder|searchableBox" system/snappyHexMeshDict >/dev/null 2>&1; then
    run_log snappyHexMesh snappyHexMesh -overwrite || SN=$?
  else
    echo "--- snappyHexMesh skipped (no surface/searchable geometry in snappyHexMeshDict) ---"
  fi
fi

if [[ -f constant/dynamicMeshDict ]]; then
  echo "--- dynamicMeshDict (first 40 lines) ---"
  head -n 40 constant/dynamicMeshDict
fi

if command -v blastFoam >/dev/null 2>&1 && [[ -f system/controlDict ]]; then
  foamDictionary -entry endTime -set 1e-12 system/controlDict 2>/dev/null || true
  run_log blastFoam blastFoam || true
else
  echo "--- blastFoam skipped (solver not in PATH or no controlDict) ---"
fi

echo "=== AMR / refinement lines (last 120 lines of log.blastFoam if present) ==="
if [[ -f log.blastFoam ]]; then
  tail -n 120 log.blastFoam | grep -E -i "refin|unrefin|celllevel|imbalance|error|fatal" || true
else
  echo "(no log.blastFoam)"
fi

echo "=== cellLevel field at time directories (if any) ==="
find . -maxdepth 2 -type f -name cellLevel 2>/dev/null | head -n 20 || true

exit 0
