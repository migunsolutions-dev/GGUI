import os
import math
import shlex
from typing import Optional

# Pre-solver sanity check: 0/alpha.c4 must have at least one positive internalField value.
# Used in Allrun and written to check_alpha_c4.sh for GUI init. Catches: uniform 0, nonuniform all zeros.
# Use [ \t]+ (not [[:space:]]) so all awks (mawk, gawk, busybox) match; handle OpenFOAM format: count line, then ( then one value per line.
ALPHA_C4_CHECK_SCRIPT = r'''#!/usr/bin/env bash
# Pre-solver sanity: alpha.c4 must have at least one positive internalField value (no explosive mass -> blastFoam "No mass was found in the domain")
if [ ! -f 0/alpha.c4 ]; then
  echo "FATAL: alpha.c4 has no positive internalField values (no explosive mass). File 0/alpha.c4 is missing."
  exit 1
fi
awk '
BEGIN { in_list = 0; found = 0; }
/^internalField[ \t]+uniform[ \t]/ {
  n = $3; gsub(/[;]/, "", n);
  found = ((n + 0) > 0) ? 1 : 0;
  exit;
}
/^internalField[ \t]+nonuniform/ {
  in_list = 0;
  next;
}
in_list && /^[ \t]*\)/ {
  next;
}
# Match lines that start with a number (no $ so trailing space/CR does not break)
in_list && /^[0-9eE+.\t -]+/ {
  for (i = 1; i <= NF; i++) {
    if ($i + 0 > 0) { found = 1; break; }
  }
  next;
}
/\(/ {
  in_list = 1;
  next;
}
END { exit (found ? 0 : 1); }
' 0/alpha.c4 || {
  echo "FATAL: No cells captured inside the charge volume. Increase charge pre-refinement (Inside/Outside), increase Outside extent, or refine base mesh near the charge."
  exit 1
}
'''

class BaseGenerator:
    """
    Base class containing shared utilities for OpenFOAM case generation.
    Used by both 1D and 3D generators.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path

    def _foam_header(self, object_name: str, foam_class: str, location: Optional[str] = None) -> str:
        loc = f'    location    "{location}";\n' if location else ""
        return (
            "/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "| =========                 |                                                 |\n"
            "| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
            "|  \\\\    /   O peration     |                                                 |\n"
            "|   \\\\  /    A nd           |                                                 |\n"
            "|    \\\\/     M anipulation  |                                                 |\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            f"    class       {foam_class};\n"
            f"    object      {object_name};\n"
            f"{loc}"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
        )

    def _write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    def create_case_dirs(self, case_name: str) -> str:
        case_dir = os.path.join(self.base_path, case_name)
        # Create 0.orig explicitly
        for sub in ("0", "0.orig", "constant", "system"):
            os.makedirs(os.path.join(case_dir, sub), exist_ok=True)
        return case_dir

    def calculate_charge_radius(self, mass_kg: float, density_kg_m3: float) -> float:
        if density_kg_m3 <= 0: return 0.05
        volume = float(mass_kg) / float(density_kg_m3)
        return ((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0)
    
    def write_scripts(
        self,
        case_dir: str,
        openfoam_bashrc: str,
        use_snappy: bool = False,
        cores: int = 1,
        init_from_1d: bool = False,
        mapped_source_dir_linux: Optional[str] = None,
        mapped_source_time: Optional[str] = None,
        solver_app: str = "blastFoam",
        use_set_refined: bool = False,
        use_seed_bubble: bool = False,
        startup_refinement_levels: Optional[int] = None,
        remaining_inside_levels: int = 0,
        use_charge_region_check: bool = False,
        use_charge_interior_refinement: bool = False,
        inside_levels: int = 0,
        capture_levels: int = 0,
        charge_levels: int = 0,
        outside_levels: int = 0,
        charge_capture_impossible_message: Optional[str] = None,
        envelope_empty_message: Optional[str] = None,
        charge_region_empty_message: Optional[str] = None,
        placement_use_setfields: bool = False,
    ) -> None:
        """
        Write Allrun and Allclean scripts.
        When init_from_1d is True, mapped_source_dir_linux and mapped_source_time must be set;
        Allrun will run postProcess writeCellCentres and remap_radial.py (after mesh + cp 0.orig 0) then the solver, skipping setFields.
        When use_set_refined is True, Allrun uses setRefinedFields instead of setFields.
        When use_charge_interior_refinement is True: topoSet captureEnvelope, refineMesh captureEnvelope capture_levels times (guarantee capture),
        then topoSet chargeRegion (true geometry), refineMesh chargeRegion charge_levels times (Inside = only inside true charge). inside_levels = capture_levels + charge_levels.
        Then topoSet outsideShell and refineMesh outside_levels times, rm hexRef metadata, addEmptyPatch, etc. Final explosive fill uses true charge geometry only (setFieldsDict).
        When solver_app is rhoCentralFoam (inert continuation), no detonation/activation is used.
        """
        
        # Start with Allclean (safe: artifacts only; preserves 0.orig, system/, constant dicts, triSurface/*.stl)
        script_head = f"""#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
source "{openfoam_bashrc}" || true
set -e

# 1. Clean previous results (safe Allclean: artifacts only; preserves inputs)
[ -x ./Allclean ] && ./Allclean || true

# 2. Restore 0 from template (Safe Refresh)
if [ -d "0.orig" ]; then
    rm -rf 0
    cp -r 0.orig 0
fi

# Ensure case.foam exists for Paraview/PyVista
touch case.foam

# Stage verification: tee all output to log.stageVerification and define stage_check
# alpha_check only meaningful AFTER setRefinedFields/setFields (stages 5/6); before that 0/alpha.c4 is still 0 from 0.orig
exec > >(tee log.stageVerification) 2>&1
stage_check () {{
  echo "=== STAGE: $1 ==="
  if [ -f 0/alpha.c4 ]; then
    grep -n "internalField" -m 1 0/alpha.c4 || true
    if [[ "$1" == *"5_after_setFields"* ]] || [[ "$1" == *"6_before"* ]]; then
      if [ -f ./check_alpha_c4.sh ]; then
        bash ./check_alpha_c4.sh && echo "alpha_check: PASS" || echo "alpha_check: FAIL"
      else
        echo "alpha_check: N/A (no check_alpha_c4.sh)"
      fi
    else
      echo "alpha_check: N/A (before setRefinedFields/setFields)"
    fi
  else
    echo "alpha_check: FAIL (missing 0/alpha.c4)"
  fi
}}
stage_check "1_after_restore_0_from_0.orig"
"""
        use_mapping = init_from_1d and mapped_source_dir_linux and mapped_source_time
        inert_continuation = use_mapping and (solver_app == "rhoCentralFoam")
        if use_mapping:
            if inert_continuation and mapped_source_time:
                # Inert continuation: write remap into t_map/, then rhoCentralFoam starts from startTime t_map
                t_map = mapped_source_time
                init_fields_block = f"""
# 7. Write cell centres for remap (writes to 0/C)
postProcess -func writeCellCentres > log.writeCellCentres 2>&1
# 8. Prepare t_map and map ICs from 1D into t_map (rhoCentralFoam will start from startTime t_map)
MAP_TIME="{t_map}"
mkdir -p "$MAP_TIME"
cp -r 0.orig "$MAP_TIME"
export MAP_TIME
python3 remap_radial.py > log.remap_radial 2>&1 || exit 1
"""
            else:
                init_fields_block = """
# 7. Write cell centres for remap script (no PyVista required in WSL)
postProcess -func writeCellCentres > log.writeCellCentres 2>&1
# 8. Map ICs from 1D (radial remap, Autodyn-style)
python3 remap_radial.py > log.remap_radial 2>&1
"""
        else:
            # Sanity check: alpha.c4 must have at least one positive internalField value (uniform or nonuniform)
            self._write_text(os.path.join(case_dir, "check_alpha_c4.sh"), ALPHA_C4_CHECK_SCRIPT)
            alpha_check_block = """
# Pre-solver sanity: alpha.c4 must have at least one positive value (no explosive mass -> blastFoam "No mass was found in the domain")
bash ./check_alpha_c4.sh || exit 1
"""
            if use_set_refined:
                # setRefinedFields only places charge; refinement inside charge is done by topoSet chargeRegion + refineMesh (before addEmptyPatch) when use_charge_interior_refinement.
                # When use_charge_interior_refinement: mesh refinement metadata was removed, so setRefinedFields must run with -noRefine (placement-only; no refinement).
                # Cuboid uses setFields for placement (setRefinedFields does not support boxToCell); mesh already refined by topoSet/refineMesh.
                # Resize 0/ to refined mesh first: setRefinedFields -noRefine with minimal dict, then setFields to fill box.
                if use_charge_interior_refinement and placement_use_setfields:
                    init_fields_block = """
# 7. Initialize Fields (Cuboid: 0.orig/alpha.c4 is uniform 0 and valid for any mesh size;
#    run setFields directly with boxToCell to fill the charge region)
setFields > log.setFields 2>&1
"""
                else:
                    set_refined_cmd = "setRefinedFields -noRefine" if use_charge_interior_refinement else "setRefinedFields"
                    init_fields_block = f"""
# 7. Initialize Fields (setRefinedFields: charge region placement only)
{set_refined_cmd} > log.setRefinedFields 2>&1
"""
            else:
                init_fields_block = """
# 7. Initialize Fields (Serial on full mesh)
setFields > log.setFields 2>&1
"""
            # Optional: charge region-consistency check (non-remap only; requires 0/C after mesh finalization)
            # Order: setRefinedFields/setFields (charge placement) -> writeCellCentres (after final mesh at time 0) -> check_charge_region
            if use_charge_region_check:
                init_fields_block += """
postProcess -func writeCellCentres -time 0 > log.writeCellCentres 2>&1 || { echo "FATAL: postProcess writeCellCentres failed. Check log.writeCellCentres."; exit 1; }
[ -f 0/C ] || { echo "FATAL: 0/C missing after writeCellCentres. Check controlDict writeFormat and log.writeCellCentres."; exit 1; }
python3 check_charge_region.py || exit 1
"""
            # Stage 5 runs after setFields/setRefinedFields; then gate on alpha check; internalPatch preflight; Stage 6 before solver
            init_fields_block += """
stage_check "5_after_setFields_or_setRefinedFields"
""" + alpha_check_block + """
[ -f check_internal_patch.sh ] && { bash check_internal_patch.sh || exit 1; } || true
stage_check "6_before_blastFoam"
"""

        if cores > 1:
            # === PARALLEL EXECUTION FLOW ===
            # Workflow: decomposePar -> (snappyHexMesh -parallel if use_snappy) -> reconstructParMesh if snappy
            #           -> setFields -> decomposePar -force -> mpirun blastFoam -parallel -> reconstructPar
            script_body = ""
            if use_snappy:
                script_body += """
# 3. Preflight: STL files exist (obstacle consistency)
if [ -f system/expectedFeatureEdges.txt ]; then
  while read -r f; do
    [ -z "$f" ] && continue
    base="${f##*/}"
    base="${base%.extendedFeatureEdgeMesh}"
    base="${base%.eMesh}"
    stl="constant/triSurface/${base}.stl"
    if [ ! -f "$stl" ]; then
      echo "FATAL: Referenced STL missing: $stl (required for obstacle). Add file or remove obstacle."
      exit 1
    fi
  done < system/expectedFeatureEdges.txt
fi
# 4. Extract features
surfaceFeatures > log.surfaceFeatures 2>&1
# Preflight: feature edge file at canonical path (each line = full path)
if [ -f system/expectedFeatureEdges.txt ]; then
  while read -r f; do
    [ -z "$f" ] && continue
    if [ ! -f "$f" ]; then
      echo "FATAL: Feature edge file missing at $f after surfaceFeatures. Fix: check STL in constant/triSurface, re-run surfaceFeatures, or remove obstacle."
      exit 1
    fi
  done < system/expectedFeatureEdges.txt
fi
"""
            script_body += """
# 5. Background mesh
blockMesh > log.blockMesh 2>&1

# 6. Decompose (Initial)
decomposePar > log.decomposePar 2>&1
"""
            if use_snappy:
                script_body += """
# 6. SnappyHexMesh (Parallel)
mpirun -np $(grep "numberOfSubdomains" system/decomposeParDict | awk '{print $2}' | tr -d ';') snappyHexMesh -parallel -overwrite > log.snappyHexMesh 2>&1

# --- CRITICAL: Reconstruct Mesh & Reset Fields ---
reconstructParMesh -constant > log.reconstructParMesh 2>&1
"""
            if use_snappy:
                if use_charge_interior_refinement and (capture_levels > 0 or charge_levels > 0):
                    env_msg = envelope_empty_message or "Capture envelope is empty. Reduce base Cell Size and try again."
                    chg_empty = charge_region_empty_message or "True charge region has no cells. Reduce base Cell Size or increase Inside."
                    # Stage 1a: [topoSet captureEnvelope + refineMesh once] x capture_levels
                    # Re-selecting on the current mesh before each refinement is required because
                    # refineMesh does NOT update the source cellSet in place — after each split the
                    # old cell IDs are stale. Recomputing via topoSet guarantees the correct cells
                    # at every level.
                    for lv in range(capture_levels):
                        script_body += "# --- Capture envelope level %d/%d: re-select on current mesh, refine once ---\n" % (lv + 1, capture_levels)
                        script_body += "topoSet -dict system/topoSet_captureEnvelopeDict >> log.topoSet_captureEnvelope 2>&1\n"
                        if lv == 0:
                            script_body += 'test -s constant/polyMesh/sets/captureEnvelope || { echo "FATAL: %s"; exit 1; }\n' % env_msg.replace('"', '\\"')
                        else:
                            script_body += 'test -s constant/polyMesh/sets/captureEnvelope || { echo "FATAL: Capture envelope empty at level %d/%d. Increase Inside or reduce Cell Size."; exit 1; }\n' % (lv + 1, capture_levels)
                        script_body += "refineMesh -dict system/refineMesh_captureEnvelopeDict -overwrite >> log.refineMesh_captureEnvelope 2>&1\n"
                    # Stage 1b: [topoSet chargeRegion + refineMesh once] x charge_levels
                    if charge_levels > 0:
                        for lv in range(charge_levels):
                            script_body += "# --- Charge region level %d/%d: re-select on current mesh, refine once ---\n" % (lv + 1, charge_levels)
                            script_body += "topoSet -dict system/topoSet_chargeRegionDict >> log.topoSet_chargeRegion 2>&1\n"
                            if lv == 0:
                                script_body += 'test -s constant/polyMesh/sets/chargeRegion || { echo "FATAL: %s"; exit 1; }\n' % chg_empty.replace('"', '\\"')
                            else:
                                script_body += 'test -s constant/polyMesh/sets/chargeRegion || { echo "FATAL: Charge region empty at level %d/%d. Adjust charge geometry or Inside."; exit 1; }\n' % (lv + 1, charge_levels)
                            script_body += "refineMesh -dict system/refineMesh_chargeDict -overwrite >> log.refineMesh_charge 2>&1\n"
                    # Outside shell: [topoSet outsideShell + refineMesh once] x outside_levels
                    if outside_levels > 0:
                        for lv in range(outside_levels):
                            script_body += "# --- Outside shell level %d/%d ---\n" % (lv + 1, outside_levels)
                            script_body += "topoSet -dict system/topoSet_outsideShellDict >> log.topoSet_outsideShell 2>&1\n"
                            script_body += "refineMesh -dict system/refineMesh_outsideDict -overwrite >> log.refineMesh_outside 2>&1\n"
                    # Final topoSet_chargeRegion on the fully-refined mesh: provides accurate cell
                    # count for the realization check even when charge_levels=0 (only capture ran).
                    script_body += "# --- Final cell count in true charge region (for realization check) ---\n"
                    script_body += "topoSet -dict system/topoSet_chargeRegionDict >> log.topoSet_chargeRegion 2>&1\n"
                    # Post-refinement realization check: verifies requested Inside level was achieved.
                    script_body += "[ -f check_realization.py ] && { python3 check_realization.py || exit 1; } || true\n"
                    script_body += "# Remove hexRef metadata so setRefinedFields can place charge (region selection fails with refineMesh history present)\n"
                    script_body += "rm -f constant/polyMesh/cellLevel constant/polyMesh/pointLevel constant/polyMesh/level0Edge constant/polyMesh/refinementHistory 2>/dev/null || true\n"
                script_body += """
# --- Add internal patch FIRST (mesh must have internalPatch before 0/ restore) ---
addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1
rm -rf processor* 0
cp -r 0.orig 0
# --- changeDictionary adds boundaryField for ALL patches (internalPatch + obs.*) ---
changeDictionary > log.changeDictionary 2>&1
checkMesh > log.checkMesh 2>&1
"""
            else:
                pass

            script_body += init_fields_block
            # 3D REMAP + cores>1 only: add internal patch before blastFoam -parallel (balancing requires it)
            if use_mapping and cores > 1:
                script_body += """
# 8a. Reconstruct serial mesh + fields at time 0 (required when balancing is enabled)
reconstructPar -constant > log.reconstructParMesh 2>&1 || exit 1
reconstructPar -time 0     > log.reconstructPar0    2>&1 || exit 1
# 8b. Add internal patch (serial)
addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1 || exit 1
# 8c. Re-decompose for parallel solver
rm -rf processor*
decomposePar -force > log.decomposeParFinal 2>&1 || exit 1

# 9. Run Solver (Parallel)
mpirun -np $(grep "numberOfSubdomains" system/decomposeParDict | awk '{print $2}' | tr -d ';') """ + solver_app + """ -parallel 2>&1 | tee log.""" + solver_app + """

# 10. Reconstruct Results (skip on failure — AMR changes mesh per timestep,
#     so reconstructPar may fail; ParaView reads processor dirs directly)
reconstructPar > log.reconstructPar 2>&1 || true
"""
            else:
                script_body += """
# 8. Re-Decompose (Force) for Solver
decomposePar -force > log.decomposeParFinal 2>&1

# 9. Run Solver (Parallel)
mpirun -np $(grep "numberOfSubdomains" system/decomposeParDict | awk '{print $2}' | tr -d ';') """ + solver_app + """ -parallel 2>&1 | tee log.""" + solver_app + """

# 10. Reconstruct Results (skip on failure — AMR changes mesh per timestep,
#     so reconstructPar may fail; ParaView reads processor dirs directly)
reconstructPar > log.reconstructPar 2>&1 || true
"""

        else:
            # === SERIAL EXECUTION FLOW ===
            script_body = ""
            if use_snappy:
                script_body += """
# 3. Preflight: STL files exist (obstacle consistency)
if [ -f system/expectedFeatureEdges.txt ]; then
  while read -r f; do
    [ -z "$f" ] && continue
    base="${f##*/}"
    base="${base%.extendedFeatureEdgeMesh}"
    base="${base%.eMesh}"
    stl="constant/triSurface/${base}.stl"
    if [ ! -f "$stl" ]; then
      echo "FATAL: Referenced STL missing: $stl (required for obstacle). Add file or remove obstacle."
      exit 1
    fi
  done < system/expectedFeatureEdges.txt
fi
# 4. Extract features
surfaceFeatures > log.surfaceFeatures 2>&1
# Preflight: feature edge file at canonical path (each line = full path)
if [ -f system/expectedFeatureEdges.txt ]; then
  while read -r f; do
    [ -z "$f" ] && continue
    if [ ! -f "$f" ]; then
      echo "FATAL: Feature edge file missing at $f after surfaceFeatures. Fix: check STL in constant/triSurface, re-run surfaceFeatures, or remove obstacle."
      exit 1
    fi
  done < system/expectedFeatureEdges.txt
fi
"""
            script_body += """
# 5. Background mesh
blockMesh > log.blockMesh 2>&1
stage_check "2_after_blockMesh"
"""
            if use_snappy:
                script_body += """
# 5. SnappyHexMesh (Serial)
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
"""
                if use_charge_interior_refinement and (capture_levels > 0 or charge_levels > 0):
                    env_msg = envelope_empty_message or "Capture envelope is empty. Reduce base Cell Size and try again."
                    chg_empty = charge_region_empty_message or "True charge region has no cells. Reduce base Cell Size or increase Inside."
                    # Stage 1a: [topoSet captureEnvelope + refineMesh once] x capture_levels
                    # Re-selecting on the current mesh before each refinement is required: after each
                    # refineMesh call the mesh changes and old cell IDs are stale.
                    for lv in range(capture_levels):
                        script_body += "# --- Capture envelope level %d/%d: re-select on current mesh, refine once ---\n" % (lv + 1, capture_levels)
                        script_body += "topoSet -dict system/topoSet_captureEnvelopeDict >> log.topoSet_captureEnvelope 2>&1\n"
                        if lv == 0:
                            script_body += 'test -s constant/polyMesh/sets/captureEnvelope || { echo "FATAL: %s"; exit 1; }\n' % env_msg.replace('"', '\\"')
                        else:
                            script_body += 'test -s constant/polyMesh/sets/captureEnvelope || { echo "FATAL: Capture envelope empty at level %d/%d. Increase Inside or reduce Cell Size."; exit 1; }\n' % (lv + 1, capture_levels)
                        script_body += "refineMesh -dict system/refineMesh_captureEnvelopeDict -overwrite >> log.refineMesh_captureEnvelope 2>&1\n"
                    # Stage 1b: [topoSet chargeRegion + refineMesh once] x charge_levels
                    if charge_levels > 0:
                        for lv in range(charge_levels):
                            script_body += "# --- Charge region level %d/%d: re-select on current mesh, refine once ---\n" % (lv + 1, charge_levels)
                            script_body += "topoSet -dict system/topoSet_chargeRegionDict >> log.topoSet_chargeRegion 2>&1\n"
                            if lv == 0:
                                script_body += 'test -s constant/polyMesh/sets/chargeRegion || { echo "FATAL: %s"; exit 1; }\n' % chg_empty.replace('"', '\\"')
                            else:
                                script_body += 'test -s constant/polyMesh/sets/chargeRegion || { echo "FATAL: Charge region empty at level %d/%d. Adjust charge geometry or Inside."; exit 1; }\n' % (lv + 1, charge_levels)
                            script_body += "refineMesh -dict system/refineMesh_chargeDict -overwrite >> log.refineMesh_charge 2>&1\n"
                    # Outside shell: [topoSet outsideShell + refineMesh once] x outside_levels
                    if outside_levels > 0:
                        for lv in range(outside_levels):
                            script_body += "# --- Outside shell level %d/%d ---\n" % (lv + 1, outside_levels)
                            script_body += "topoSet -dict system/topoSet_outsideShellDict >> log.topoSet_outsideShell 2>&1\n"
                            script_body += "refineMesh -dict system/refineMesh_outsideDict -overwrite >> log.refineMesh_outside 2>&1\n"
                    # Final topoSet_chargeRegion on the fully-refined mesh: provides accurate cell
                    # count for the realization check even when charge_levels=0 (only capture ran).
                    script_body += "# --- Final cell count in true charge region (for realization check) ---\n"
                    script_body += "topoSet -dict system/topoSet_chargeRegionDict >> log.topoSet_chargeRegion 2>&1\n"
                    # Post-refinement realization check: verifies requested Inside level was achieved.
                    script_body += "[ -f check_realization.py ] && { python3 check_realization.py || exit 1; } || true\n"
                    script_body += "# Remove hexRef metadata so setRefinedFields can place charge\n"
                    script_body += "rm -f constant/polyMesh/cellLevel constant/polyMesh/pointLevel constant/polyMesh/level0Edge constant/polyMesh/refinementHistory 2>/dev/null || true\n"
                script_body += """
# --- Add internal patch FIRST so mesh has internalPatch before 0/ restore ---
addEmptyPatch internalPatch internal -overwrite > log.addEmptyPatch 2>&1
stage_check "3_after_addEmptyPatch"
rm -rf 0
cp -r 0.orig 0
stage_check "4_after_restore_0"
# --- changeDictionary adds boundaryField for ALL patches (internalPatch + obs.*) ---
changeDictionary > log.changeDictionary 2>&1
stage_check "5_after_changeDictionary"
checkMesh > log.checkMesh 2>&1
"""

            script_body += init_fields_block
            script_body += """
# 7. Run Solver
""" + solver_app + " 2>&1 | tee log." + solver_app + """
"""

        # Safe cleanup only: artifacts and mesh; preserve 0.orig, system/, constant/*.dict, constant/triSurface/*.stl
        allclean = """#!/usr/bin/env bash
# Safe Allclean: remove run artifacts only. Preserves 0.orig, system/, constant dicts, constant/triSurface/*.stl
cd "$(dirname "$0")" || exit 1
echo "Running Allclean"
rm -rf processor* [1-9]* 0.[0-9]* constant/polyMesh constant/extendedFeatureEdgeMesh constant/triSurface/*.eMesh postProcessing dynamicCode 2>/dev/null || true
rm -f log.* log.stageVerification verification_run.log *.foam case.foam 2>/dev/null || true
# Remove 0 so Allrun can restore from 0.orig (0 is not an input; 0.orig is)
rm -rf 0 2>/dev/null || true
"""
        
        self._write_text(os.path.join(case_dir, "Allrun"), script_head + script_body)
        self._write_text(os.path.join(case_dir, "Allclean"), allclean)