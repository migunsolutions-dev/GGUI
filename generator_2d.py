"""OpenFOAM/blastFoam case generator for an r-z axisymmetric wedge."""
from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
from dataclasses import asdict
from typing import Dict

from axisymmetric_2d import (
    BOUNDARY_OPEN,
    DIRECT_SOURCE,
    DYNAMIC_MESH,
    validate_case_inputs_2d,
)
from base_generator import ALPHA_C4_CHECK_SCRIPT, BaseGenerator
from charge_capture import CAPTURE_CELL_SAFETY, auto_charge_capture_radius_m
from material_catalog import jwl_parameters
from material_validation import validate_required_values
from models_2d import CaseInputs2D, HOB_SOURCE_USER_TARGET
from output_options import REMAP_2D_FILENAME, extra_function_objects
from path_utils import win_to_wsl_path
from remap_fields_2d import charge_center_xyz, remap_region_metadata
from remap_handoff_1d import (
    merge_target_handoff,
    read_handoff_metadata,
    write_handoff_metadata,
)
from remap_snapshot_1d import canonical_case_path, resolve_remap_source
from validation.auto_points import FO_2D_VALIDATION, plan_2d, runtime_logical_dpi_x, stamp_plan
from validation.kb_propagation import copied_1d2d_radius_m
from validation.remap_timing import build_remap_timing, remap_timing_from_mapping
from validation.sampling_io import write_sampling_plan


WEDGE_HALF_ANGLE_DEG = 5.0  # exact blastFoam 6.2.0 axisymmetricCharge convention


class Generator2D(BaseGenerator):
    """Generate static or adaptive axisymmetric blastFoam cases."""

    def __init__(self, base_path: str, openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc"):
        super().__init__(base_path)
        self.openfoam_bashrc = openfoam_bashrc

    def generate(self, case_name: str, inputs: CaseInputs2D) -> str:
        validate_required_values(
            inputs,
            undefined_keys=getattr(inputs, "undefined_keys", ()) or (),
            require_imported_physics=bool(getattr(inputs, "undefined_keys", ())),
        ).raise_if_invalid()
        checked = validate_case_inputs_2d(inputs).require_valid()
        assert checked.domain is not None
        case_dir = self.create_case_dirs(case_name)
        self._write_block_mesh(case_dir, checked.domain)
        self._write_initial_fields(case_dir, inputs)
        self._write_phase_properties(case_dir, inputs)
        self._write_dynamic_mesh(case_dir, inputs)
        self._write_set_fields(case_dir, inputs, checked)
        self._write_system_files(case_dir, inputs)
        if inputs.initialization_source != DIRECT_SOURCE:
            self._write_remap_scripts(case_dir, inputs)
        self._write_scripts_2d(case_dir, inputs)
        self._write_metadata(case_dir, inputs, checked)
        pathlib.Path(case_dir, "case.foam").touch()
        return case_dir

    def initialization_command(self, inputs: CaseInputs2D) -> str:
        """Command used by the GUI to initialize, without starting blastFoam.

        setRefinedFields is used only when Dynamic Mesh startup seed level > 0.
        Effective Auto seed L0 (already resolved base mesh) must use setFields;
        otherwise setRefinedFields aborts with "Maximum refinement could not be
        determined" when no region level/maxRefinement is present.
        """
        restore = "rm -rf 0 && cp -r 0.orig 0"
        internal_patch = (
            " && addEmptyPatch internalPatch internal -overwrite"
            if inputs.mesh_mode == DYNAMIC_MESH and inputs.cores > 1
            else ""
        )
        if inputs.initialization_source == DIRECT_SOURCE:
            checked = validate_case_inputs_2d(inputs)
            seed_level = (
                checked.seed_plan.level_effective
                if checked.seed_plan is not None
                else 0
            )
            use_refined = inputs.mesh_mode == DYNAMIC_MESH and int(seed_level) > 0
            setter = "setRefinedFields" if use_refined else "setFields"
            return (
                f"{restore} && blockMesh && {setter}{internal_patch} && "
                "bash ./check_alpha_c4.sh && checkMesh"
            )
        # Radial remap about the user HOB: sample the 1D profile at
        # r_source = hypot(r, z - HOB). Do not use rotateFields, which maps
        # from the origin [0, 0, 0] and leaves an elevated charge on the ground.
        base = (
            f"{restore} && blockMesh && postProcess -func writeCellCentres "
            "&& python3 remap_2d.py"
        )
        return base + internal_patch + " && checkMesh"

    def _write_remap_scripts(self, case_dir: str, inputs: CaseInputs2D) -> None:
        src = pathlib.Path(__file__).with_name("remap_fields_2d.py")
        dest = pathlib.Path(case_dir) / "remap_fields_2d.py"
        if not src.is_file():
            raise RuntimeError("remap_fields_2d.py is missing next to generator_2d.py")
        shutil.copyfile(src, dest)
        snap_src = pathlib.Path(__file__).with_name("remap_snapshot_1d.py")
        if snap_src.is_file():
            shutil.copyfile(snap_src, pathlib.Path(case_dir) / "remap_snapshot_1d.py")
        source = win_to_wsl_path(inputs.mapping.case_path)
        source_time = (
            inputs.mapping.specific_time
            if str(getattr(inputs.mapping, "time_mode", "") or "") == "specific"
            else "latest"
        )
        if not source_time or source_time in ("latestTime", "latesttime"):
            source_time = "latest"
        hob = float(inputs.height_of_burst)
        mapped = float(getattr(inputs.mapping, "mapped_radius", 0.0) or 0.0)
        driver = (
            "#!/usr/bin/env python3\n"
            "from remap_fields_2d import run_case_remap\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(run_case_remap(\n"
            f"        source_case={source!r},\n"
            f"        source_time={source_time!r},\n"
            f"        hob={hob:.12g},\n"
            f"        mapped_radius={mapped:.12g},\n"
            "    ))\n"
        )
        self._write_text(os.path.join(case_dir, "remap_2d.py"), driver)

    def _write_block_mesh(self, case_dir: str, domain) -> None:
        """Wedge topology and orientation copied from blastFoam axisymmetricCharge."""
        angle = math.radians(WEDGE_HALF_ANGLE_DEG)
        x = domain.effective_radius * math.cos(angle)
        z = domain.effective_radius * math.sin(angle)
        h = domain.effective_height
        content = self._foam_header("blockMeshDict", "dictionary", "system") + f"""
convertToMeters 1;

vertices
(
    (0 0 0)
    ({x:.12g} 0 {-z:.12g})
    ({x:.12g} {h:.12g} {-z:.12g})
    (0 {h:.12g} 0)
    ({x:.12g} 0 {z:.12g})
    ({x:.12g} {h:.12g} {z:.12g})
);

blocks
(
    hex (0 1 2 3 0 4 5 3)
        ({domain.radial_cells} {domain.vertical_cells} 1)
        simpleGrading (1 1 1)
);

edges ();

boundary
(
    ground
    {{
        type patch;
        faces ((0 1 4 0));
    }}
    outerRadius
    {{
        type patch;
        faces ((1 2 5 4));
    }}
    top
    {{
        type patch;
        faces ((3 2 5 3));
    }}
    wedge0
    {{
        type wedge;
        faces ((0 1 2 3));
    }}
    wedge1
    {{
        type wedge;
        faces ((0 4 5 3));
    }}
);

mergePatchPairs ();
"""
        self._write_text(os.path.join(case_dir, "system", "blockMeshDict"), content)

    @staticmethod
    def _scalar_boundary(name: str, internal: float, inputs: CaseInputs2D) -> str:
        presets = {
            "ground": inputs.bottom_boundary,
            "outerRadius": inputs.outer_boundary,
            "top": inputs.top_boundary,
        }
        lines = ["boundaryField", "{"]
        for patch, preset in presets.items():
            if preset == BOUNDARY_OPEN and name == "p":
                lines.append(
                    f"    {patch} {{ type pressureWaveTransmissive; value uniform {internal:.12g}; }}"
                )
            else:
                lines.append(f"    {patch} {{ type zeroGradient; }}")
        lines.extend(
            (
                "    wedge0 { type wedge; }",
                "    wedge1 { type wedge; }",
                "    defaultFaces { type empty; }",
                "}\n",
            )
        )
        return "\n".join(lines)

    @staticmethod
    def _vector_boundary(inputs: CaseInputs2D) -> str:
        presets = {
            "ground": inputs.bottom_boundary,
            "outerRadius": inputs.outer_boundary,
            "top": inputs.top_boundary,
        }
        lines = ["boundaryField", "{"]
        for patch, preset in presets.items():
            bc = "slip" if preset != BOUNDARY_OPEN else "zeroGradient"
            lines.append(f"    {patch} {{ type {bc}; }}")
        lines.extend(
            (
                "    wedge0 { type wedge; }",
                "    wedge1 { type wedge; }",
                "    defaultFaces { type empty; }",
                "}\n",
            )
        )
        return "\n".join(lines)

    def _write_initial_fields(self, case_dir: str, inputs: CaseInputs2D) -> None:
        zero = os.path.join(case_dir, "0.orig")
        rho_air = inputs.p_atm / (287.05 * inputs.t_atm)
        scalar_specs = (
            ("p", "[1 -1 -2 0 0 0 0]", inputs.p_atm),
            ("T", "[0 0 0 1 0 0 0]", inputs.t_atm),
            ("rho.c4", "[1 -3 0 0 0 0 0]", inputs.rho_charge),
            ("rho.air", "[1 -3 0 0 0 0 0]", rho_air),
            ("alpha.c4", "[0 0 0 0 0 0 0]", 0.0),
        )
        for name, dimensions, value in scalar_specs:
            content = (
                self._foam_header(name, "volScalarField", "0")
                + f"dimensions {dimensions};\ninternalField uniform {value:.12g};\n"
                + self._scalar_boundary(name, value, inputs)
            )
            self._write_text(os.path.join(zero, name), content)
        u = (
            self._foam_header("U", "volVectorField", "0")
            + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
            + self._vector_boundary(inputs)
        )
        self._write_text(os.path.join(zero, "U"), u)

    def _write_phase_properties(self, case_dir: str, inputs: CaseInputs2D) -> None:
        validate_required_values(
            inputs,
            undefined_keys=getattr(inputs, "undefined_keys", ()) or (),
        ).raise_if_invalid()
        j = jwl_parameters(inputs.material_name, inputs.material_props)
        cv = j["CvCoeffs"]
        remap = inputs.initialization_source != DIRECT_SOURCE
        use_com = "no" if remap else "yes"
        points = (
            f"        points ((0 {inputs.height_of_burst:.12g} 0));\n"
            if remap
            else ""
        )
        content = self._foam_header("phaseProperties", "dictionary", "constant") + f"""
phases (c4 air);
c4
{{
    type detonating;
    reactants
    {{
        thermoType {{ transport const; thermo eConst; equationOfState BirchMurnaghan3; }}
        equationOfState {{ rho0 {inputs.rho_charge:.12g}; Gamma 0.25; pRef 101298; K0 8.04e9; K0Prime 7.97; }}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        thermodynamics {{ Cv 1400; Hf 0.0; }}
    }}
    products
    {{
        thermoType {{ transport const; thermo ePolynomial; equationOfState JWL; }}
        equationOfState {{ rho0 {inputs.rho_charge:.12g}; A {j['A']:.12g}; B {j['B']:.12g}; R1 {j['R1']}; R2 {j['R2']}; omega {j['omega']}; }}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        thermodynamics {{ CvCoeffs<8> ({cv[0]} {cv[1]} 0 0 0 0 0 0); Sf 0.0; Hf 0.0; }}
    }}
    activationModel {'none' if remap else 'pressureBased'};
    initiation
    {{
        E0 {j['E0']:.12g};
        I 4.0e6; a 0.0367; b 0.667; x 7.0; maxLambdaI 0.022;
        G1 1.4997e-7; c 0.667; d 0.33; y 2.0; minLambda1 0.022;
        G2 0.0; e 0.667; f 0.667; z 3.0; minLambda2 0.022;
        pMin {inputs.p_atm:.12g};
        useCOM {use_com};
{points}        radius {max(0.01, 3.0 * inputs.cell_size / (2 ** max(0, inputs.dyn_refine_max))):.12g};
        vDet 7850;
    }}
    residualRho 1e-6;
    residualAlpha 1e-6;
}}
air
{{
    type basic;
    thermoType {{ transport const; thermo eConst; equationOfState idealGas; }}
    equationOfState {{ gamma 1.4; }}
    specie {{ molWeight 28.97; }}
    transport {{ mu 0; Pr 1; }}
    thermodynamics {{ type eConst; Cv 718; Hf 0; }}
    residualRho 1e-6;
    residualAlpha 1e-6;
}}
"""
        const = os.path.join(case_dir, "constant")
        self._write_text(os.path.join(const, "phaseProperties"), content)
        self._write_text(
            os.path.join(const, "turbulenceProperties"),
            self._foam_header("turbulenceProperties", "dictionary", "constant")
            + "simulationType laminar;\n",
        )

    def _write_dynamic_mesh(self, case_dir: str, inputs: CaseInputs2D) -> None:
        if inputs.mesh_mode != DYNAMIC_MESH:
            body = "dynamicFvMesh staticFvMesh;\n"
        else:
            optional = f"maxCells {inputs.dynamic_max_cells};\n"
            if inputs.begin_unrefine is not None:
                optional += f"beginUnrefine {inputs.begin_unrefine:.12g};\n"
            if inputs.upper_refine_level is not None:
                optional += f"upperRefineLevel {inputs.upper_refine_level:.12g};\n"
            if inputs.upper_unrefine_level is not None:
                optional += f"upperUnrefineLevel {inputs.upper_unrefine_level:.12g};\n"
            if inputs.enable_balancing:
                interval = inputs.balance_interval or 10
                optional += (
                    f"balanceInterval {interval};\n"
                    "loadBalance\n{\n"
                    "    balance yes;\n"
                    "    allowableImbalance 0.2;\n"
                    "    method scotch;\n"
                    "}\n"
                )
            body = f"""dynamicFvMesh adaptiveFvMesh;
errorEstimator densityGradient;
refineInterval {inputs.refine_interval};
unrefineInterval {inputs.unrefine_interval};
lowerRefineLevel {inputs.lower_refine_threshold:.12g};
unrefineLevel {inputs.unrefine_threshold:.12g};
nBufferLayers {inputs.n_buffer_layers_dynamic};
maxRefinement {inputs.dyn_refine_max};
dumpLevel {'true' if inputs.dump_level else 'false'};
refineProbes {'true' if inputs.refine_probes else 'false'};
{optional}"""
        self._write_text(
            os.path.join(case_dir, "constant", "dynamicMeshDict"),
            self._foam_header("dynamicMeshDict", "dictionary", "constant") + body,
        )

    def _write_set_fields(self, case_dir: str, inputs: CaseInputs2D, checked) -> None:
        if inputs.initialization_source != DIRECT_SOURCE:
            regions = ""
            level = 0
        else:
            charge = checked.charge
            assert charge is not None
            level = checked.seed_plan.level_effective if checked.seed_plan else 0
            refine = (
                f"refineInternal yes; level {level};"
                if inputs.mesh_mode == DYNAMIC_MESH and level > 0
                else ""
            )
            zc = inputs.height_of_burst
            if charge.shape == "Sphere":
                r = charge.radius_m
                if refine:
                    backup = auto_charge_capture_radius_m(
                        r, inputs.cell_size, 0.0, inputs.cell_size, 1.0
                    )
                    regions = (
                        "sphericalMassToCell\n    {\n"
                        f"        rho {inputs.rho_charge:.12g}; mass {inputs.mass_kg:.12g};\n"
                        f"        centre (0 {zc:.12g} 0);\n"
                        f"        backup {{ centre (0 {zc:.12g} 0); radius {backup:.12g}; }}\n"
                        f"        {refine}\n"
                        "        fieldValues (volScalarFieldValue alpha.c4 1);\n"
                        "    }"
                    )
                else:
                    regions = (
                        f"sphereToCell {{ centre (0 {zc:.12g} 0); radius {r:.12g}; "
                        "fieldValues (volScalarFieldValue alpha.c4 1); }"
                    )
            else:
                r = charge.cylinder_radius_m
                half = charge.length_m / 2.0
                if refine:
                    backup = auto_charge_capture_radius_m(
                        r, inputs.cell_size, 0.0, inputs.cell_size, 1.0
                    )
                    backup_len = max(
                        charge.length_m,
                        2.0 * r,
                        CAPTURE_CELL_SAFETY * inputs.cell_size,
                    )
                    regions = (
                        "cylindericalMassToCell\n    {\n"
                        f"        rho {inputs.rho_charge:.12g}; mass {inputs.mass_kg:.12g};\n"
                        f"        centre (0 {zc:.12g} 0); direction (0 1 0); LbyD {inputs.charge_aspect:.12g};\n"
                        f"        backup {{ centre (0 {zc:.12g} 0); L (0 {backup_len:.12g} 0); radius {backup:.12g}; }}\n"
                        f"        {refine}\n"
                        "        fieldValues (volScalarFieldValue alpha.c4 1);\n"
                        "    }"
                    )
                else:
                    regions = (
                        "cylinderToCell { "
                        f"p1 (0 {zc - half:.12g} 0); p2 (0 {zc + half:.12g} 0); "
                        f"radius {r:.12g}; "
                        "fieldValues (volScalarFieldValue alpha.c4 1); }"
                    )
        content = self._foam_header("setFieldsDict", "dictionary", "system") + f"""
fields (alpha.c4);
nBufferLayers {inputs.buffer_layers};
defaultFieldValues (volScalarFieldValue alpha.c4 0);
regions ({regions});
"""
        self._write_text(os.path.join(case_dir, "system", "setFieldsDict"), content)

    def _write_system_files(self, case_dir: str, inputs: CaseInputs2D) -> None:
        system = os.path.join(case_dir, "system")
        fv_solution = self._foam_header("fvSolution", "dictionary", "system") + r"""
solvers
{
    "(rho|rhoU|rhoE|alpha|.*)" { solver diagonal; }
    p { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0.05; }
}
PIMPLE { nCorrectors 3; nNonOrthogonalCorrectors 0; }
"""
        fv_schemes = self._foam_header("fvSchemes", "dictionary", "system") + r"""
fluxScheme Tadmor;
ddtSchemes { default Euler; timeIntegrator Euler; }
gradSchemes { default cellMDLimited leastSquares 1.0; }
divSchemes { default none; div(alphaRhoPhi.c4,lambda.c4) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes
{
    default linear;
    "reconstruct(alpha.c4)" vanLeer;
    "reconstruct(rho)" vanLeer;
    "reconstruct(U)" vanLeer;
    "reconstruct(e)" vanLeer;
    "reconstruct(p)" vanLeer;
    "reconstruct(T)" vanLeer;
    "reconstruct(speedOfSound)" vanLeer;
}
snGradSchemes { default corrected; }
"""
        self._write_text(os.path.join(system, "fvSolution"), fv_solution)
        self._write_text(os.path.join(system, "fvSchemes"), fv_schemes)

        write_control = inputs.write_control_type
        write_interval = (
            inputs.write_interval_time
            if write_control == "adjustableRunTime"
            else inputs.write_interval_steps
        )
        probes = []
        for probe in inputs.probes:
            probes.append(
                f"            ({probe.radius:.12g} {probe.height:.12g} 0)"
            )
        # Pressure histories only. Probe-driven refinement is the separate
        # dynamicMeshDict Switch ``refineProbes`` (errorEstimator), which marks
        # cells at existing ``probes`` / ``blastProbes`` function-object locations.
        # There is no supported controlDict function type ``refineProbes``.
        probe_fields = list(inputs.output_fields) if inputs.output_fields else ["p"]
        if bool(getattr(inputs, "enable_impulse", False)) and "impulse" not in probe_fields:
            probe_fields.append("impulse")
        if bool(getattr(inputs, "enable_dynamic_pressure", False)) and "dynamicPressure" not in probe_fields:
            probe_fields.append("dynamicPressure")
        val_plan = plan_2d(
            mass_kg=float(inputs.mass_kg or 0.0),
            domain_radius_m=float(inputs.radius),
            domain_height_m=float(inputs.height),
            hob_m=float(inputs.height_of_burst),
            cell_size=inputs.cell_size,
            logical_dpi_x=runtime_logical_dpi_x(),
            remap_receive_r_max=(
                copied_1d2d_radius_m(
                    source_1d_case=str(inputs.mapping.case_path or ""),
                    widget_mapped_radius=float(
                        getattr(inputs.mapping, "mapped_radius", 0.0) or 0.0
                    ),
                )
                if inputs.initialization_source != DIRECT_SOURCE
                else None
            ),
        )
        remap_timing = None
        if inputs.initialization_source != DIRECT_SOURCE:
            src_label = (
                inputs.mapping.specific_time
                if str(getattr(inputs.mapping, "time_mode", "") or "") == "specific"
                else ""
            )
            remap_timing = remap_timing_from_mapping(
                mapping_time=src_label or None,
                mapping_time_mode=getattr(inputs.mapping, "time_mode", None),
                target_time_label="0",
            )
            if remap_timing.source_physical_time is None:
                handoff = read_handoff_metadata(str(inputs.mapping.case_path or "")) or {}
                phys = handoff.get("source_physical_time")
                if phys is None:
                    phys = handoff.get("handoff_time_s")
                try:
                    phys_f = float(phys)
                except (TypeError, ValueError):
                    phys_f = None
                if phys_f is not None and phys_f == phys_f and phys_f > 0.0:
                    remap_timing = build_remap_timing(
                        source_time_label=f"{phys_f:.12g}",
                        target_time_label="0",
                    )
        val_plan = stamp_plan(
            val_plan,
            case_path=case_dir,
            cell_size=inputs.cell_size,
            hob_m=float(inputs.height_of_burst),
            domain_height_m=float(inputs.height),
            remap_receive_r_max=val_plan.remap_receive_r_max,
            remap_timing=None if remap_timing is None else remap_timing.as_dict(),
        )
        write_impulse = (
            bool(getattr(inputs, "enable_impulse", False))
            or "impulse" in probe_fields
            or bool(val_plan.points)
        )
        extras = extra_function_objects(
            p_atm=float(inputs.p_atm),
            impulse=write_impulse,
            overpressure=False,
            dynamic_pressure=bool(getattr(inputs, "enable_dynamic_pressure", False)),
            peaks=False,
        )
        remap_block = ""
        if bool(getattr(inputs, "output_remap_data", False)):
            remap_block = """    remapDump
    {
        type            writeObjects;
        libs            ("libutilityFunctionObjects.so");
        objects         (p U rho T alpha.c4);
        writeControl    onEnd;
    }
"""
        probes_block = ""
        if probes:
            probes_block = f"""
    probes2d
    {{
        type probes;
        libs ("libfieldFunctionObjects.so");
        fields ({' '.join(probe_fields)});
        writeControl timeStep;
        writeInterval 1;
        probeLocations
        (
{os.linesep.join(probes)}
        );
    }}
"""
        validation_block = ""
        if val_plan.points:
            val_pts = [
                f"            ({pt.x:.12g} {pt.y:.12g} {pt.z:.12g})" for pt in val_plan.points
            ]
            validation_block = f"""
    {FO_2D_VALIDATION}
    {{
        type probes;
        libs ("libfieldFunctionObjects.so");
        fields (p impulse);
        writeControl timeStep;
        writeInterval 1;
        probeLocations
        (
{os.linesep.join(val_pts)}
        );
    }}
"""
        retained_cycle = (
            int(inputs.cycle_write)
            if bool(getattr(inputs, "keep_openfoam_time_folders", False))
            else 0
        )
        control = self._foam_header("controlDict", "dictionary", "system") + f"""
application blastFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {inputs.end_time_s:.12g};
deltaT {inputs.delta_t:.12g};
adjustTimeStep {'yes' if inputs.adjust_time_step else 'no'};
maxCo {inputs.max_co:.12g};
writeControl {write_control};
writeInterval {write_interval};
purgeWrite {retained_cycle};
writeFormat ascii;
writePrecision 12;
writeCompression off;
timeFormat general;
timePrecision 10;
runTimeModifiable true;
functions
{{
{extras}{probes_block}{validation_block}{remap_block}}}
"""
        self._write_text(os.path.join(system, "controlDict"), control)
        if val_plan.points or val_plan.notes:
            write_sampling_plan(case_dir, val_plan)
        decompose = (
            self._foam_header("decomposeParDict", "dictionary", "system")
            + f"numberOfSubdomains {inputs.cores};\nmethod scotch;\n"
        )
        self._write_text(os.path.join(system, "decomposeParDict"), decompose)

    def _write_scripts_2d(self, case_dir: str, inputs: CaseInputs2D) -> None:
        self._write_text(os.path.join(case_dir, "check_alpha_c4.sh"), ALPHA_C4_CHECK_SCRIPT)
        init = self.initialization_command(inputs)
        solver = (
            f"mpirun -np {inputs.cores} blastFoam -parallel"
            if inputs.cores > 1
            else "blastFoam"
        )
        decompose = "decomposePar -force && " if inputs.cores > 1 else ""
        allrun = f"""#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
source "{self.openfoam_bashrc}" || true
set -eo pipefail
{init}
{decompose}{solver} 2>&1 | tee log.blastFoam
solver_rc=${{PIPESTATUS[0]}}
exit "$solver_rc"
"""
        if bool(getattr(inputs, "output_remap_data", False)):
            allrun += f"""
latest=$(ls -1d [0-9]* 0.[0-9]* 2>/dev/null | sort -g | tail -1)
{{
  echo '/* GGUI 2D remap snapshot ({REMAP_2D_FILENAME}) */'
  echo "time            ${{latest:-latest}};"
  echo 'sourceCase      ".";'
  echo 'fields          (p rho U T alpha.c4);'
}} > {REMAP_2D_FILENAME}
"""
        allclean = f"""#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
rm -rf processor* [1-9]* 0.[0-9]* constant/polyMesh postProcessing dynamicCode 2>/dev/null || true
rm -f log.* *.foam {REMAP_2D_FILENAME} 2>/dev/null || true
rm -rf 0 2>/dev/null || true
"""
        self._write_text(os.path.join(case_dir, "Allrun"), allrun)
        self._write_text(os.path.join(case_dir, "Allclean"), allclean)

    def _write_metadata(self, case_dir: str, inputs: CaseInputs2D, checked) -> None:
        hob = float(inputs.height_of_burst)
        remap = inputs.initialization_source != DIRECT_SOURCE
        remap_timing = None
        remap_region = None
        if remap:
            src_label = (
                inputs.mapping.specific_time
                if str(getattr(inputs.mapping, "time_mode", "") or "") == "specific"
                else ""
            )
            remap_timing = remap_timing_from_mapping(
                mapping_time=src_label or None,
                mapping_time_mode=getattr(inputs.mapping, "time_mode", None),
                target_time_label="0",
            )
            remap_region = remap_region_metadata(
                hob,
                mapped_radius=float(getattr(inputs.mapping, "mapped_radius", 0.0) or 0.0),
                source_time=src_label or str(getattr(inputs.mapping, "time_mode", "") or "latest"),
                time_mode=str(getattr(inputs.mapping, "time_mode", "") or ""),
                target_time="0",
            )
            user_r = float(getattr(inputs.mapping, "mapped_radius", 0.0) or 0.0)
            if user_r > 0.0:
                remap_region["copied_radius_m"] = user_r
                remap_region["requested_mapped_radius_m"] = user_r
            src_meta = read_handoff_metadata(str(inputs.mapping.case_path or "")) or {}
            src_extent = src_meta.get("field_r_max_m")
            if src_extent is None:
                from remap_snapshot_1d import read_snapshot_metadata

                snap_meta = read_snapshot_metadata(str(inputs.mapping.case_path or "")) or {}
                src_extent = snap_meta.get("field_r_max_m")
            if src_extent is not None:
                remap_region["source_field_r_max_m"] = float(src_extent)
            remap_region["physical_time_offset"] = remap_timing.physical_time_offset
            remap_region["source_physical_time"] = remap_timing.source_physical_time
            remap_region["target_initial_time"] = remap_timing.target_initial_time
            resolved = resolve_remap_source(str(inputs.mapping.case_path or ""))
            if resolved.ok:
                remap_region["remap_source_type"] = resolved.source_type
                if resolved.physical_time is not None:
                    remap_region["source_physical_time"] = resolved.physical_time
                    remap_timing = build_remap_timing(
                        source_time_label=f"{resolved.physical_time:.12g}",
                        target_time_label="0",
                    )
                    remap_region["physical_time_offset"] = remap_timing.physical_time_offset
                    remap_region["source_physical_time"] = remap_timing.source_physical_time
                    remap_region["target_initial_time"] = remap_timing.target_initial_time
        data: Dict[str, object] = {
            "dimension": "2D-axisymmetric",
            "coordinates": "r-z",
            "wedge_half_angle_deg": WEDGE_HALF_ANGLE_DEG,
            "mirrored_display_only": bool(inputs.mirrored_view),
            "domain": checked.domain.to_metadata(),
            "initialization_source": inputs.initialization_source,
            "height_of_burst_m": hob,
            "charge_center": list(charge_center_xyz(hob)),
            "hob_source": HOB_SOURCE_USER_TARGET,
            "mesh_mode": inputs.mesh_mode,
            "startup_seed": checked.seed_plan.to_dict() if checked.seed_plan else None,
            "runtime_amr_max_refinement": (
                inputs.dyn_refine_max if inputs.mesh_mode == DYNAMIC_MESH else None
            ),
            "mapping": asdict(inputs.mapping),
            "remap_region": remap_region,
            "remap_timing": None if remap_timing is None else remap_timing.as_dict(),
            "warnings": list(checked.warnings),
        }
        if remap:
            src_handoff = read_handoff_metadata(str(inputs.mapping.case_path or ""))
            if src_handoff is None:
                resolved_src = resolve_remap_source(str(inputs.mapping.case_path or ""))
                meta = resolved_src.metadata if resolved_src is not None else None
                if isinstance(meta, dict) and meta.get("handoff_radius_m") is not None:
                    src_handoff = {
                        "remap_radius_m": meta.get("remap_radius_m"),
                        "dr_1d_m": meta.get("dr_1d_m"),
                        "remap_front_buffer_cells": meta.get("remap_front_buffer_cells"),
                        "handoff_radius_m": meta.get("handoff_radius_m"),
                        "handoff_time_s": meta.get("handoff_time_s")
                        or meta.get("arrival_time_s")
                        or meta.get("source_physical_time"),
                        "source_1d_case": meta.get("source_1d_case")
                        or meta.get("source_case_path"),
                        "field_r_max_m": meta.get("field_r_max_m"),
                    }
            if src_handoff:
                merged = merge_target_handoff(
                    src_handoff,
                    target_2d_case=canonical_case_path(case_dir),
                    hob_m=hob,
                    charge_center=list(charge_center_xyz(hob)),
                    actual_remap_geometry=remap_region,
                )
                write_handoff_metadata(case_dir, merged)
                data["remap_handoff"] = merged
        self._write_text(
            os.path.join(case_dir, "case_2d.json"),
            json.dumps(data, indent=2, sort_keys=True) + "\n",
        )
