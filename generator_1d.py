import json
import os
import math
from typing import Dict, Optional, Tuple

import ig_source_state as igs
import jwl_activation_energy as jwl_act
from base_generator import BaseGenerator
from models import (
    BOUNDARY_1D_REFLECT,
    BOUNDARY_1D_TERMINATE,
    CaseInputs1D,
    RecommendedParams1D,
    RUN_MODE_REFLECT,
    is_ideal_gas_source,
    normalize_source_model,
)
from output_options import drop_unavailable_phase_fields, extra_function_objects
from completion_1d import (
    initial_completion_record,
    normalize_run_mode,
    right_boundary_for_mode,
    write_completion_record,
)
from remap_handoff_1d import (
    HANDOFF_CRITERION,
    handoff_plan,
    handoff_radius_m,
    uses_remap_handoff,
    write_handoff_metadata,
)
from validation.auto_points import plan_1d, runtime_logical_dpi_x, stamp_plan
from validation.map_1d import merge_radii
from validation.sampling_io import write_sampling_plan

# OpenFOAM requires an endTime. The GUI value is always written; Terminate uses
# it as an upper bound, Reflect uses it as the successful completion time.

class Generator1D(BaseGenerator):
    """
    Handles 1D Wedge geometry generation.
    """
    def __init__(self, base_path: str, openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc"):
        super().__init__(base_path)
        self.openfoam_bashrc = openfoam_bashrc

    def generate(self, case_name: str, inputs: CaseInputs1D, rec: RecommendedParams1D) -> str:
        if uses_remap_handoff(inputs):
            handoff_radius_m(float(inputs.radius), float(inputs.cell_size))
        # 1. Create Dirs
        case_dir = self.create_case_dirs(case_name)
        
        # 2. Derived Calculations
        charge_radius = self.calculate_charge_radius(inputs.mass_kg, inputs.rho_charge)
        # IG needs r_min and dx up front: the burst state depends on which radial
        # shell the source cells occupy, so it cannot be derived per-file.
        ig_state = self.ig_state(inputs, rec)

        # 3. Write Files
        self.write_initial_conditions(case_dir, inputs, ig_state)

        if ig_state is not None:
            self.write_ig_constant_files(case_dir, ig_state)
        else:
            self.write_constant_files(
                case_dir,
                inputs.material_props,
                inputs.energy_j_per_kg,
                charge_radius,
                self.ignition_point_in_wedge(inputs, rec),
                rec.ignition_radius,
                rho_charge=float(inputs.rho_charge),
                material_name=str(getattr(inputs, "material_name", "") or ""),
            )

        val_plan = None
        try:
            val_plan = plan_1d(
                mass_kg=float(inputs.mass_kg),
                domain_radius_m=float(inputs.radius),
                cell_size=float(inputs.cell_size),
                logical_dpi_x=runtime_logical_dpi_x(),
            )
        except (TypeError, ValueError):
            val_plan = None
        if val_plan is not None:
            try:
                val_plan = stamp_plan(
                    val_plan,
                    case_path=case_dir,
                    cell_size=float(inputs.cell_size),
                    hob_m=0.0,
                    domain_height_m=None,
                    source_model=self.source_model(inputs),
                )
            except (TypeError, ValueError):
                pass

        val_radii = tuple(pt.range_m for pt in val_plan.points) if val_plan is not None else ()
        self.write_system_files(
            case_dir, inputs, rec, charge_radius,
            validation_radii=val_radii,
            ig_state=ig_state,
        )
        self._write_completion_request(case_dir, inputs)
        if uses_remap_handoff(inputs):
            write_handoff_metadata(
                case_dir,
                handoff_plan(
                    float(inputs.radius),
                    float(inputs.cell_size),
                    source_1d_case=case_dir,
                ),
            )
        
        # --- FIX: Write Scripts ---
        self.write_scripts(
            case_dir,
            self.openfoam_bashrc,
            use_ig_source_check=ig_state is not None,
            ig_source_check_p_atm=float(inputs.p_atm),
        )

        if ig_state is not None:
            self.write_ig_source_audit(case_dir, inputs, ig_state)
        else:
            self.write_jwl_energy_audit(case_dir, inputs)

        try:
            if val_plan is not None and val_plan.points:
                write_sampling_plan(case_dir, val_plan)
        except (OSError, TypeError, ValueError):
            pass
        
        # Create case.foam for ParaView compatibility
        import pathlib
        pathlib.Path(case_dir, "case.foam").touch()
        
        return case_dir

    @staticmethod
    def source_model(inputs: CaseInputs1D) -> str:
        return normalize_source_model(getattr(inputs, "source_model", None))

    @staticmethod
    def ig_state(
        inputs: CaseInputs1D, rec: RecommendedParams1D
    ) -> Optional[igs.IgBurstState]:
        """The burst state for an IG case, or ``None`` for JWL.

        ``None`` is what keeps every JWL dictionary byte-identical to the pre-feature
        generator: the IG code paths are only reachable through this object.
        """
        if not is_ideal_gas_source(getattr(inputs, "source_model", None)):
            return None
        return igs.derive_ig_state(
            mass_kg=float(inputs.mass_kg),
            rho_charge=float(inputs.rho_charge),
            energy_j_per_kg=float(inputs.energy_j_per_kg),
            p_atm=float(inputs.p_atm),
            t_atm=float(inputs.t_atm),
            r_min_m=float(rec.r_min),
            cell_size_m=float(inputs.cell_size),
        )

    def write_jwl_energy_audit(self, case_dir: str, inputs: CaseInputs1D) -> None:
        state = jwl_act.v2_activation(
            float(inputs.energy_j_per_kg),
            float(inputs.rho_charge),
            material_name=str(getattr(inputs, "material_name", "") or ""),
            dimension="1D",
        )
        jwl_act.write_jwl_energy_audit(
            case_dir, state, mass_kg=float(inputs.mass_kg)
        )

    def write_ig_source_audit(
        self, case_dir: str, inputs: CaseInputs1D, state: igs.IgBurstState
    ) -> None:
        """Sidecar so an old case can be identified without reading its dictionaries."""
        payload = igs.audit_dict(
            state,
            case_path=case_dir,
            material_name=str(getattr(inputs, "material_name", "") or ""),
        )
        self._write_text(
            os.path.join(case_dir, "ggui_ig_source_audit.json"),
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
        )

    @staticmethod
    def _run_mode(inputs: CaseInputs1D) -> str:
        return normalize_run_mode(
            getattr(inputs, "stop_mode", None),
            getattr(inputs, "right_boundary", None),
        )

    @staticmethod
    def _right_boundary(inputs: CaseInputs1D) -> str:
        if Generator1D._run_mode(inputs) == RUN_MODE_REFLECT:
            return BOUNDARY_1D_REFLECT
        return BOUNDARY_1D_TERMINATE

    @staticmethod
    def _vtx_spherical(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
        st, ct = math.sin(theta), math.cos(theta)
        sp, cp = math.sin(phi), math.cos(phi)
        return (r * ct, r * st * cp, r * st * sp)

    @staticmethod
    def wedge_angles(inputs: CaseInputs1D) -> Tuple[float, float, float]:
        """Return (axis_eps, cone_half, wedge_half) in radians, matching blockMesh."""
        wedge_half = math.radians(inputs.wedge_angle_deg) / 2.0
        cone_half = math.radians(inputs.cone_half_angle_deg)
        requested_eps = max(1e-9, float(inputs.axis_epsilon))
        min_axis_eps = min(0.10, cone_half * 0.45)
        axis_eps = min(max(requested_eps, min_axis_eps), cone_half * 0.5)
        return axis_eps, cone_half, wedge_half

    STANDALONE_OUTLET_PAD = 1.1

    @staticmethod
    def source_domain_radius_m(inputs: CaseInputs1D) -> float:
        """Outer mesh radius. Remap 1D must not solve past the user limit."""
        user = float(inputs.radius)
        if uses_remap_handoff(inputs):
            return user
        return user * Generator1D.STANDALONE_OUTLET_PAD

    @staticmethod
    def resolved_stop_radius_m(inputs: CaseInputs1D) -> float:
        domain = float(inputs.radius)
        if uses_remap_handoff(inputs):
            return handoff_radius_m(domain, float(inputs.cell_size))
        raw = getattr(inputs, "stop_radius_m", None)
        if raw is None:
            return domain
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return domain
        if not math.isfinite(value) or value <= 0.0:
            return domain
        return min(value, domain)

    @staticmethod
    def resolved_end_time_s(inputs: CaseInputs1D) -> float:
        try:
            value = float(inputs.end_time_s)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    @staticmethod
    def _write_completion_request(case_dir: str, inputs: CaseInputs1D) -> None:
        mode = Generator1D._run_mode(inputs)
        remap = uses_remap_handoff(inputs)
        plan = (
            handoff_plan(float(inputs.radius), float(inputs.cell_size), source_1d_case=case_dir)
            if remap
            else None
        )
        write_completion_record(
            case_dir,
            initial_completion_record(
                mode=mode,
                requested_stop_radius_m=Generator1D.resolved_stop_radius_m(inputs),
                p_atm=float(inputs.p_atm),
                right_boundary=right_boundary_for_mode(mode),
                end_time_s=Generator1D.resolved_end_time_s(inputs),
                remap_for_2d=remap,
                remap_radius_m=None if plan is None else plan["remap_radius_m"],
                dr_1d_m=None if plan is None else plan["dr_1d_m"],
                remap_front_buffer_cells=(
                    None if plan is None else plan["remap_front_buffer_cells"]
                ),
                handoff_radius_m=None if plan is None else plan["handoff_radius_m"],
                criterion=HANDOFF_CRITERION if remap else None,
                source_model=Generator1D.source_model(inputs),
            ),
        )

    def ignition_point_in_wedge(
        self, inputs: CaseInputs1D, rec: RecommendedParams1D
    ) -> Tuple[float, float, float]:
        """Place the detonation point inside a real wedge cell, not on the Cartesian axis.

        ``profiles.compute_recommended_1d`` returns ``(r_ign, 0, 0)``. That coordinate
        is outside the spherical-wedge mesh (cells start at ``axis_eps``), so blastFoam
        raises ``No cells will be activated using the detonation point`` on fine
        meshes where ``findCell`` cannot snap the on-axis point into a cell.
        """
        axis_eps, cone_half, _wedge_half = self.wedge_angles(inputs)
        theta_mid = 0.5 * (axis_eps + cone_half)
        r = math.sqrt(sum(float(c) * float(c) for c in rec.ignition_point))
        if r <= 0.0:
            r = max(float(rec.r_min), 1.0e-6)
        return self._vtx_spherical(r, theta_mid, 0.0)

    def write_initial_conditions(
        self,
        case_dir: str,
        inputs: CaseInputs1D,
        ig_state: Optional[igs.IgBurstState] = None,
    ) -> None:
        # Write to 0.orig so Allrun's "cp -r 0.orig 0" restores initial conditions (same as 3D flow).
        zero_dir = os.path.join(case_dir, "0.orig")
        rho_air = 1.225
        patches = ["origin", "outlet", "axis", "outerCone", "wedgeFront", "wedgeBack"]
        right = self._right_boundary(inputs)

        def scalar_bcs(name, val):
            lines = ["boundaryField", "{"]
            for pch in patches:
                if pch.startswith("wedge"):
                    lines.append(f"    {pch} {{ type wedge; }}")
                elif pch in ("axis", "outerCone", "origin"):
                    lines.append(f"    {pch} {{ type symmetry; }}")
                elif pch == "outlet":
                    if right != BOUNDARY_1D_REFLECT and name == "p":
                        lines.append(
                            f"    {pch} {{ type pressureWaveTransmissive; value uniform {inputs.p_atm}; }}"
                        )
                    else:
                        lines.append(f"    {pch} {{ type zeroGradient; }}")
                else:
                    lines.append(f"    {pch} {{ type zeroGradient; }}")
            lines.append("}\n")
            return "\n".join(lines)

        def vector_bcs():
            lines = ["boundaryField", "{"]
            for pch in patches:
                if pch.startswith("wedge"):
                    lines.append(f"    {pch} {{ type wedge; }}")
                elif pch in ("axis", "outerCone", "origin"):
                    lines.append(f"    {pch} {{ type symmetry; }}")
                elif pch == "outlet" and right == BOUNDARY_1D_REFLECT:
                    lines.append(f"    {pch} {{ type slip; }}")
                else:
                    lines.append(f"    {pch} {{ type zeroGradient; }}")
            lines.append("}\n")
            return "\n".join(lines)

        self._write_text(os.path.join(zero_dir, "p"), self._foam_header("p", "volScalarField", "0") + f"dimensions [1 -1 -2 0 0 0 0]; internalField uniform {inputs.p_atm};\n" + scalar_bcs("p", inputs.p_atm))
        self._write_text(os.path.join(zero_dir, "T"), self._foam_header("T", "volScalarField", "0") + f"dimensions [0 0 0 1 0 0 0]; internalField uniform {inputs.t_atm};\n" + scalar_bcs("T", inputs.t_atm))
        if ig_state is not None:
            # Single phase: one unsuffixed rho, at the EOS-consistent ambient density.
            # A hard-coded 1.225 would contradict p_atm/T_atm away from 101325 Pa / 288 K
            # and seed a spurious wave, because blastFoam derives e from p and rho.
            rho_atm = ig_state.ambient.rho
            self._write_text(os.path.join(zero_dir, "rho"), self._foam_header("rho", "volScalarField", "0") + f"dimensions [1 -3 0 0 0 0 0]; internalField uniform {rho_atm:.12g};\n" + scalar_bcs("rho", rho_atm))
        else:
            self._write_text(os.path.join(zero_dir, "rho.c4"), self._foam_header("rho.c4", "volScalarField", "0") + f"dimensions [1 -3 0 0 0 0 0]; internalField uniform {inputs.rho_charge};\n" + scalar_bcs("rho", inputs.rho_charge))
            self._write_text(os.path.join(zero_dir, "rho.air"), self._foam_header("rho.air", "volScalarField", "0") + f"dimensions [1 -3 0 0 0 0 0]; internalField uniform {rho_air};\n" + scalar_bcs("rho", rho_air))
            self._write_text(os.path.join(zero_dir, "alpha.c4"), self._foam_header("alpha.c4", "volScalarField", "0") + f"dimensions [0 0 0 0 0 0 0]; internalField uniform 0;\n" + scalar_bcs("alpha", 0))
        self._write_text(os.path.join(zero_dir, "U"), self._foam_header("U", "volVectorField", "0") + f"dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);\n" + vector_bcs())

    def write_ig_constant_files(self, case_dir: str, state: igs.IgBurstState) -> None:
        """Single-material ideal gas, in the form blastFoam's own Sedov case uses.

        Omitting the ``phases`` entry is what selects the single-phase path:
        ``compressibleSystemNew`` reads ``phases`` with a default of an empty list and
        builds ``singlePhaseCompressibleSystem`` when fewer than two are present. A
        confirmation run logs ``Selecting thermodynamics package
        fluid<basic<const<eConst<idealGas<specie>>>>>`` -- ``fluid<basic<...>>`` rather
        than a two-phase package. There is no activation model and no detonation point,
        because the charge is already fully detonated at t = 0.
        """
        const_dir = os.path.join(case_dir, "constant")
        self._write_text(
            os.path.join(const_dir, "turbulenceProperties"),
            self._foam_header("turbulenceProperties", "dictionary", "constant")
            + "simulationType laminar;\n",
        )

        pp_content = self._foam_header("phaseProperties", "dictionary", location="constant") + f"""
type            basic;
thermoType {{ transport const; thermo eConst; equationOfState idealGas; }}
equationOfState {{ gamma {state.ambient.gamma:.10g}; }}
specie          {{ molWeight 28.97; }}
transport       {{ mu 0; Pr 1; }}
thermodynamics  {{ Cv {state.ambient.cv:.10g}; Hf 0; }}
"""
        self._write_text(os.path.join(const_dir, "phaseProperties"), pp_content)
        self._write_text(
            os.path.join(const_dir, "dynamicMeshDict"),
            self._foam_header("dynamicMeshDict", "dictionary", "constant")
            + "dynamicFvMesh staticFvMesh;\n",
        )

    def write_constant_files(
        self,
        case_dir: str,
        mat_props: Dict,
        energy: float,
        charge_radius: float,
        ignition_point: Tuple,
        ignition_radius: float,
        rho_charge: Optional[float] = None,
        material_name: str = "",
    ) -> None:
        const_dir = os.path.join(case_dir, "constant")
        self._write_text(os.path.join(const_dir, "turbulenceProperties"), 
                         self._foam_header("turbulenceProperties", "dictionary", "constant") + "simulationType laminar;\n")

        ign_str = f"({ignition_point[0]:.10g} {ignition_point[1]:.10g} {ignition_point[2]:.10g})"
        rho0 = float(rho_charge) if rho_charge is not None else float(mat_props["rho"])
        act = jwl_act.v2_activation(energy, rho0, material_name=material_name, dimension="1D")
        
        pp_content = self._foam_header("phaseProperties", "dictionary", location="constant") + f"""
phases (c4 air);
c4
{{
    type detonating;
    reactants
    {{
        thermoType {{ transport const; thermo eConst; equationOfState BirchMurnaghan3; }}
        equationOfState {{ rho0 {rho0:.12g}; Gamma 0.25; pRef 101298; K0 8.04e9; K0Prime 7.97; }}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        thermodynamics {{ Cv 1400; Hf 0.0; }}
    }}
    products
    {{
        thermoType {{ transport const; thermo ePolynomial; equationOfState JWL; }}
        equationOfState 
        {{ 
            rho0 {rho0:.12g}; 
            A {mat_props['A']}; B {mat_props['B']}; 
            R1 {mat_props['R1']}; R2 {mat_props['R2']}; omega {mat_props['omega']}; 
        }}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        thermodynamics {{ CvCoeffs<8> (413.15 2.1538 0 0 0 0 0 0); Sf 0.0; Hf 0.0; }}
    }}
    activationModel pressureBased;
    initiation
    {{
        E0 {act.E0_pa:.12g}; 
        I 4.0e6; a 0.0367; b 0.667; x 7.0; maxLambdaI 0.022;
        G1 1.4997e-7; c 0.667; d 0.33; y 2.0; minLambda1 0.022;
        G2 0.0; e 0.667; f 0.667; z 3.0; minLambda2 0.022;
        pMin 1000; 
        useCOM no; 
        points ({ign_str}); 
        radius {ignition_radius:.10g}; 
        vDet 7850;
    }}
    residualRho 1e-6; residualAlpha 1e-6;
}}
air
{{
    type basic;
    thermoType {{ transport const; thermo eConst; equationOfState idealGas; }}
    equationOfState {{ gamma 1.4; }}
    specie {{ molWeight 28.97; }}
    transport {{ mu 0; Pr 1; }}
    thermodynamics {{ type eConst; Cv 718; Hf 0; }}
    residualRho 1e-6; residualAlpha 1e-6;
}}
"""
        self._write_text(os.path.join(const_dir, "phaseProperties"), pp_content)
        self._write_text(os.path.join(const_dir, "dynamicMeshDict"), 
                         self._foam_header("dynamicMeshDict", "dictionary", "constant") + "dynamicFvMesh staticFvMesh;\n")

    def write_system_files(
        self,
        case_dir: str,
        inputs: CaseInputs1D,
        rec: RecommendedParams1D,
        charge_radius: float,
        validation_radii: Tuple[float, ...] = (),
        ig_state: Optional[igs.IgBurstState] = None,
    ) -> None:
        sys_dir = os.path.join(case_dir, "system")
        # Remap: user radius is the physical outer limit. Standalone Terminate
        # keeps a 10% outlet pad so the stop probe is not on the boundary.
        target_radius = float(inputs.radius)
        physical_radius = self.source_domain_radius_m(inputs)
        r_max_val = physical_radius
        dx = max(float(inputs.cell_size), 1e-9)
        r_min = rec.r_min
        if r_max_val <= r_min:
            r_max_val = r_min + 10.0 * dx
        n_r = max(20, int((r_max_val - r_min) / dx))

        # Axis face length is r*sin(θ)*Δφ. Thin wedges + tiny θ freeze CFL at the origin.
        axis_eps, cone_half, wedge_half = self.wedge_angles(inputs)
        vtx_spherical = self._vtx_spherical

        # blockMesh: hex (0 3 2 1 4 7 6 5). All face normals must point outward.
        # Outlet (0 1 2 3): (0->1)x(1->2)=+r => 0->1=+theta, 1->2=+phi => 0=(ae,-w), 1=(ch,-w), 2=(ch,w), 3=(ae,w).
        # Axis (0 4 7 3): (0->4)x(4->7)=-e_theta => 4->7=-phi => 4=(ae,w), 7=(ae,-w). Origin (4 7 6 5): 4->7=-phi, 7->6=+theta => -r.
        vertices = (
            vtx_spherical(r_max_val, axis_eps, -wedge_half),   # 0 outlet (ae,-w)
            vtx_spherical(r_max_val, cone_half, -wedge_half),   # 1 outlet (ch,-w) -> 0->1=+theta
            vtx_spherical(r_max_val, cone_half, wedge_half),    # 2 outlet (ch,w) -> 1->2=+phi
            vtx_spherical(r_max_val, axis_eps, wedge_half),    # 3 outlet (ae,w)
            vtx_spherical(r_min, axis_eps, wedge_half),    # 4 origin (ae,w) for axis
            vtx_spherical(r_min, cone_half, -wedge_half),   # 5
            vtx_spherical(r_min, cone_half, wedge_half),    # 6
            vtx_spherical(r_min, axis_eps, -wedge_half),   # 7 origin (ae,-w) for axis
        )
        mesh = [self._foam_header("blockMeshDict", "dictionary", location="system")]
        mesh.append("convertToMeters 1;\nvertices\n(")
        for v in vertices:
            mesh.append(f"    ({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})")
        mesh.append(");\nblocks\n(")
        mesh.append(f"    hex (0 3 2 1 4 7 6 5) (1 1 {n_r}) simpleGrading (1 1 1)")
        mesh.append(");\nedges\n(\n);\nboundary\n(")
        outlet_type = "wall" if self._right_boundary(inputs) == BOUNDARY_1D_REFLECT else "patch"
        mesh.append("    origin     { type symmetry; faces ((4 7 6 5)); }")
        mesh.append(f"    outlet     {{ type {outlet_type};    faces ((0 3 2 1)); }}")
        mesh.append("    axis       { type symmetry; faces ((0 4 7 3)); }")
        mesh.append("    outerCone  { type symmetry; faces ((1 2 6 5)); }")
        mesh.append("    wedgeFront { type wedge;    faces ((0 1 5 4)); }")
        mesh.append("    wedgeBack  { type wedge;    faces ((3 2 6 7)); }")
        mesh.append(");\nmergePatchPairs\n(\n);\n")
        self._write_text(os.path.join(sys_dir, "blockMeshDict"), "\n".join(mesh))

        # Visualization probes plus exact automatic Validation radii.
        probe_points = []
        p_r_start, p_r_end = r_min, min(target_radius, r_max_val - 1e-7)
        if p_r_end <= p_r_start:
            p_r_end = p_r_start + dx
        n_probe_cells = max(2, int((p_r_end - r_min) / dx)) if dx > 0 else 20
        actual_probes = max(2, min(int(inputs.n_probes), n_probe_cells))
        theta_mid = 0.5 * (axis_eps + cone_half)
        linear_radii = []
        for i in range(actual_probes):
            frac = i / (actual_probes - 1) if actual_probes > 1 else 0.5
            r_i = p_r_start + frac * (p_r_end - p_r_start)
            r_i = max(r_min + 1e-7, min(p_r_end - 1e-7, r_i))
            linear_radii.append(r_i)
        merged = merge_radii(
            linear_radii,
            validation_radii or (),
            r_lo=r_min + 1e-7,
            r_hi=p_r_end - 1e-7,
        )
        for r_i in merged:
            v = vtx_spherical(r_i, theta_mid, 0.0)
            probe_points.append(f"            ({v[0]:.12g} {v[1]:.12g} {v[2]:.12g})")

        fv_sol = self._foam_header("fvSolution", "dictionary", "system") + r"""
solvers { "(rho|rhoU|rhoE|alpha|.*)" { solver diagonal; } p { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0.05; } }
PIMPLE { nCorrectors 3; nNonOrthogonalCorrectors 0; }
"""
        self._write_text(os.path.join(sys_dir, "fvSolution"), fv_sol)

        if ig_state is not None:
            # No alpha.c4 and no lambda.c4 exist on the single-phase path, so their
            # scheme entries are dropped. `divSchemes { default none; }` alone is
            # accepted: the confirmation run completed with no unresolved-scheme error,
            # because the single-phase system takes all its fluxes from fluxScheme.
            fv_sch = self._foam_header("fvSchemes", "dictionary", "system") + r"""
fluxScheme      Tadmor;
ddtSchemes      { default Euler; timeIntegrator Euler; }
gradSchemes     { default cellMDLimited leastSquares 1.0; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; "reconstruct(rho)" vanLeer; "reconstruct(U)" vanLeer; "reconstruct(e)" vanLeer; "reconstruct(p)" vanLeer; "reconstruct(T)" vanLeer; "reconstruct(speedOfSound)" vanLeer; }
snGradSchemes   { default corrected; }
"""
        else:
            fv_sch = self._foam_header("fvSchemes", "dictionary", "system") + r"""
fluxScheme      Tadmor;
ddtSchemes      { default Euler; timeIntegrator Euler; }
gradSchemes     { default cellMDLimited leastSquares 1.0; }
divSchemes      { default none; div(alphaRhoPhi.c4,lambda.c4) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; "reconstruct(alpha.c4)" vanLeer; "reconstruct(rho)" vanLeer; "reconstruct(U)" vanLeer; "reconstruct(e)" vanLeer; "reconstruct(p)" vanLeer; "reconstruct(T)" vanLeer; "reconstruct(speedOfSound)" vanLeer; }
snGradSchemes   { default corrected; }
"""
        self._write_text(os.path.join(sys_dir, "fvSchemes"), fv_sch)

        if ig_state is not None:
            # The burst radius is snapped to a cell face, so sphereToCell selects
            # exactly n_source_cells whichever way OpenFOAM places cell centres.
            # p and rho are the only fields set; blastFoam derives e and T from them.
            sf = self._foam_header("setFieldsDict", "dictionary", "system") + f"""
defaultFieldValues ( volScalarFieldValue rho {ig_state.ambient.rho:.12g} volScalarFieldValue p {float(inputs.p_atm):.10g} );
regions ( sphereToCell {{ centre (0 0 0); radius {ig_state.shell.set_fields_radius_m:.12g}; fieldValues ( volScalarFieldValue rho {ig_state.rho_source:.12g} volScalarFieldValue p {ig_state.p_source:.12g} ); }} );
"""
        else:
            sf = self._foam_header("setFieldsDict", "dictionary", "system") + f"""
defaultFieldValues ( volScalarFieldValue alpha.c4 0 );
regions ( sphereToCell {{ centre (0 0 0); radius {float(charge_radius):.10g}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }} );
"""
        self._write_text(os.path.join(sys_dir, "setFieldsDict"), sf)

        end_time = self.resolved_end_time_s(inputs)
        # write_interval_s <= 0: one field dump at the configured OpenFOAM endTime.
        # Probes still stream the 1D graph; Terminate dumps via writeNow on arrival.
        user_write = float(inputs.write_interval_s)
        field_write_interval = user_write if user_write > 0.0 else end_time
        probe_steps = max(1, int(inputs.probe_write_interval_steps))
        foam_fields = tuple(getattr(inputs, "probe_fields", ("p",)) or ("p",))
        # Authoritative guard: the Output Options dialog does not know the source
        # model, so a stale alpha.c4 request would abort an IG run at the first
        # probe write. Filter here, where the model is certain.
        foam_fields = drop_unavailable_phase_fields(
            foam_fields, self.source_model(inputs)
        ) or ("p",)
        if "p" not in foam_fields:
            foam_fields = ("p",) + foam_fields
        fields_txt = " ".join(foam_fields)
        extras = extra_function_objects(
            p_atm=float(inputs.p_atm),
            impulse=bool(getattr(inputs, "enable_impulse", False)) or "impulse" in foam_fields,
            overpressure=False,
            dynamic_pressure=bool(getattr(inputs, "enable_dynamic_pressure", False)),
            peaks=False,
        )
        gauges_block = ""
        user_gauges = tuple(getattr(inputs, "gauge_locations", ()) or ())
        if user_gauges:
            gauge_pts = []
            for radius, _label in user_gauges:
                r_i = max(r_min + 1e-7, min(float(radius), r_max_val - 1e-7))
                v = vtx_spherical(r_i, theta_mid, 0.0)
                gauge_pts.append(f"            ({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})")
            gauges_block = f"""
    gauges1d
    {{
        type            probes;
        libs            ("libfieldFunctionObjects.so");
        fields          ({fields_txt});
        writeControl    timeStep;
        writeInterval   {probe_steps};
        probeLocations  ( {os.linesep.join(gauge_pts)} );
    }}
"""

        watchdog_path = os.path.join(case_dir, ".watchdog_target_radius")
        stop_radius = max(
            r_min + 1e-7,
            min(self.resolved_stop_radius_m(inputs), r_max_val - 1e-7),
        )
        watchdog_mid = vtx_spherical(stop_radius, theta_mid, 0.0)
        watchdog_point = (
            f"            ({watchdog_mid[0]:.6g} {watchdog_mid[1]:.6g} {watchdog_mid[2]:.6g})"
        )
        watchdog_block = f"""
    watchdog_probe
    {{
        type            probes;
        libs            ("libfieldFunctionObjects.so");
        fields          (p);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations  ( {watchdog_point} );
    }}
"""
        try:
            with open(watchdog_path, "w", encoding="utf-8") as f:
                f.write(f"{stop_radius:.6g}\n")
        except OSError:
            pass

        cd = self._foam_header("controlDict", "dictionary", "system") + f"""
application     blastFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.10g};
deltaT          {float(rec.dt0):.10g};
adjustTimeStep  yes;
maxCo           {float(rec.maxCo):.10g};
maxDeltaT       {float(rec.maxDeltaT):.10g};
writeControl    runTime;
writeInterval   {field_write_interval:.10g};
purgeWrite      1;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
runTimeModifiable true;
OptimisationSwitches
{{
    fileModificationSkew 0;
    fileModificationChecking timeStamp;
}}
functions
{{
{extras}    probes1d
    {{
        type            probes;
        libs            ("libfieldFunctionObjects.so");
        fields          ({fields_txt});
        writeControl    timeStep;
        writeInterval   {probe_steps};
        probeLocations  ( {os.linesep.join(probe_points)} );
    }}
{gauges_block}{watchdog_block}}}
"""
        self._write_text(os.path.join(sys_dir, "controlDict"), cd)
        self._write_text(os.path.join(sys_dir, "decomposeParDict"), self._foam_header("decomposeParDict", "dictionary") + "numberOfSubdomains 1; method scotch;")