import os
import math
from typing import Dict, Tuple

from base_generator import BaseGenerator
from models import CaseInputs1D, RecommendedParams1D

class Generator1D(BaseGenerator):
    """
    Handles 1D Wedge geometry generation.
    """
    def __init__(self, base_path: str, openfoam_bashrc: str = "/opt/openfoam9/etc/bashrc"):
        super().__init__(base_path)
        self.openfoam_bashrc = openfoam_bashrc

    def generate(self, case_name: str, inputs: CaseInputs1D, rec: RecommendedParams1D) -> str:
        # 1. Create Dirs
        case_dir = self.create_case_dirs(case_name)
        
        # 2. Derived Calculations
        charge_radius = self.calculate_charge_radius(inputs.mass_kg, inputs.rho_charge)
        
        # 3. Write Files
        self.write_initial_conditions(case_dir, inputs)
        
        self.write_constant_files(
            case_dir, 
            inputs.material_props, 
            inputs.energy_j_per_kg, 
            charge_radius, 
            rec.ignition_point, 
            rec.ignition_radius
        )

        self.write_system_files(case_dir, inputs, rec, charge_radius)
        
        # --- FIX: Write Scripts ---
        self.write_scripts(case_dir, self.openfoam_bashrc)
        
        # Create case.foam for ParaView compatibility
        import pathlib
        pathlib.Path(case_dir, "case.foam").touch()
        
        return case_dir

    def write_initial_conditions(self, case_dir: str, inputs: CaseInputs1D) -> None:
        # Write to 0.orig so Allrun's "cp -r 0.orig 0" restores initial conditions (same as 3D flow).
        zero_dir = os.path.join(case_dir, "0.orig")
        rho_air = 1.225
        patches = ["origin", "outlet", "axis", "outerCone", "wedgeFront", "wedgeBack"]

        # Outlet: pressureWaveTransmissive for p (like mappedBuilding3D) minimizes reflections; zeroGradient for U.
        # If blastFoam does not support pressureWaveTransmissive, fall back to advective for p.
        def scalar_bcs(name, val):
            lines = ["boundaryField", "{"]
            for pch in patches:
                if pch.startswith("wedge"): lines.append(f"    {pch} {{ type wedge; }}")
                elif pch in ("axis", "outerCone", "origin"):
                    lines.append(f"    {pch} {{ type symmetry; }}")
                else:
                    if name == "p" and pch == "outlet":
                        lines.append(f"    {pch} {{ type pressureWaveTransmissive; value uniform {inputs.p_atm}; }}")
                    else:
                        lines.append(f"    {pch} {{ type zeroGradient; }}")
            lines.append("}\n")
            return "\n".join(lines)

        def vector_bcs():
            lines = ["boundaryField", "{"]
            for pch in patches:
                if pch.startswith("wedge"): lines.append(f"    {pch} {{ type wedge; }}")
                elif pch in ("axis", "outerCone", "origin"):
                    lines.append(f"    {pch} {{ type symmetry; }}")
                else:
                    # Outlet: zeroGradient like mappedBuilding3D (allows outflow without reflection)
                    lines.append(f"    {pch} {{ type zeroGradient; }}")
            lines.append("}\n")
            return "\n".join(lines)

        self._write_text(os.path.join(zero_dir, "p"), self._foam_header("p", "volScalarField", "0") + f"dimensions [1 -1 -2 0 0 0 0]; internalField uniform {inputs.p_atm};\n" + scalar_bcs("p", inputs.p_atm))
        self._write_text(os.path.join(zero_dir, "T"), self._foam_header("T", "volScalarField", "0") + f"dimensions [0 0 0 1 0 0 0]; internalField uniform {inputs.t_atm};\n" + scalar_bcs("T", inputs.t_atm))
        self._write_text(os.path.join(zero_dir, "rho.c4"), self._foam_header("rho.c4", "volScalarField", "0") + f"dimensions [1 -3 0 0 0 0 0]; internalField uniform {inputs.rho_charge};\n" + scalar_bcs("rho", inputs.rho_charge))
        self._write_text(os.path.join(zero_dir, "rho.air"), self._foam_header("rho.air", "volScalarField", "0") + f"dimensions [1 -3 0 0 0 0 0]; internalField uniform {rho_air};\n" + scalar_bcs("rho", rho_air))
        self._write_text(os.path.join(zero_dir, "alpha.c4"), self._foam_header("alpha.c4", "volScalarField", "0") + f"dimensions [0 0 0 0 0 0 0]; internalField uniform 0;\n" + scalar_bcs("alpha", 0))
        self._write_text(os.path.join(zero_dir, "U"), self._foam_header("U", "volVectorField", "0") + f"dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);\n" + vector_bcs())

    def write_constant_files(self, case_dir: str, mat_props: Dict, energy: float, charge_radius: float, ignition_point: Tuple, ignition_radius: float) -> None:
        const_dir = os.path.join(case_dir, "constant")
        self._write_text(os.path.join(const_dir, "turbulenceProperties"), 
                         self._foam_header("turbulenceProperties", "dictionary", "constant") + "simulationType laminar;\n")

        ign_str = f"({ignition_point[0]:.10g} {ignition_point[1]:.10g} {ignition_point[2]:.10g})"
        
        pp_content = self._foam_header("phaseProperties", "dictionary", location="constant") + f"""
phases (c4 air);
c4
{{
    type detonating;
    reactants
    {{
        thermoType {{ transport const; thermo eConst; equationOfState BirchMurnaghan3; }}
        equationOfState {{ rho0 {mat_props['rho']}; Gamma 0.25; pRef 101298; K0 8.04e9; K0Prime 7.97; }}
        specie {{ molWeight 55.0; }}
        transport {{ mu 0; Pr 1; }}
        thermodynamics {{ Cv 1400; Hf 0.0; }}
    }}
    products
    {{
        thermoType {{ transport const; thermo ePolynomial; equationOfState JWL; }}
        equationOfState 
        {{ 
            rho0 {mat_props['rho']}; 
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
        E0 {energy}; 
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

    def write_system_files(self, case_dir: str, inputs: CaseInputs1D, rec: RecommendedParams1D, charge_radius: float) -> None:
        sys_dir = os.path.join(case_dir, "system")
        # User's target radius (for mapping); domain is buffered so shock can be detected before boundary
        target_radius = float(inputs.radius)
        physical_radius = target_radius * 1.1  # 10% buffer to avoid boundary artifacts at target
        r_max_val = physical_radius
        dx = max(float(inputs.cell_size), 1e-9)
        r_min = rec.r_min
        if r_max_val <= r_min:
            r_max_val = r_min + 10.0 * dx
        n_r = max(20, int((r_max_val - r_min) / dx))

        wedge_half = math.radians(inputs.wedge_angle_deg) / 2.0
        cone_half = math.radians(inputs.cone_half_angle_deg)
        # Small theta so axis face has finite area; axis_epsilon is a small number (e.g. 1e-3 rad)
        axis_eps = max(1e-9, min(float(inputs.axis_epsilon), cone_half * 0.5))

        # Spherical wedge: vertices on spheres r=const so rotateFields produces a sphere (not a cylinder).
        # x = r*cos(theta), y = r*sin(theta)*cos(phi), z = r*sin(theta)*sin(phi); axis = x, theta from axis.
        def vtx_spherical(r, theta, phi):
            st, ct = math.sin(theta), math.cos(theta)
            sp, cp = math.sin(phi), math.cos(phi)
            return (r * ct, r * st * cp, r * st * sp)

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
        mesh.append("    origin     { type symmetry; faces ((4 7 6 5)); }")
        mesh.append("    outlet     { type patch;    faces ((0 3 2 1)); }")
        mesh.append("    axis       { type symmetry; faces ((0 4 7 3)); }")
        mesh.append("    outerCone  { type symmetry; faces ((1 2 6 5)); }")
        mesh.append("    wedgeFront { type wedge;    faces ((0 1 5 4)); }")
        mesh.append("    wedgeBack  { type wedge;    faces ((3 2 6 7)); }")
        mesh.append(");\nmergePatchPairs\n(\n);\n")
        self._write_text(os.path.join(sys_dir, "blockMeshDict"), "\n".join(mesh))

        # Visualization probes: spherical radius r_min to target_radius; place at mid-wedge (theta=cone/2, phi=0)
        probe_points = []
        p_r_start, p_r_end = r_min, min(target_radius, r_max_val - 1e-7)
        if p_r_end <= p_r_start:
            p_r_end = p_r_start + dx
        n_probe_cells = max(2, int((p_r_end - r_min) / dx)) if dx > 0 else 20
        actual_probes = max(2, min(int(inputs.n_probes), n_probe_cells))
        theta_mid = 0.5 * (axis_eps + cone_half)
        for i in range(actual_probes):
            frac = i / (actual_probes - 1) if actual_probes > 1 else 0.5
            r_i = p_r_start + frac * (p_r_end - p_r_start)
            r_i = max(r_min + 1e-7, min(p_r_end - 1e-7, r_i))
            v = vtx_spherical(r_i, theta_mid, 0.0)
            probe_points.append(f"            ({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})")

        fv_sol = self._foam_header("fvSolution", "dictionary", "system") + r"""
solvers { "(rho|rhoU|rhoE|alpha|.*)" { solver diagonal; } p { solver PCG; preconditioner DIC; tolerance 1e-5; relTol 0.05; } }
PIMPLE { nCorrectors 3; nNonOrthogonalCorrectors 0; }
"""
        self._write_text(os.path.join(sys_dir, "fvSolution"), fv_sol)

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

        sf = self._foam_header("setFieldsDict", "dictionary", "system") + f"""
defaultFieldValues ( volScalarFieldValue alpha.c4 0 );
regions ( sphereToCell {{ centre (0 0 0); radius {float(charge_radius):.10g}; fieldValues ( volScalarFieldValue alpha.c4 1 ); }} );
"""
        self._write_text(os.path.join(sys_dir, "setFieldsDict"), sf)

        # Watchdog probe at spherical radius target_radius, mid-wedge (theta=cone/2, phi=0)
        watchdog_mid = vtx_spherical(target_radius, theta_mid, 0.0)
        watchdog_point = f"            ({watchdog_mid[0]:.6g} {watchdog_mid[1]:.6g} {watchdog_mid[2]:.6g})"
        # Safety end time so run does not end before shock can reach target (slow shock ~300 m/s, *2 margin)
        safe_end_time = (target_radius / 300.0) * 2.0
        end_time = max(float(inputs.end_time_s), safe_end_time)

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
writeInterval   {float(inputs.write_interval_s):.10g};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
runTimeModifiable true;
functions
{{
    probes1d
    {{
        type            probes;
        libs            ("libfieldFunctionObjects.so");
        fields          (p);
        writeControl    timeStep;
        writeInterval   {inputs.probe_write_interval_steps};
        probeLocations  ( {os.linesep.join(probe_points)} );
    }}
    watchdog_probe
    {{
        type            probes;
        libs            ("libfieldFunctionObjects.so");
        fields          (p);
        writeControl    timeStep;
        writeInterval   1;
        probeLocations  ( {watchdog_point} );
    }}
}}
"""
        self._write_text(os.path.join(sys_dir, "controlDict"), cd)
        self._write_text(os.path.join(sys_dir, "decomposeParDict"), self._foam_header("decomposeParDict", "dictionary") + "numberOfSubdomains 1; method scotch;")
        # Store target_radius for solver_runner watchdog log message
        try:
            with open(os.path.join(case_dir, ".watchdog_target_radius"), "w", encoding="utf-8") as f:
                f.write(f"{target_radius:.6g}\n")
        except OSError:
            pass