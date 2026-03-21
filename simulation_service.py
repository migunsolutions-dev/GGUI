from __future__ import annotations
import os
from datetime import datetime

from models import CaseInputs, CaseInputs1D, CaseInputs3D, GenerationResult1D, RecommendedParams1D
from generator_1d import Generator1D
from generator_3d import Generator3D
from profiles import get_profile, compute_recommended_1d

class SimulationService:
    def __init__(self, *, base_projects_path: str, openfoam_bashrc: str):
        self.base_projects_path = base_projects_path
        self.openfoam_bashrc = openfoam_bashrc
        # Keep generator instances for accessing post-generation metadata
        self.generator_1d = None
        self.generator_3d = None

    def make_case_name(self, prefix: str = "Case") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}"

    def generate_case(self, case_name: str, inputs: CaseInputs) -> str:
        
        # --- מסלול 1D ---
        if isinstance(inputs, CaseInputs1D):
            # יצירת גנרטור עם נתיב ה-Bashrc
            self.generator_1d = Generator1D(self.base_projects_path, self.openfoam_bashrc)
            gen = self.generator_1d 
            
            # חישוב המלצות
            charge_radius = gen.calculate_charge_radius(inputs.mass_kg, inputs.rho_charge)
            profile = get_profile("Balanced")
            rec = compute_recommended_1d(
                radius=inputs.radius,
                cell_size=inputs.cell_size,
                charge_radius=charge_radius,
                profile=profile,
                max_cfl_from_ui=inputs.max_cfl,
            )
            
            # יצירה
            case_dir = gen.generate(case_name, inputs, rec)
            return case_dir

        # --- מסלול 3D ---
        elif isinstance(inputs, CaseInputs3D):
            # יצירת גנרטור עם נתיב ה-Bashrc
            self.generator_3d = Generator3D(self.base_projects_path, self.openfoam_bashrc)
            gen = self.generator_3d
            case_dir = gen.generate(case_name, inputs)
            return case_dir

        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")