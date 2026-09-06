"""Physics tests for the Ideal-Gas Isothermal Burst source-state derivation.

These assert the derivation is exactly the documented closed form -- no fitted
coefficients, no multipliers, no scaled-distance dependence -- and that mass and
energy are conserved identically on any mesh.
"""

from __future__ import annotations

import json
import math
import unittest

import ig_source_state as igs
from models import (
    SOURCE_MODEL_IG,
    SOURCE_MODEL_JWL,
    SourceModelError,
    normalize_source_model,
)

# GGUI TNT catalog entry, W = 1 kg, at the verified 1 mm 1D resolution.
TNT = dict(mass_kg=1.0, rho_charge=1630.0, energy_j_per_kg=4.29e6)
ATM = dict(p_atm=101325.0, t_atm=288.0)
MESH_1MM = dict(r_min_m=0.01, cell_size_m=0.001)


def baseline_state() -> igs.IgBurstState:
    return igs.derive_ig_state(**TNT, **ATM, **MESH_1MM)


class GasConstantTests(unittest.TestCase):
    def test_specific_gas_constant_matches_econst_idealgas_pair(self):
        # blastFoam builds R implicitly from eConst Cv and the idealGas gamma.
        self.assertAlmostEqual(igs.specific_gas_constant(), 287.2, places=10)

    def test_ambient_density_is_derived_not_hard_coded(self):
        amb = igs.ambient_state(101325.0, 288.0)
        self.assertAlmostEqual(amb.rho, 101325.0 / (287.2 * 288.0), places=14)
        # The legacy hard-coded 1.225 only happens to be right at this one ambient.
        self.assertAlmostEqual(amb.rho, 1.225, places=3)

    def test_ambient_state_round_trips_through_the_eos(self):
        # p = (gamma-1)*rho*e must return exactly p_atm, or the run starts with a
        # spurious wave in the far field.
        for p_atm, t_atm in ((101325.0, 288.0), (90000.0, 250.0), (110000.0, 320.0)):
            with self.subTest(p_atm=p_atm, t_atm=t_atm):
                amb = igs.ambient_state(p_atm, t_atm)
                self.assertAlmostEqual(
                    (amb.gamma - 1.0) * amb.rho * amb.e / p_atm, 1.0, places=13
                )

    def test_ambient_rejects_non_physical_inputs(self):
        for kwargs in (
            dict(p_atm=0.0, t_atm=288.0),
            dict(p_atm=-1.0, t_atm=288.0),
            dict(p_atm=101325.0, t_atm=0.0),
            dict(p_atm=float("nan"), t_atm=288.0),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(igs.IgSourceStateError):
                    igs.ambient_state(**kwargs)

    def test_gamma_must_exceed_one(self):
        with self.assertRaises(igs.IgSourceStateError):
            igs.ambient_state(101325.0, 288.0, gamma=1.0)


class ChargeGeometryTests(unittest.TestCase):
    def test_charge_radius_matches_the_existing_generator_value(self):
        # generator_1d writes 0.05271180996 for this charge; the IG path must agree,
        # otherwise BF-IG and BF-JWL would not be initialising the same charge.
        charge = igs.charge_geometry(**TNT)
        self.assertAlmostEqual(charge.radius_m, 0.05271180996, places=10)
        self.assertAlmostEqual(charge.volume_m3, 1.0 / 1630.0, places=15)
        self.assertAlmostEqual(charge.source_energy_j, 4.29e6, places=6)

    def test_charge_rejects_non_physical_inputs(self):
        for bad in (dict(mass_kg=0.0), dict(rho_charge=-1.0), dict(energy_j_per_kg=0.0)):
            kwargs = dict(TNT)
            kwargs.update(bad)
            with self.subTest(**bad):
                with self.assertRaises(igs.IgSourceStateError):
                    igs.charge_geometry(**kwargs)

    def test_charge_radius_scales_as_cube_root_of_mass(self):
        r1 = igs.charge_geometry(1.0, 1630.0, 4.29e6).radius_m
        for w in (0.1, 8.0, 27.0):
            with self.subTest(w=w):
                rw = igs.charge_geometry(w, 1630.0, 4.29e6).radius_m
                self.assertAlmostEqual(rw / (r1 * w ** (1.0 / 3.0)), 1.0, places=12)


class SourceShellTests(unittest.TestCase):
    def test_shell_matches_the_measured_openfoam_cell_count(self):
        # A live setFields run on the 1090-cell retest mesh reported
        # "Adding cells with centre within sphere ... radius = 0.0527118" and selected
        # exactly 43 cells spanning r = 0.010 .. 0.053.
        shell = igs.source_shell(0.01, 0.001, 0.05271180996)
        self.assertEqual(shell.n_cells, 43)
        self.assertAlmostEqual(shell.r_outer_m, 0.053, places=12)
        self.assertAlmostEqual(
            shell.volume_full_sphere_m3, 0.000619425729113197, places=15
        )

    def test_set_fields_radius_is_a_cell_face_radius(self):
        # A face radius lies strictly between the centres of the cells either side of
        # it, so the selected count is exact for any centroid convention -- which
        # matters because the wedge is twisted and its cell centres are not ideal
        # frustum centroids.
        shell = igs.source_shell(0.01, 0.001, 0.05271180996)
        n_faces = (shell.set_fields_radius_m - shell.r_min_m) / shell.cell_size_m
        self.assertAlmostEqual(n_faces, round(n_faces), places=9)

        inner_centre = igs.frustum_centroid_radius(
            shell.r_outer_m - shell.cell_size_m, shell.r_outer_m
        )
        outer_centre = igs.frustum_centroid_radius(
            shell.r_outer_m, shell.r_outer_m + shell.cell_size_m
        )
        self.assertLess(inner_centre, shell.set_fields_radius_m)
        self.assertLess(shell.set_fields_radius_m, outer_centre)

    def test_shell_volume_stays_within_the_cell_quantization_bound(self):
        # r_outer can only land on a cell face, so it is at worst dx/2 from the
        # volume-matched target. Since dV/V = 3 dr/r, the volume mismatch cannot
        # exceed 3*(dx/2)/r_outer at any resolution. Anything larger would mean the
        # cell count is being chosen wrongly rather than merely quantized.
        charge = igs.charge_geometry(**TNT)
        errors = {}
        for dx in (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4):
            with self.subTest(dx=dx):
                shell = igs.source_shell(0.01, dx, charge.radius_m)
                err = abs(shell.volume_ratio_to_charge - 1.0)
                bound = 3.0 * (dx / 2.0) / shell.r_outer_m
                self.assertLessEqual(err, bound * 1.05)
                errors[dx] = err
        # Refinement still buys real accuracy end to end.
        self.assertLess(errors[1e-4], 0.1 * errors[4e-3])

    def test_shell_reports_the_core_excluded_by_the_mesh(self):
        # profiles.compute_recommended_1d already notes "Missing mass is (r_min/R)^3".
        shell = igs.source_shell(0.01, 0.001, 0.05271180996)
        self.assertAlmostEqual(
            shell.core_volume_fraction, (0.01 / 0.05271180996) ** 3, places=12
        )
        self.assertAlmostEqual(shell.core_volume_fraction, 0.006823, places=5)

    def test_shell_rejects_an_inner_radius_outside_the_charge(self):
        with self.assertRaises(igs.IgSourceStateError):
            igs.source_shell(0.06, 0.001, 0.0527118)

    def test_shell_keeps_at_least_one_cell_on_a_very_coarse_mesh(self):
        self.assertEqual(igs.source_shell(0.01, 0.5, 0.0527118).n_cells, 1)

    def test_frustum_centroid_lies_outside_the_arithmetic_midpoint(self):
        a, b = 0.052, 0.053
        centroid = igs.frustum_centroid_radius(a, b)
        self.assertGreater(centroid, 0.5 * (a + b))
        self.assertAlmostEqual(centroid, 0.05250317, places=7)
        self.assertAlmostEqual(igs.frustum_centroid_radius(0.0, 1.0), 0.75, places=15)

    def test_frustum_centroid_rejects_a_degenerate_shell(self):
        with self.assertRaises(igs.IgSourceStateError):
            igs.frustum_centroid_radius(0.05, 0.05)


class BurstStateTests(unittest.TestCase):
    def test_burst_state_is_the_documented_closed_form(self):
        st = baseline_state()
        self.assertAlmostEqual(
            st.rho_source, st.charge.mass_kg / st.shell.volume_full_sphere_m3, places=9
        )
        self.assertAlmostEqual(
            st.e_source, st.ambient.cv * st.ambient.t_atm + st.charge.energy_j_per_kg
        )
        self.assertAlmostEqual(
            st.t_source,
            st.ambient.t_atm + st.charge.energy_j_per_kg / st.ambient.cv,
            places=9,
        )
        self.assertAlmostEqual(
            st.p_source / ((st.ambient.gamma - 1.0) * st.rho_source * st.e_source),
            1.0,
            places=14,
        )

    def test_burst_state_uses_added_energy_pressure_and_temperature(self):
        st = baseline_state()
        self.assertAlmostEqual(st.rho_source, 1614.398552, places=6)
        self.assertAlmostEqual(st.e_source, 4.29e6 + 718.0 * 288.0, places=6)
        self.assertAlmostEqual(st.t_source, 288.0 + 4.29e6 / 718.0, places=6)
        self.assertAlmostEqual(st.p_source / 2.903840631e9, 1.0, places=6)

    def test_mass_and_detonation_energy_are_conserved_exactly_on_any_mesh(self):
        charge = igs.charge_geometry(**TNT)
        for dx in (1e-2, 4e-3, 1e-3, 5e-4, 1e-4):
            for r_min in (1e-4, 1e-3, 5e-3, 0.01):
                if r_min >= charge.radius_m:
                    continue
                with self.subTest(dx=dx, r_min=r_min):
                    st = igs.derive_ig_state(**TNT, **ATM, r_min_m=r_min, cell_size_m=dx)
                    self.assertAlmostEqual(st.mass_error_rel, 0.0, places=12)
                    self.assertAlmostEqual(st.detonation_energy_error_rel, 0.0, places=12)
                    self.assertAlmostEqual(
                        st.delta_e_j / (TNT["mass_kg"] * TNT["energy_j_per_kg"]),
                        1.0,
                        places=12,
                    )
                    self.assertGreater(st.e_final_internal_j, st.delta_e_j)

    def test_source_pressure_uses_radii_only(self):
        st = baseline_state()
        rho = 1.0 / (4.0 / 3.0 * math.pi * (0.053 ** 3 - 0.01 ** 3))
        expected = 0.4 * rho * (4.29e6 + 718.0 * 288.0)
        self.assertAlmostEqual(st.p_source / expected, 1.0, places=12)

    def test_temperature_is_atm_plus_added_energy_over_cv(self):
        a = igs.derive_ig_state(**TNT, **ATM, **MESH_1MM)
        b = igs.derive_ig_state(
            mass_kg=17.0,
            rho_charge=900.0,
            energy_j_per_kg=4.29e6,
            p_atm=90000.0,
            t_atm=250.0,
            r_min_m=0.002,
            cell_size_m=0.0005,
        )
        self.assertAlmostEqual(a.t_source, 288.0 + 4.29e6 / 718.0, places=9)
        self.assertAlmostEqual(b.t_source, 250.0 + 4.29e6 / 718.0, places=9)
        self.assertNotAlmostEqual(a.t_source, b.t_source, places=6)

    def test_detonation_energy_is_added_not_the_total_internal_energy(self):
        st = baseline_state()
        self.assertEqual(
            st.production_energy_convention, igs.ENERGY_CONVENTION_DETONATION_ADDED
        )
        self.assertAlmostEqual(st.delta_e_j / st.charge.source_energy_j, 1.0, places=12)
        self.assertAlmostEqual(
            st.e_final_internal_j,
            st.e_initial_sensible_j + st.e_detonation_added_j,
            places=6,
        )
        self.assertGreater(st.e_final_internal_j, st.charge.source_energy_j)

    def test_ig_state_is_tagged_with_the_source_model_and_derivation(self):
        st = baseline_state()
        self.assertEqual(st.source_model, SOURCE_MODEL_IG)
        self.assertEqual(st.derivation_id, igs.DERIVATION_ID)
        self.assertGreaterEqual(st.source_model_schema_version, 1)


class HopkinsonScalingTests(unittest.TestCase):
    def test_scaling_is_exact_when_the_mesh_scales_with_the_charge(self):
        # Series A of the mass-scaling study: scale dx and r_min by W^(1/3) and the
        # discrete problem is identical up to similarity, so the intensive burst state
        # must be reproduced. This is what would break if anything were hard-coded in
        # absolute units.
        ref = igs.derive_ig_state(
            mass_kg=1.0,
            rho_charge=1630.0,
            energy_j_per_kg=4.29e6,
            **ATM,
            r_min_m=0.01,
            cell_size_m=0.001,
        )
        for w in (0.001, 0.1, 10.0, 1000.0):
            with self.subTest(w=w):
                s = w ** (1.0 / 3.0)
                st = igs.derive_ig_state(
                    mass_kg=w,
                    rho_charge=1630.0,
                    energy_j_per_kg=4.29e6,
                    **ATM,
                    r_min_m=0.01 * s,
                    cell_size_m=0.001 * s,
                )
                self.assertEqual(st.shell.n_cells, ref.shell.n_cells)
                self.assertAlmostEqual(st.rho_source / ref.rho_source, 1.0, places=12)
                self.assertAlmostEqual(st.p_source / ref.p_source, 1.0, places=12)
                self.assertAlmostEqual(st.t_source / ref.t_source, 1.0, places=12)
                self.assertAlmostEqual(
                    st.shell.volume_ratio_to_charge
                    / ref.shell.volume_ratio_to_charge,
                    1.0,
                    places=12,
                )


class WedgeMeshDiagnosticTests(unittest.TestCase):
    def test_wedge_solid_angle_matches_the_measured_mesh(self):
        # axis_eps 0.10, cone_half 12 deg, wedge_half 7.5 deg, as generator_1d emits.
        omega = igs.wedge_solid_angle(0.10, math.radians(12.0), math.radians(15.0) / 2.0)
        self.assertAlmostEqual(omega, 0.004413038278, places=10)
        # The measured mesh total volume implies 0.98273 of this, from the twisted hex.
        measured_c = 0.0014456024557576665
        self.assertAlmostEqual(measured_c / (omega / 3.0), 0.98273, places=4)

    def test_cone_volume_ratio_exposes_the_twisted_mesh(self):
        # Measured OpenFOAM volumes for the two innermost cells of the 1090-cell
        # retest mesh. Near the axis the wedge is far from a true cone, which is
        # precisely why the source derivation never sums mesh volumes.
        omega = igs.wedge_solid_angle(0.10, math.radians(12.0), math.radians(15.0) / 2.0)
        ratios = igs.wedge_cone_volume_ratio(
            (1.837350e-10, 2.515190e-10), 0.01, 0.001, omega
        )
        self.assertAlmostEqual(ratios[0], 0.37735, places=4)
        self.assertAlmostEqual(ratios[1], 0.43069, places=4)
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], 1.0)


class AuditPayloadTests(unittest.TestCase):
    def test_audit_dict_reports_every_required_quantity(self):
        st = baseline_state()
        audit = igs.audit_dict(st, case_path="/tmp/case", material_name="TNT")

        self.assertEqual(audit["source_model"], SOURCE_MODEL_IG)
        self.assertEqual(audit["material_name"], "TNT")

        ui = audit["user_inputs"]
        self.assertEqual(ui["W_kg"], 1.0)
        self.assertEqual(ui["rho_charge_kg_m3"], 1630.0)
        self.assertEqual(ui["E_charge_J_kg"], 4.29e6)

        self.assertAlmostEqual(
            audit["charge_ideal"]["intended_detonation_energy_J"], 4.29e6, places=6
        )

        region = audit["source_region"]
        self.assertEqual(region["n_source_cells"], 43)
        self.assertAlmostEqual(region["set_fields_radius_m"], 0.053, places=12)
        self.assertAlmostEqual(region["cells_across_charge_radius"], 52.7118, places=3)

        init = audit["initialized_state"]
        self.assertAlmostEqual(init["p_source_Pa"] / 2.903840631e9, 1.0, places=6)
        self.assertAlmostEqual(init["T_source_K"], 6262.93, places=1)
        self.assertAlmostEqual(init["rho_source_over_rho_charge"], 0.99044, places=4)
        self.assertAlmostEqual(init["e_initial_J_kg"], 718.0 * 288.0, places=6)

        cons = audit["conservation"]
        self.assertAlmostEqual(cons["mass_error_rel"], 0.0, places=12)
        self.assertAlmostEqual(cons["DeltaE_error_rel"], 0.0, places=12)
        self.assertAlmostEqual(cons["E_detonation_added_J"], 4.29e6, places=6)
        self.assertGreater(cons["E_final_internal_J"], cons["E_detonation_added_J"])
        self.assertIn("not equal to W*E_charge", cons["note"])
        self.assertEqual(
            audit["energy_convention"]["production"],
            igs.ENERGY_CONVENTION_DETONATION_ADDED,
        )

    def test_audit_dict_records_mesh_verification_when_measured(self):
        st = baseline_state()
        audit = igs.audit_dict(
            st, measured_source_cells=43, measured_wedge_source_volume_m3=1.7710891e-07
        )
        mesh = audit["mesh_verification"]
        self.assertIs(mesh["cell_count_matches"], True)
        self.assertEqual(mesh["predicted_source_cells"], 43)

        mismatched = igs.audit_dict(st, measured_source_cells=44)
        self.assertIs(mismatched["mesh_verification"]["cell_count_matches"], False)

    def test_audit_dict_is_json_serializable(self):
        payload = json.dumps(igs.audit_dict(baseline_state()))
        self.assertIn("IG_ISOTHERMAL_BURST", payload)


class SourceModelDiscriminatorTests(unittest.TestCase):
    def test_normalize_defaults_legacy_cases_to_jwl(self):
        for legacy in (None, "", "   "):
            with self.subTest(value=legacy):
                self.assertEqual(normalize_source_model(legacy), SOURCE_MODEL_JWL)

    def test_normalize_round_trips_known_values(self):
        self.assertEqual(normalize_source_model(SOURCE_MODEL_JWL), SOURCE_MODEL_JWL)
        self.assertEqual(normalize_source_model(SOURCE_MODEL_IG), SOURCE_MODEL_IG)
        self.assertEqual(
            normalize_source_model("  IG_ISOTHERMAL_BURST  "), SOURCE_MODEL_IG
        )

    def test_normalize_rejects_unknown_values(self):
        # Silently falling back would generate the wrong case.
        for bad in ("jwl", "ig", "IdealGas", "IG_ISOTHERMAL", "JWL_DETONATION_V2"):
            with self.subTest(value=bad):
                with self.assertRaises(SourceModelError):
                    normalize_source_model(bad)


if __name__ == "__main__":
    unittest.main()

