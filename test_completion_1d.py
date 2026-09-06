"""1D run modes: Terminate (verified arrival) vs Reflect (user stop)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest import mock

from completion_1d import (
    ARRIVAL_CRITERION,
    ARRIVAL_OVERPRESSURE_PA,
    COMPLETION_FILENAME,
    RUN_MODE_REFLECT,
    RUN_MODE_TERMINATE,
    STOP_REASON_END_TIME_REACHED,
    STOP_REASON_END_TIME_WITHOUT_ARRIVAL,
    STOP_REASON_NO_ARRIVAL,
    STOP_REASON_USER_STOPPED,
    STOP_REASON_WAVE_RADIUS_REACHED,
    arrival_time_from_history,
    detect_arrival_in_case,
    finalize_completion_record,
    initial_completion_record,
    normalize_run_mode,
    overpressure_arrived,
    read_completion_record,
    reset_completion_for_new_run,
    write_completion_record,
)
from result_storage import solver_run_succeeded


def _write_control(case: str, end_time: float = 1.0e9) -> None:
    sys_dir = os.path.join(case, "system")
    os.makedirs(sys_dir, exist_ok=True)
    with open(os.path.join(sys_dir, "controlDict"), "w", encoding="utf-8") as handle:
        handle.write(
            f"endTime         {end_time};\n"
            "functions\n{\n    probes1d { writeInterval 25; }\n}\n"
        )


def _write_log(case: str, time_s: float, *, fatal: bool = False) -> None:
    body = f"Time = {time_s}\nEnd\n"
    if fatal:
        body = (
            "--> FOAM FATAL ERROR: \n"
            "No cells will be activated using the detonation point (0.0105 0 0)\n"
        )
    with open(os.path.join(case, "log.blastFoam"), "w", encoding="utf-8") as handle:
        handle.write(body)


def _write_probe(
    case: str,
    *,
    fo_name: str,
    location: str,
    samples: list[tuple[float, float]],
) -> None:
    folder = os.path.join(case, "postProcessing", fo_name, "0")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "p")
    lines = [f"# Probe 0 ({location})\n"]
    for sample_t, pressure in samples:
        lines.append(f"{sample_t} {pressure}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)


def _arrived_record(case: str, *, mode: str, radius: float = 0.8) -> None:
    write_completion_record(
        case,
        initial_completion_record(
            mode=mode,
            requested_stop_radius_m=radius,
            p_atm=101325.0,
            right_boundary="Reflect" if mode == RUN_MODE_REFLECT else "Terminate",
        ),
    )
    _write_probe(
        case,
        fo_name="watchdog_probe",
        location=f"{radius} 0 0",
        samples=((0.0, 101325.0), (0.0012, 120000.0)),
    )


class ArrivalCriterionTests(unittest.TestCase):
    def test_first_crossing_above_8kpa_overpressure(self) -> None:
        self.assertFalse(overpressure_arrived(101325.0, p_atm=101325.0))
        self.assertFalse(overpressure_arrived(105000.0, p_atm=101325.0))
        self.assertTrue(overpressure_arrived(109325.0, p_atm=101325.0))
        arrived = arrival_time_from_history(
            (0.0, 0.001, 0.002),
            (101325.0, 105000.0, 120000.0),
            p_atm=101325.0,
        )
        self.assertAlmostEqual(arrived, 0.002)

    def test_ambient_noise_does_not_count_as_arrival(self) -> None:
        self.assertIsNone(
            arrival_time_from_history(
                (0.0, 0.001, 0.002),
                (101325.0, 101400.0, 105000.0),
                p_atm=101325.0,
            )
        )


class RunModeNormalizeTests(unittest.TestCase):
    def test_right_boundary_wins_over_legacy_stop_mode(self) -> None:
        self.assertEqual(
            normalize_run_mode("end_time", "Terminate"), RUN_MODE_TERMINATE
        )
        self.assertEqual(
            normalize_run_mode("wave_radius", "Reflect"), RUN_MODE_REFLECT
        )
        self.assertEqual(normalize_run_mode("end_time", None), RUN_MODE_REFLECT)
        self.assertEqual(normalize_run_mode("transmit", None), RUN_MODE_TERMINATE)


class CompletionRecordTests(unittest.TestCase):
    def test_reset_clears_stale_arrival_and_final_time(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            stale = initial_completion_record(
                mode=RUN_MODE_TERMINATE,
                requested_stop_radius_m=1.0,
                p_atm=101325.0,
                end_time_s=0.025,
            )
            stale.wave_radius_reached = True
            stale.detected_arrival_time_s = 0.004
            stale.final_solver_time_s = 0.004
            stale.stop_reason = STOP_REASON_WAVE_RADIUS_REACHED
            stale.probe_function_object = "watchdog_probe"
            stale.probe_location = "1 0 0"
            write_completion_record(case, stale)
            reset = reset_completion_for_new_run(case)
            self.assertEqual(reset.mode, RUN_MODE_TERMINATE)
            self.assertEqual(reset.stop_mode, RUN_MODE_TERMINATE)
            self.assertAlmostEqual(reset.requested_stop_radius_m, 1.0)
            self.assertAlmostEqual(reset.end_time_s, 0.025)
            self.assertFalse(reset.wave_radius_reached)
            self.assertIsNone(reset.detected_arrival_time_s)
            self.assertIsNone(reset.final_solver_time_s)
            self.assertEqual(reset.stop_reason, "")
            loaded = read_completion_record(case)
            self.assertIsNotNone(loaded)
            self.assertFalse(loaded.wave_radius_reached)
            self.assertIsNone(loaded.final_solver_time_s)
            with open(os.path.join(case, COMPLETION_FILENAME), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertFalse(payload.get("arrival_event"))
            self.assertEqual(payload.get("requested_radius_m"), 1.0)
            self.assertEqual(payload.get("end_time_s"), 0.025)
            self.assertEqual(payload.get("endTime"), 0.025)

    def test_detect_arrival_uses_watchdog_probe_near_requested_radius(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_probe(
                case,
                fo_name="watchdog_probe",
                location="0.8 0 0",
                samples=((0.0, 101325.0), (0.0012, 120000.0)),
            )
            record = initial_completion_record(
                mode=RUN_MODE_TERMINATE,
                requested_stop_radius_m=0.8,
                p_atm=101325.0,
            )
            detected = detect_arrival_in_case(case, record)
            self.assertTrue(detected.wave_radius_reached)
            self.assertAlmostEqual(detected.detected_arrival_time_s, 0.0012)
            self.assertEqual(detected.probe_function_object, "watchdog_probe")
            self.assertEqual(detected.probe_index, 0)
            self.assertAlmostEqual(detected.probe_radius_m, 0.8)
            self.assertEqual(detected.threshold_overpressure_pa, ARRIVAL_OVERPRESSURE_PA)
            self.assertEqual(detected.criterion, ARRIVAL_CRITERION)

    def test_remap_detect_arrival_ignores_8kpa_plateau(self) -> None:
        from remap_handoff_1d import HANDOFF_CRITERION

        with tempfile.TemporaryDirectory() as case:
            _write_probe(
                case,
                fo_name="watchdog_probe",
                location="0.59 0 0",
                samples=(
                    (0.0, 101325.0),
                    (1.0e-4, 120000.0),
                    (2.0e-4, 146000.0),
                    (3.0e-4, 400000.0),
                ),
            )
            record = initial_completion_record(
                mode=RUN_MODE_TERMINATE,
                requested_stop_radius_m=0.59,
                p_atm=101325.0,
                remap_for_2d=True,
                remap_radius_m=0.60,
                dr_1d_m=0.001,
                remap_front_buffer_cells=10,
                handoff_radius_m=0.59,
            )
            detected = detect_arrival_in_case(case, record)
            self.assertTrue(detected.wave_radius_reached)
            self.assertAlmostEqual(detected.detected_arrival_time_s, 3.0e-4)
            self.assertEqual(detected.criterion, HANDOFF_CRITERION)
            self.assertTrue(detected.remap_for_2d)

    def test_finalize_persists_stop_reason_and_arrival_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _arrived_record(case, mode=RUN_MODE_TERMINATE)
            record = finalize_completion_record(
                case,
                return_code=1,
                user_stopped=False,
                final_solver_time_s=0.00125,
                reached_end_time=False,
                end_time_s=0.03,
            )
            self.assertEqual(record.mode, RUN_MODE_TERMINATE)
            self.assertEqual(record.stop_reason, STOP_REASON_WAVE_RADIUS_REACHED)
            self.assertTrue(record.wave_radius_reached)
            self.assertAlmostEqual(record.detected_arrival_time_s, 0.0012)
            self.assertAlmostEqual(record.final_solver_time_s, 0.00125)
            self.assertAlmostEqual(record.end_time_s, 0.03)
            self.assertEqual(record.return_code, 1)
            self.assertEqual(record.probe_function_object, "watchdog_probe")
            persisted = read_completion_record(case)
            self.assertEqual(persisted.stop_reason, STOP_REASON_WAVE_RADIUS_REACHED)
            with open(os.path.join(case, COMPLETION_FILENAME), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["mode"], RUN_MODE_TERMINATE)
            self.assertEqual(payload["requested_radius_m"], 0.8)
            self.assertEqual(payload["endTime"], 0.03)
            self.assertTrue(payload["arrival_event"])
            self.assertEqual(payload["stop_reason"], STOP_REASON_WAVE_RADIUS_REACHED)

    def test_finalize_without_arrival_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=0.8,
                    p_atm=101325.0,
                ),
            )
            _write_probe(
                case,
                fo_name="watchdog_probe",
                location="0.8 0 0",
                samples=((0.0, 101325.0), (0.001, 105000.0)),
            )
            record = finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.005,
                reached_end_time=False,
            )
            self.assertEqual(record.stop_reason, STOP_REASON_NO_ARRIVAL)
            self.assertFalse(record.wave_radius_reached)

    def test_terminate_end_time_without_arrival_is_not_radius_success(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case, 0.03)
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=0.8,
                    p_atm=101325.0,
                    end_time_s=0.03,
                ),
            )
            _write_probe(
                case,
                fo_name="watchdog_probe",
                location="0.8 0 0",
                samples=((0.0, 101325.0), (0.03, 105000.0)),
            )
            _write_log(case, 0.03)
            record = finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.03,
                reached_end_time=True,
                end_time_s=0.03,
            )
            self.assertEqual(
                record.stop_reason, STOP_REASON_END_TIME_WITHOUT_ARRIVAL
            )
            self.assertFalse(record.wave_radius_reached)
            self.assertFalse(solver_run_succeeded(case, 0))


class SolverSuccessModeTests(unittest.TestCase):
    def test_terminate_is_done_only_after_verified_arrival(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            _arrived_record(case, mode=RUN_MODE_TERMINATE)
            _write_log(case, 0.00125)
            finalize_completion_record(
                case,
                return_code=1,
                user_stopped=False,
                final_solver_time_s=0.00125,
                reached_end_time=False,
            )
            self.assertTrue(solver_run_succeeded(case, 1))

    def test_terminate_without_verified_arrival_is_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=0.8,
                    p_atm=101325.0,
                ),
            )
            _write_log(case, 0.005776)
            finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.005776,
                reached_end_time=False,
            )
            self.assertFalse(solver_run_succeeded(case, 0))

    def test_reflect_with_verified_arrival_is_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            _arrived_record(case, mode=RUN_MODE_REFLECT)
            _write_log(case, 0.00125)
            record = finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.00125,
                reached_end_time=False,
            )
            self.assertTrue(record.wave_radius_reached)
            self.assertFalse(solver_run_succeeded(case, 0))

    def test_reflect_completes_successfully_at_end_time(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case, 0.03)
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_REFLECT,
                    requested_stop_radius_m=0.8,
                    p_atm=101325.0,
                    right_boundary="Reflect",
                    end_time_s=0.03,
                ),
            )
            _write_log(case, 0.03)
            record = finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.03,
                reached_end_time=True,
                end_time_s=0.03,
            )
            self.assertEqual(record.stop_reason, STOP_REASON_END_TIME_REACHED)
            self.assertFalse(record.wave_radius_reached)
            self.assertTrue(solver_run_succeeded(case, 0))

    def test_reflect_end_time_keeps_arrival_as_event_not_completion(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case, 0.03)
            _arrived_record(case, mode=RUN_MODE_REFLECT)
            _write_log(case, 0.03)
            record = finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.03,
                reached_end_time=True,
                end_time_s=0.03,
            )
            self.assertEqual(record.stop_reason, STOP_REASON_END_TIME_REACHED)
            self.assertTrue(record.wave_radius_reached)
            self.assertAlmostEqual(record.detected_arrival_time_s, 0.0012)
            self.assertTrue(solver_run_succeeded(case, 0))

    def test_reflect_records_arrival_event_on_user_stop(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            _arrived_record(case, mode=RUN_MODE_REFLECT)
            _write_log(case, 0.004)
            record = finalize_completion_record(
                case,
                return_code=1,
                user_stopped=True,
                final_solver_time_s=0.004,
                reached_end_time=False,
            )
            self.assertEqual(record.mode, RUN_MODE_REFLECT)
            self.assertEqual(record.stop_reason, STOP_REASON_USER_STOPPED)
            self.assertTrue(record.wave_radius_reached)
            self.assertAlmostEqual(record.detected_arrival_time_s, 0.0012)
            self.assertFalse(solver_run_succeeded(case, 1, user_stopped=True))
            with open(os.path.join(case, COMPLETION_FILENAME), encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["arrival_event"])
            self.assertEqual(payload["stop_reason"], STOP_REASON_USER_STOPPED)
            self.assertEqual(payload["mode"], RUN_MODE_REFLECT)

    def test_reflect_user_stop_without_arrival_is_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_REFLECT,
                    requested_stop_radius_m=0.8,
                    p_atm=101325.0,
                    right_boundary="Reflect",
                ),
            )
            _write_log(case, 0.002)
            record = finalize_completion_record(
                case,
                return_code=1,
                user_stopped=True,
                final_solver_time_s=0.002,
                reached_end_time=False,
            )
            self.assertEqual(record.stop_reason, STOP_REASON_USER_STOPPED)
            self.assertFalse(record.wave_radius_reached)
            self.assertFalse(solver_run_succeeded(case, 1, user_stopped=True))

    def test_unfinalized_reflect_record_is_not_done(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            _write_control(case, 0.03)
            write_completion_record(
                case,
                initial_completion_record(
                    stop_mode="end_time", requested_stop_radius_m=None, end_time_s=0.03
                ),
            )
            _write_log(case, 0.03)
            self.assertFalse(solver_run_succeeded(case, 0))
            self.assertFalse(solver_run_succeeded(case, 0, user_stopped=True))
            finalize_completion_record(
                case,
                return_code=0,
                user_stopped=False,
                final_solver_time_s=0.03,
                reached_end_time=True,
                end_time_s=0.03,
            )
            self.assertTrue(solver_run_succeeded(case, 0))
            self.assertFalse(solver_run_succeeded(case, 0, user_stopped=True))


class WaveWatchdogModeTests(unittest.TestCase):
    def test_stale_probe_file_does_not_trigger_wave_stop(self) -> None:
        from solver_runner import SolverRunner

        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=0.8,
                    p_atm=101325.0,
                ),
            )
            _write_probe(
                case,
                fo_name="watchdog_probe",
                location="0.8 0 0",
                samples=((0.0, 101325.0), (0.0012, 120000.0)),
            )
            probe_path = os.path.join(
                case, "postProcessing", "watchdog_probe", "0", "p"
            )
            runner = SolverRunner(case)
            runner._run_started_at = time.time()
            past = runner._run_started_at - 10.0
            os.utime(probe_path, (past, past))
            runner._check_watchdog_trigger(case)
            self.assertFalse(runner._watchdog_triggered)
            _write_probe(
                case,
                fo_name="watchdog_probe",
                location="0.8 0 0",
                samples=((0.0, 101325.0), (0.0012, 120000.0)),
            )
            runner._check_watchdog_trigger(case)
            self.assertTrue(runner._watchdog_triggered)

    def test_reflect_does_not_stop_on_verified_arrival(self) -> None:
        from solver_runner import SolverRunner

        with tempfile.TemporaryDirectory() as case:
            _write_control(case)
            _arrived_record(case, mode=RUN_MODE_REFLECT)
            runner = SolverRunner(case)
            runner._run_started_at = time.time() - 1.0
            with mock.patch("solver_runner.request_solver_write_and_stop") as write_now:
                runner._check_watchdog_trigger(case)
            write_now.assert_not_called()
            self.assertFalse(runner._watchdog_triggered)
            self.assertTrue(runner._wave_arrival_recorded)
            record = read_completion_record(case)
            self.assertIsNotNone(record)
            self.assertEqual(record.mode, RUN_MODE_REFLECT)
            self.assertTrue(record.wave_radius_reached)
            self.assertAlmostEqual(record.detected_arrival_time_s, 0.0012)

    def test_grace_terminate_does_not_mark_user_interrupt(self) -> None:
        from solver_runner import SolverRunner

        with tempfile.TemporaryDirectory() as case:
            runner = SolverRunner(case)
            runner._watchdog_triggered = True
            runner._watchdog_stop_requested_time = time.time() - 10.0
            proc = mock.Mock()
            proc.poll.return_value = None
            runner._proc = proc
            runner.keep_running = True
            runner._stop_requested = False
            with mock.patch("solver_runner.terminate_process_tree") as terminate:
                runner._maybe_stop_after_watchdog()
            terminate.assert_called_once_with(proc)
            self.assertTrue(runner.keep_running)
            self.assertFalse(runner._stop_requested)


class SourceModelPersistenceTests(unittest.TestCase):
    def test_initial_record_defaults_to_jwl(self) -> None:
        record = initial_completion_record(
            mode=RUN_MODE_TERMINATE, requested_stop_radius_m=1.0
        )
        self.assertEqual(record.source_model, "JWL_DETONATION")

    def test_ig_source_model_survives_write_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as case:
            write_completion_record(
                case,
                initial_completion_record(
                    mode=RUN_MODE_TERMINATE,
                    requested_stop_radius_m=1.0,
                    source_model="IG_ISOTHERMAL_BURST",
                ),
            )
            stored = read_completion_record(case)
            self.assertIsNotNone(stored)
            self.assertEqual(stored.source_model, "IG_ISOTHERMAL_BURST")
            reset = reset_completion_for_new_run(case)
            self.assertEqual(reset.source_model, "IG_ISOTHERMAL_BURST")


if __name__ == "__main__":
    unittest.main()
