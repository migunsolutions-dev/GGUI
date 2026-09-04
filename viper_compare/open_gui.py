"""Open production GGUI on completed comparison cases and show Validation."""
from __future__ import annotations

import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

print("open_gui: starting", flush=True)

from PyQt5.QtCore import QEvent, QObject, QTimer
from PyQt5.QtWidgets import QMessageBox

from ggui_logging import configure_logging
from qt_bootstrap import prepare_qt_application


class AutoAcceptDialogs(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QMessageBox):
            QTimer.singleShot(80, obj.accept)
        return super().eventFilter(obj, event)


def open_validation(run_dir: str) -> int:
    summary_path = os.path.join(run_dir, "report", "summary.json")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)
    cases = summary["ggui_cases"]
    case_1d = cases["GGUI_BF_1D"]["case_dir"]
    case_2d_dir = cases["GGUI_BF_2D_DIRECT"]["case_dir"]
    print(f"open_gui: 1D {case_1d}", flush=True)
    print(f"open_gui: 2D {case_2d_dir}", flush=True)

    configure_logging()
    app = prepare_qt_application()
    app.setStyle("Fusion")
    acceptor = AutoAcceptDialogs()
    app.installEventFilter(acceptor)
    print("open_gui: importing BlastFoamApp", flush=True)
    from main_new import BlastFoamApp
    from tab_validation import MODE_KB

    print("open_gui: creating window", flush=True)
    win = BlastFoamApp()
    win.showMaximized()
    print("open_gui: window shown", flush=True)

    def apply():
        try:
            win.tab_time_history._run_cases["1d"] = case_1d
            win.tab_time_history._run_cases["2d"] = case_2d_dir
            win.tab_1d.refresh_remap_status(case_1d)
            win.tab_2d._last_1d_case_dir = case_1d
            win.active_case_dir_2d = case_2d_dir
            win.tab_1d.set_case_inputs(
                {
                    "radius": 1.0,
                    "cell_size": 0.001,
                    "p_atm": 101325.0,
                    "t_atm": 288.0,
                    "mass_kg": 1.0,
                    "rho_charge": 1600.0,
                    "energy_j_per_kg": 4.52e6,
                    "material_name": "Custom",
                }
            )
            win.tab_2d.set_case_inputs(
                {
                    "radius": 2.0,
                    "height": 2.0,
                    "cell_size": 0.01,
                    "mass_kg": 1.0,
                    "height_of_burst": 1.0,
                    "p_atm": 101325.0,
                    "t_atm": 288.0,
                    "material_name": "Custom",
                }
            )
            win.tabs.setCurrentWidget(win.tab_validation)
            val = win.tab_validation
            val.refresh_current_run()
            val.combo_mode.setCurrentText(MODE_KB)
            val.radio_kb_sph.setChecked(True)
            val.combo_kb_source.setCurrentText("UFC 3-340-02")
            val.radio_kb_range.setChecked(True)
            val.radio_auto_points.setChecked(True)
            val.chk_show_1d.setChecked(True)
            val.chk_show_2d.setChecked(True)
            val.chk_show_3d.setChecked(False)
            val._use_current()
            val._redraw()
            print("GGUI Validation opened: Free-air spherical, 1D+2D.", flush=True)
            print("Leave this window open for inspection.", flush=True)
        except Exception:
            traceback.print_exc()

    QTimer.singleShot(400, apply)
    return app.exec_()


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    if not run_dir:
        raise SystemExit("usage: python -m viper_compare.open_gui <run_dir>")
    raise SystemExit(open_validation(run_dir))
