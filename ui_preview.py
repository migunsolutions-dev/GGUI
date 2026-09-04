"""Launch a native GGUI preview without CFD, initialization, or case generation.

Example:
    python ui_preview.py --tab 2d --state ready --review
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable, Optional

from ui_metrics import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH


TAB_CHOICES = ("1d", "2d", "3d", "time_history", "validation")
STATE_CHOICES = ("ready", "initialized", "running", "completed")

# States applied only through public UI/status APIs. Never invent physics.
_TAB_STATES = {
    "1d": frozenset({"ready", "running"}),
    "2d": frozenset({"ready", "initialized", "running", "completed"}),
    "3d": frozenset({"ready", "running"}),
    "time_history": frozenset({"ready"}),
    "validation": frozenset({"ready"}),
}


class PreviewStateError(ValueError):
    """Requested preview chrome cannot be applied through public UI APIs."""


def _disconnect_all(signal) -> None:
    try:
        signal.disconnect()
    except TypeError:
        pass


def disable_cfd_actions(window) -> None:
    """Drop init/run connections so this process cannot generate a case."""
    signals: Iterable = (
        getattr(getattr(window, "tab_1d", None), "sig_request_run", None),
        getattr(getattr(window, "tab_1d", None), "sig_request_stop", None),
        getattr(getattr(window, "tab_2d", None), "sig_request_init", None),
        getattr(getattr(window, "tab_2d", None), "sig_request_run_exact_end", None),
        getattr(getattr(window, "tab_2d", None), "sig_request_stop", None),
        getattr(getattr(window, "tab_2d", None), "sig_request_prepare_transfer", None),
        getattr(getattr(window, "tab_3d", None), "sig_request_init", None),
        getattr(getattr(window, "tab_3d", None), "sig_request_run", None),
        getattr(getattr(window, "tab_3d", None), "sig_request_run_exact_1", None),
        getattr(getattr(window, "tab_3d", None), "sig_request_run_exact_end", None),
        getattr(getattr(window, "tab_3d", None), "sig_request_stop", None),
    )
    for signal in signals:
        if signal is not None:
            _disconnect_all(signal)


def select_tab(window, tab: str):
    mapping = {
        "1d": getattr(window, "tab_1d", None),
        "2d": getattr(window, "tab_2d", None),
        "3d": getattr(window, "tab_3d", None),
        "time_history": getattr(window, "tab_time_history", None),
        "validation": getattr(window, "tab_validation", None),
    }
    widget = mapping.get(tab)
    if widget is None:
        raise PreviewStateError(f"Unknown tab {tab!r}")
    window.tabs.setCurrentWidget(widget)
    return widget


def apply_preview_state(window, tab: str, state: str) -> None:
    """Apply chrome-only preview state. Never writes OpenFOAM dictionaries."""
    supported = _TAB_STATES.get(tab, frozenset())
    if state not in supported:
        allowed = ", ".join(sorted(supported)) or "(none)"
        raise PreviewStateError(
            f"Preview state {state!r} is not supported for tab {tab!r} "
            f"through public UI APIs. Supported: {allowed}. "
            "Refusing to invent physics values or simulate initialization."
        )
    status = getattr(window, "status_bar", None)
    if tab == "2d":
        from models_2d import SimulationState2D

        tab_widget = window.tab_2d
        if state == "ready":
            if status is not None:
                status.set_status("Ready", "#2ecc71")
            return
        mapping = {
            "initialized": (SimulationState2D.INITIALIZED, "Initialized", "#2ecc71"),
            "running": (SimulationState2D.RUNNING, "Running...", "#f39c12"),
            "completed": (SimulationState2D.COMPLETED, "Completed", "#2ecc71"),
        }
        sim_state, label, color = mapping[state]
        tab_widget.set_simulation_state(sim_state)
        if status is not None:
            status.set_status(label, color)
        return
    # 1D / 3D / Time History: status chrome only.
    if state == "ready" and status is not None:
        status.set_status("Ready", "#2ecc71")
    elif state == "running" and status is not None:
        status.set_status("Running...", "#f39c12")


def write_session(window, tab: str, state: str, review: bool, output_dir: str) -> str:
    from ui_review_mode import SESSION_NAME, review_dir

    folder = review_dir(output_dir)
    path = os.path.join(folder, SESSION_NAME)
    payload = {
        "pid": os.getpid(),
        "tab": tab,
        "state": state,
        "review": bool(review),
        "window_size": {
            "width": int(window.width()),
            "height": int(window.height()),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GGUI UI preview (no CFD / no case generation)."
    )
    parser.add_argument("--tab", required=True, choices=TAB_CHOICES)
    parser.add_argument("--state", required=True, choices=STATE_CHOICES)
    parser.add_argument(
        "--review",
        action="store_true",
        help="Enable UI Review Mode overlays (Ctrl+Shift+I toggles).",
    )
    return parser


def launch_preview(
    tab: str,
    state: str,
    review: bool,
    *,
    output_dir: Optional[str] = None,
    show: bool = True,
    exec_loop: bool = True,
):
    from ggui_logging import configure_logging
    from qt_bootstrap import prepare_qt_application
    from ui_review_mode import attach_ui_review, repo_root

    configure_logging()
    app = prepare_qt_application()
    from main_new import BlastFoamApp
    app.setStyle("Fusion")

    window = BlastFoamApp()
    window._ui_preview_mode = True
    window._ui_preview_tab = tab
    window._ui_preview_state = state
    select_tab(window, tab)
    apply_preview_state(window, tab, state)
    disable_cfd_actions(window)
    if show:
        window.show()
    window.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    window._opening_geometry_applied = True
    if hasattr(window, "_apply_opening_computational_left_width"):
        window._apply_opening_computational_left_width()
    app.processEvents()
    out = output_dir or os.path.join(repo_root(), "_ui_review")
    if review:
        attach_ui_review(window, enabled=True, output_dir=out)
        app.processEvents()
    write_session(window, tab, state, review, out)
    if exec_loop:
        return app.exec_()
    return window, app


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code = launch_preview(args.tab, args.state, args.review)
    except PreviewStateError as exc:
        print(f"ui_preview: {exc}", file=sys.stderr)
        return 2
    return int(code or 0)


if __name__ == "__main__":
    sys.exit(main())
