"""Shared OpenGL/QtInteractor lifecycle helpers for embedded PyVista viewers."""
from __future__ import annotations

import inspect
import logging
import time
import weakref
from typing import Any, Dict, Optional

LOG = logging.getLogger("ggui.viewer_gl")

# Live embedded viewers keyed by Python id. Values are weak refs so GC still works.
_VIEWER_REGISTRY: Dict[int, Any] = {}


def register_viewer(owner: str, viewer: Any, plotter: Any) -> dict:
    """Record a live viewer/render-window identity for diagnostics."""
    info = {
        "owner": owner,
        "viewer_id": id(viewer),
        "plotter_id": id(plotter) if plotter is not None else None,
        "render_window": None,
        "created_at": time.time(),
        "shutdown_at": None,
    }
    try:
        if plotter is not None and hasattr(plotter, "render_window"):
            info["render_window"] = hex(id(plotter.render_window))
    except Exception:
        pass
    _VIEWER_REGISTRY[id(viewer)] = {"info": info, "ref": weakref.ref(viewer)}
    LOG.info(
        "viewer_created owner=%s viewer_id=%s rw=%s",
        owner,
        info["viewer_id"],
        info["render_window"],
    )
    return info


def unregister_viewer(viewer: Any) -> None:
    entry = _VIEWER_REGISTRY.pop(id(viewer), None)
    if entry is None:
        return
    entry["info"]["shutdown_at"] = time.time()
    LOG.info(
        "viewer_shutdown owner=%s viewer_id=%s rw=%s",
        entry["info"].get("owner"),
        entry["info"].get("viewer_id"),
        entry["info"].get("render_window"),
    )


def live_viewer_registry_snapshot() -> list:
    out = []
    dead = []
    for key, entry in _VIEWER_REGISTRY.items():
        if entry["ref"]() is None:
            dead.append(key)
            continue
        out.append(dict(entry["info"]))
    for key in dead:
        _VIEWER_REGISTRY.pop(key, None)
    return out


def stop_plotter_render_timer(plotter: Any) -> None:
    """Stop pyvistaqt auto-update timer if present."""
    if plotter is None:
        return
    timer = getattr(plotter, "render_timer", None)
    if timer is None:
        return
    try:
        timer.stop()
    except Exception:
        pass


def create_embedded_interactor(parent_frame, *, auto_update: bool = False):
    """Create a QtInteractor that does NOT auto-render on a hidden timer.

    pyvistaqt defaults auto_update=5.0 (render every 200 ms). On Windows, a
    hidden QTabWidget page's vtkWin32OpenGLRenderWindow then repeatedly calls
    wglMakeCurrent against an invalid/inactive HWND.
    """
    from pyvistaqt import QtInteractor

    plotter = QtInteractor(parent_frame, auto_update=False if not auto_update else auto_update)
    stop_plotter_render_timer(plotter)
    return plotter


def close_plotter_safely(plotter: Any, *, owner: str = "") -> None:
    """Finalize an embedded plotter at most once while its Qt context is valid."""
    if plotter is None:
        return
    stop_plotter_render_timer(plotter)
    closed = bool(getattr(plotter, "_closed", False))
    if closed:
        return
    try:
        plotter.close()
        LOG.info("plotter_closed owner=%s plotter_id=%s", owner, id(plotter))
    except Exception as exc:
        LOG.warning("plotter_close_failed owner=%s err=%s", owner, exc)


def scalar_bar_kwargs(**requested):
    """Pass only kwargs accepted by the installed Plotter.add_scalar_bar."""
    try:
        import pyvista as pv

        accepted = set(inspect.signature(pv.Plotter.add_scalar_bar).parameters)
    except Exception:
        accepted = set()
    return {key: value for key, value in requested.items() if key in accepted}
