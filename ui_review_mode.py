"""Opt-in UI Review overlays for visual refinement.

Production behavior is unchanged until a preview (or test) attaches this
controller. Overlays are parented to the main window and are never inserted
into existing layouts, so widget geometry is preserved.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QKeySequence, QMouseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QBoxLayout,
    QScrollArea,
    QShortcut,
    QSplitter,
    QStatusBar,
    QTabBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

SCHEMA_VERSION = 1
REVIEW_DIRNAME = "_ui_review"
REGION_MAP_NAME = "region_map.json"
SELECTION_NAME = "selection.json"
SCREENSHOT_NAME = "current.png"
SESSION_NAME = "session.json"

_ROOT_ID_RE = re.compile(r"^R(\d+)$")
_NESTED_ID_RE = re.compile(r"^R(\d+(?:\.\d+)*)$")

_SOURCE_ATTRS = (
    ("tab_1d", "tab_1d.py"),
    ("tab_2d", "tab_2d.py"),
    ("tab_3d", "tab_3d_general.py"),
    ("tab_probes", "tab_probes.py"),
    ("tab_jotter", "tab_log.py"),
    ("status_bar", "main_new.py"),
    ("info_panel", "main_new.py"),
)


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def review_dir(base: Optional[str] = None) -> str:
    root = base or os.path.join(repo_root(), REVIEW_DIRNAME)
    os.makedirs(root, exist_ok=True)
    return root


def git_identity(cwd: Optional[str] = None) -> Dict[str, str]:
    """Return branch/HEAD when git is available. Never include env or secrets."""
    identity: Dict[str, str] = {}
    workdir = cwd or repo_root()
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workdir,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return identity
    if branch:
        identity["branch"] = branch
    if head:
        identity["head"] = head
    return identity


def _is_overlay(widget: Optional[QWidget]) -> bool:
    return bool(widget is not None and widget.property("ui_review_overlay"))


def _walk_parents(widget: Optional[QWidget]):
    current = widget
    while current is not None:
        yield current
        current = current.parentWidget()


def _is_under(window: QWidget, widget: Optional[QWidget]) -> bool:
    return widget is not None and any(w is window for w in _walk_parents(widget))


def _class_name(widget: QWidget) -> str:
    return widget.__class__.__name__


def _object_name(widget: QWidget) -> str:
    return str(widget.objectName() or "")


def visible_text(widget: QWidget) -> str:
    if isinstance(widget, QGroupBox):
        return str(widget.title() or "")
    for attr in ("text", "currentText", "placeholderText"):
        getter = getattr(widget, attr, None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                continue
            if isinstance(value, str) and value.strip():
                return value
    title = getattr(widget, "title", None)
    if callable(title):
        try:
            value = title()
        except Exception:
            value = ""
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _tabwidget_of(widget: QWidget) -> Optional[QTabWidget]:
    for parent in _walk_parents(widget):
        if isinstance(parent, QTabWidget):
            return parent
    return None


def _tab_text_for_page(page: QWidget) -> str:
    owner = page.parentWidget()
    if isinstance(owner, QTabWidget):
        idx = owner.indexOf(page)
        if idx >= 0:
            return owner.tabText(idx)
    return ""


def structural_locator(widget: QWidget, main_tab: str) -> str:
    parts: List[str] = []
    is_leaf = True
    for node in _walk_parents(widget):
        cls = _class_name(node)
        name = _object_name(node)
        extra = ""
        if isinstance(node, QGroupBox):
            extra = f"[title={node.title()}]"
        elif isinstance(node, QTabWidget):
            titles = ",".join(node.tabText(i) for i in range(node.count()))
            extra = f"[tabs={titles}]"
        elif name:
            extra = f"[name={name}]"
        else:
            text = visible_text(node) or _tab_text_for_page(node)
            if text:
                extra = f"[text={text}]" if is_leaf else f"[tab={text}]"
            elif is_leaf:
                info = layout_info(node)
                if info and info.get("index") is not None:
                    extra = f"[index={info['index']}]"
                elif info and info.get("row") is not None:
                    extra = f"[row={info['row']}]"
        parts.append(f"{cls}{extra}")
        is_leaf = False
        if isinstance(node, QMainWindow):
            break
    parts.reverse()
    return f"{main_tab}::{'/'.join(parts)}"


def layout_info(widget: QWidget) -> Optional[Dict[str, Any]]:
    parent = widget.parentWidget()
    if parent is None:
        return None
    if isinstance(parent, QSplitter):
        for index in range(parent.count()):
            if parent.widget(index) is widget:
                return {"class": "QSplitter", "index": index}
    layout = parent.layout()
    if layout is None:
        return None
    info: Dict[str, Any] = {"class": layout.__class__.__name__}
    if isinstance(layout, QFormLayout):
        for row in range(layout.rowCount()):
            for role, role_name in (
                (QFormLayout.LabelRole, "label"),
                (QFormLayout.FieldRole, "field"),
                (QFormLayout.SpanningRole, "spanning"),
            ):
                item = layout.itemAt(row, role)
                if item is not None and item.widget() is widget:
                    info["row"] = row
                    info["role"] = role_name
                    return info
    if isinstance(layout, QGridLayout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is not None and item.widget() is widget:
                row, column, row_span, column_span = layout.getItemPosition(i)
                info.update(
                    row=row,
                    column=column,
                    row_span=row_span,
                    column_span=column_span,
                )
                return info
    if isinstance(layout, QBoxLayout):
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is not None and item.widget() is widget:
                info["index"] = index
                return info
    return info


def main_tab_key(window: QWidget) -> str:
    tabs = getattr(window, "tabs", None)
    current = tabs.currentWidget() if tabs is not None else None
    if current is getattr(window, "tab_1d", None):
        return "1d"
    if current is getattr(window, "tab_2d", None):
        return "2d"
    if current is getattr(window, "tab_3d", None):
        return "3d"
    if tabs is not None:
        return f"tab:{tabs.tabText(tabs.currentIndex())}"
    return "window"


def active_sub_tab(widget: QWidget, window: QWidget) -> str:
    main_tabs = getattr(window, "tabs", None)
    for parent in _walk_parents(widget):
        if isinstance(parent, QTabWidget) and parent is not main_tabs:
            return parent.tabText(parent.currentIndex())
    return ""


def source_file_hint(widget: QWidget, window: QWidget) -> Optional[str]:
    mapping: List[Tuple[QWidget, str]] = []
    for attr, path in _SOURCE_ATTRS:
        target = getattr(window, attr, None)
        if isinstance(target, QWidget):
            mapping.append((target, path))
    for node in _walk_parents(widget):
        for target, path in mapping:
            if node is target:
                return path
        if isinstance(node, QToolBar):
            return "main_new.py"
    return None


def _sort_key(widget: QWidget) -> Tuple[int, int, str, str]:
    geo = widget.geometry()
    return (geo.y(), geo.x(), _class_name(widget), visible_text(widget) or _object_name(widget))


def _is_primary_scroll(widget: QScrollArea, window: QWidget) -> bool:
    parent = widget.parentWidget()
    if isinstance(parent, QTabWidget):
        return True
    if parent is getattr(window, "tab_1d", None):
        return True
    if parent is getattr(window, "tab_2d", None):
        return True
    if parent is getattr(window, "tab_3d", None):
        return True
    return False


class _RegionNode:
    __slots__ = ("widget", "kind", "title", "parent", "children")

    def __init__(self, widget: QWidget, kind: str, title: str):
        self.widget = widget
        self.kind = kind
        self.title = title
        self.parent: Optional["_RegionNode"] = None
        self.children: List["_RegionNode"] = []


def discover_region_nodes(window: QWidget, scope: QWidget) -> List[_RegionNode]:
    """Meaningful containers only — not every QWidget."""
    nodes: Dict[int, _RegionNode] = {}

    def add(widget: QWidget, kind: str, title: str) -> _RegionNode:
        key = id(widget)
        existing = nodes.get(key)
        if existing is not None:
            return existing
        node = _RegionNode(widget, kind, title)
        nodes[key] = node
        return node

    if isinstance(scope, QWidget):
        toolbar = getattr(window, "_main_toolbar", None)
        if isinstance(toolbar, QToolBar) and toolbar.isVisible():
            add(toolbar, "toolbar", toolbar.windowTitle() or "Toolbar")
        status = getattr(window, "status_bar", None)
        if isinstance(status, QWidget) and status.isVisible():
            add(status, "status", "Status")

        splitters = [scope] if isinstance(scope, QSplitter) else scope.findChildren(QSplitter)
        if isinstance(getattr(scope, "_main_splitter", None), QSplitter):
            splitters = [scope._main_splitter]
        for splitter in splitters:
            if splitter is None or _is_overlay(splitter):
                continue
            if splitter.count() >= 2 and splitter.isVisible():
                left = splitter.widget(0)
                if left is not None and left.isVisible():
                    add(left, "panel", "Left panel")
                right = splitter.widget(1)
                if right is not None and right.isVisible():
                    add(right, "panel", "Right panel")

        for tabwidget in scope.findChildren(QTabWidget):
            if _is_overlay(tabwidget) or not tabwidget.isVisible():
                continue
            titles = [tabwidget.tabText(i) for i in range(tabwidget.count())]
            add(tabwidget, "tabs", "|".join(titles))
            for i in range(tabwidget.count()):
                page = tabwidget.widget(i)
                if page is None:
                    continue
                add(page, "tab_page", tabwidget.tabText(i))

        for group in scope.findChildren(QGroupBox):
            if _is_overlay(group) or not group.isVisible():
                continue
            add(group, "group", group.title())

        for scroll in scope.findChildren(QScrollArea):
            if _is_overlay(scroll) or not scroll.isVisible():
                continue
            if _is_primary_scroll(scroll, window):
                add(scroll, "scroll", _tab_text_for_page(scroll) or "Scroll")

    nodes_list = list(nodes.values())
    by_widget = {id(n.widget): n for n in nodes_list}
    for node in nodes_list:
        ancestor = node.widget.parentWidget()
        while ancestor is not None and ancestor is not window:
            parent_node = by_widget.get(id(ancestor))
            if parent_node is not None and parent_node is not node:
                node.parent = parent_node
                parent_node.children.append(node)
                break
            ancestor = ancestor.parentWidget()
    for node in nodes_list:
        node.children.sort(key=lambda child: _sort_key(child.widget))
        # Unique children (a widget can only have one parent node).
        seen = set()
        unique = []
        for child in node.children:
            if id(child.widget) in seen:
                continue
            seen.add(id(child.widget))
            unique.append(child)
        node.children = unique
    return nodes_list


def _parse_ids(keys) -> Tuple[int, Dict[str, int]]:
    max_root = 0
    child_max: Dict[str, int] = {}
    for key in keys:
        match = _NESTED_ID_RE.fullmatch(key)
        if not match:
            continue
        parts = key[1:].split(".")
        max_root = max(max_root, int(parts[0]))
        if len(parts) > 1:
            parent = "R" + ".".join(parts[:-1])
            child_max[parent] = max(child_max.get(parent, 0), int(parts[-1]))
    return max_root, child_max


class RegionRegistry:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "by_tab": {}}
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(loaded, dict) and isinstance(loaded.get("by_tab"), dict):
            self.data = loaded
            self.data.setdefault("schema_version", SCHEMA_VERSION)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def tab_map(self, tab_key: str) -> Dict[str, Dict[str, Any]]:
        by_tab = self.data.setdefault("by_tab", {})
        slot = by_tab.setdefault(tab_key, {})
        return slot

    def fingerprint(self, widget: QWidget, tab_key: str) -> Dict[str, str]:
        return {
            "locator": structural_locator(widget, tab_key),
            "objectName": _object_name(widget),
            "class": _class_name(widget),
            "title": visible_text(widget) or _tab_text_for_page(widget),
        }

    def _match_id(
        self,
        fp: Dict[str, str],
        stored: Dict[str, Dict[str, Any]],
        used: set,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (region_id, ambiguity_note)."""
        name = fp.get("objectName") or ""
        if name:
            hits = [
                rid
                for rid, rec in stored.items()
                if rid not in used and rec.get("objectName") == name
            ]
            if len(hits) == 1:
                return hits[0], None
            if len(hits) > 1:
                return None, f"objectName {name!r} matches {hits}"
        locator = fp.get("locator") or ""
        hits = [
            rid
            for rid, rec in stored.items()
            if rid not in used and rec.get("locator") == locator
        ]
        if len(hits) == 1:
            return hits[0], None
        title = fp.get("title") or ""
        cls = fp.get("class") or ""
        if title and cls:
            hits = [
                rid
                for rid, rec in stored.items()
                if rid not in used
                and rec.get("class") == cls
                and rec.get("title") == title
            ]
            if len(hits) == 1:
                return hits[0], None
            if len(hits) > 1:
                return None, f"{cls} title {title!r} matches {hits}"
        return None, None

    def assign(
        self,
        nodes: List[_RegionNode],
        tab_key: str,
    ) -> Tuple[Dict[int, str], List[str]]:
        stored = self.tab_map(tab_key)
        used: set = set()
        assigned: Dict[int, str] = {}
        ambiguities: List[str] = []
        max_root, child_max = _parse_ids(stored.keys())

        def next_id(parent_id: Optional[str]) -> str:
            nonlocal max_root
            if parent_id:
                n = child_max.get(parent_id, 0) + 1
                child_max[parent_id] = n
                return f"{parent_id}.{n}"
            max_root += 1
            return f"R{max_root}"

        roots = [node for node in nodes if node.parent is None]
        roots.sort(key=lambda node: _sort_key(node.widget))

        def visit(node: _RegionNode, parent_id: Optional[str]) -> None:
            fp = self.fingerprint(node.widget, tab_key)
            rid, note = self._match_id(fp, stored, used)
            if note:
                ambiguities.append(note)
            if rid is None:
                rid = next_id(parent_id)
            used.add(rid)
            assigned[id(node.widget)] = rid
            record = dict(fp)
            record["kind"] = node.kind
            stored[rid] = record
            for child in node.children:
                if id(child.widget) in assigned:
                    continue
                visit(child, rid)

        for root in roots:
            visit(root, None)
        return assigned, ambiguities


class ReviewBadge(QLabel):
    def __init__(self, parent: QWidget, text: str, bg: str, fg: str = "#ffffff"):
        super().__init__(parent)
        self.setProperty("ui_review_overlay", True)
        self.setText(text)
        self.setAlignment(Qt.AlignCenter)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"background:{bg}; color:{fg}; font-weight:bold; font-size:10px;"
            " padding:1px 5px; border:1px solid #111;"
        )
        self.adjustSize()
        self.show()
        self.raise_()


class ReviewHighlight(QWidget):
    def __init__(self, parent: QWidget, color: QColor):
        super().__init__(parent)
        self.setProperty("ui_review_overlay", True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        r, g, b = color.red(), color.green(), color.blue()
        self.setStyleSheet(
            f"background-color: rgba({r},{g},{b},45); border: 2px solid rgb({r},{g},{b});"
        )
        self.show()
        self.raise_()


class UIReviewController(QObject):
    """Window-level overlay controller. Disabled until enable() is called."""

    def __init__(self, window: QMainWindow, output_dir: Optional[str] = None):
        super().__init__(window)
        self.window = window
        self.output_dir = review_dir(output_dir)
        self.enabled = False
        self.preview_state = str(getattr(window, "_ui_preview_state", "") or "ready")
        self.sources: List[Dict[str, Any]] = []
        self.anchor: Optional[Dict[str, Any]] = None
        self.region_ids: Dict[int, str] = {}
        self.region_nodes: List[_RegionNode] = []
        self.ambiguities: List[str] = []
        self._overlays: List[QWidget] = []
        self._filter_installed = False
        self.registry = RegionRegistry(os.path.join(self.output_dir, REGION_MAP_NAME))
        self._toggle = QShortcut(QKeySequence("Ctrl+Shift+I"), window)
        self._toggle.setContext(Qt.ApplicationShortcut)
        self._toggle.activated.connect(self.toggle)
        self._tab_hooks: List[QTabWidget] = []

    def toggle(self) -> None:
        if self.enabled:
            self.disable()
        else:
            self.enable()

    def enable(self) -> None:
        if not self.enabled:
            self.enabled = True
            app = QApplication.instance()
            if app is not None and not self._filter_installed:
                app.installEventFilter(self)
                self.window.installEventFilter(self)
                self._filter_installed = True
            self._hook_tab_changes()
        self.refresh_regions()
        self._rebuild_overlays()
        self.write_outputs()

    def disable(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        app = QApplication.instance()
        if app is not None and self._filter_installed:
            app.removeEventFilter(self)
            self.window.removeEventFilter(self)
            self._filter_installed = False
        self._clear_overlays()

    def clear_selections(self) -> None:
        self.sources = []
        self.anchor = None
        if self.enabled:
            self._rebuild_overlays()
            self.write_outputs()

    def refresh_regions(self) -> None:
        tab_key = main_tab_key(self.window)
        scope = self._scope_widget()
        self.region_nodes = discover_region_nodes(self.window, scope)
        self.region_ids, notes = self.registry.assign(self.region_nodes, tab_key)
        self.ambiguities = list(notes)
        self.registry.save()

    def _scope_widget(self) -> QWidget:
        # BlastFoamApp: number regions inside the active simulation tab.
        # Generic preview/test windows: use the central widget so nested
        # QTabWidgets (Mesh / Execution) remain in scope.
        if getattr(self.window, "tab_1d", None) is not None:
            tabs = getattr(self.window, "tabs", None)
            current = tabs.currentWidget() if tabs is not None else None
            if isinstance(current, QWidget):
                return current
        central = self.window.centralWidget()
        return central if central is not None else self.window

    def _hook_tab_changes(self) -> None:
        widgets = [self.window]
        central = self.window.centralWidget()
        if central is not None:
            widgets.append(central)
            widgets.extend(central.findChildren(QTabWidget))
        tabs = getattr(self.window, "tabs", None)
        if isinstance(tabs, QTabWidget):
            widgets.append(tabs)
        seen = set()
        for widget in widgets:
            if not isinstance(widget, QTabWidget) or id(widget) in seen:
                continue
            seen.add(id(widget))
            if widget in self._tab_hooks:
                continue
            widget.currentChanged.connect(self._on_tab_changed)
            self._tab_hooks.append(widget)

    def _on_tab_changed(self, _index: int = 0) -> None:
        if not self.enabled:
            return
        self.refresh_regions()
        self._rebuild_overlays()
        self.write_outputs()

    def eventFilter(self, obj, event):  # noqa: N802
        if not self.enabled:
            return False
        et = event.type()
        if et == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            if QApplication.activeModalWidget() is None:
                self.clear_selections()
                return True
        if et == QEvent.Resize and obj is self.window:
            self._rebuild_overlays()
        if et == QEvent.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() != Qt.LeftButton:
                return False
            target = QApplication.widgetAt(event.globalPos())
            if target is None and isinstance(obj, QWidget):
                target = obj
            if _is_overlay(target):
                return True
            if not _is_under(self.window, target):
                return False
            self.select_at(event.globalPos(), event.modifiers())
            return True
        return False

    def select_at(self, global_pos: QPoint, modifiers: Qt.KeyboardModifiers) -> Optional[Dict[str, Any]]:
        widget = QApplication.widgetAt(global_pos)
        if widget is None or _is_overlay(widget) or not _is_under(self.window, widget):
            return None
        return self.select_widget(widget, modifiers, global_pos)

    def select_widget(
        self,
        widget: QWidget,
        modifiers: Qt.KeyboardModifiers = Qt.NoModifier,
        global_pos: Optional[QPoint] = None,
    ) -> Dict[str, Any]:
        kind = "widget"
        tab_meta: Dict[str, Any] = {}
        target = widget
        if isinstance(widget, QTabBar):
            owner = widget.parentWidget()
            while owner is not None and not isinstance(owner, QTabWidget):
                owner = owner.parentWidget()
            idx = -1
            if global_pos is not None:
                idx = widget.tabAt(widget.mapFromGlobal(global_pos))
            if idx < 0 and owner is not None:
                idx = owner.currentIndex()
            if owner is not None and idx >= 0:
                page = owner.widget(idx)
                if page is not None:
                    target = page
                    kind = "tab"
                    tab_meta = {
                        "tab_text": owner.tabText(idx),
                        "tab_index": idx,
                        "owning_tabwidget": _class_name(owner),
                        "owning_objectName": _object_name(owner),
                        "page_objectName": _object_name(page),
                        "sibling_tabs": [owner.tabText(i) for i in range(owner.count())],
                    }
        record = self._describe(target, kind, tab_meta)
        if modifiers & Qt.ShiftModifier and not (modifiers & Qt.ControlModifier):
            record["alias"] = "A1"
            self.anchor = record
        elif modifiers & Qt.ControlModifier:
            aliases = {item.get("alias") for item in self.sources}
            if record.get("locator") not in {item.get("locator") for item in self.sources}:
                n = 1
                while f"S{n}" in aliases:
                    n += 1
                record["alias"] = f"S{n}"
                self.sources.append(record)
        else:
            record["alias"] = "S1"
            self.sources = [record]
        if self.enabled:
            self._rebuild_overlays()
            self.write_outputs()
        return record

    def _describe(
        self,
        widget: QWidget,
        kind: str,
        tab_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        tab_key = main_tab_key(self.window)
        geo = widget.geometry()
        top_left = widget.mapTo(self.window, QPoint(0, 0))
        region = self._region_for(widget)
        hierarchy = []
        for node in _walk_parents(widget):
            label = _class_name(node)
            name = _object_name(node)
            title = visible_text(node) or _tab_text_for_page(node)
            if name:
                label += f"#{name}"
            if title:
                label += f":{title}"
            hierarchy.append(label)
            if node is self.window:
                break
        record: Dict[str, Any] = {
            "alias": "S1",
            "kind": kind,
            "class": _class_name(widget),
            "objectName": _object_name(widget),
            "text": visible_text(widget) or tab_meta.get("tab_text", ""),
            "region": region,
            "hierarchy": hierarchy,
            "layout": layout_info(widget),
            "geometry": {
                "x": int(geo.x()),
                "y": int(geo.y()),
                "width": int(geo.width()),
                "height": int(geo.height()),
                "window_x": int(top_left.x()),
                "window_y": int(top_left.y()),
            },
            "enabled": bool(widget.isEnabled()),
            "visible": bool(widget.isVisible()),
            "main_tab": tab_key,
            "sub_tab": active_sub_tab(widget, self.window),
            "locator": structural_locator(widget, tab_key),
        }
        hint = source_file_hint(widget, self.window)
        if hint:
            record["source_file_hint"] = hint
        if tab_meta:
            record.update(tab_meta)
        return record

    def _region_for(self, widget: QWidget) -> Optional[str]:
        for node in _walk_parents(widget):
            rid = self.region_ids.get(id(node))
            if rid:
                return rid
        return None

    def _clear_overlays(self) -> None:
        for overlay in self._overlays:
            overlay.hide()
            overlay.setParent(None)
            overlay.deleteLater()
        self._overlays = []

    def _rebuild_overlays(self) -> None:
        self._clear_overlays()
        if not self.enabled:
            return
        for node in self.region_nodes:
            if not node.widget.isVisible():
                continue
            rid = self.region_ids.get(id(node.widget))
            if not rid:
                continue
            self._place_overlay(node.widget, rid, QColor("#2980b9"), "#ffffff", highlight=False)
        for item in self.sources:
            self._place_record(item, item["alias"], QColor("#e74c3c"), "#ffffff")
        if self.anchor is not None:
            self._place_record(self.anchor, "A1", QColor("#f1c40f"), "#111111")
        for overlay in self._overlays:
            overlay.raise_()

    def _widget_from_record(self, record: Dict[str, Any]) -> Optional[QWidget]:
        locator = record.get("locator")
        tab_key = main_tab_key(self.window)
        scope = self._scope_widget()
        widgets = [scope]
        widgets.extend(scope.findChildren(QWidget))
        widgets.extend(self.window.findChildren(QWidget))
        for widget in widgets:
            if _is_overlay(widget):
                continue
            if structural_locator(widget, tab_key) == locator:
                return widget
        return None

    def _tab_header_rect(self, record: Dict[str, Any]) -> Optional[QRect]:
        siblings = record.get("sibling_tabs")
        idx = record.get("tab_index")
        if not isinstance(siblings, list) or not isinstance(idx, int):
            return None
        for tabwidget in self.window.findChildren(QTabWidget):
            titles = [tabwidget.tabText(i) for i in range(tabwidget.count())]
            if titles != siblings:
                continue
            bar = tabwidget.tabBar()
            rect = bar.tabRect(idx)
            origin = bar.mapTo(self.window, rect.topLeft())
            return QRect(origin, rect.size())
        return None

    def _place_record(self, record: Dict[str, Any], text: str, color: QColor, fg: str) -> None:
        if record.get("kind") == "tab":
            header = self._tab_header_rect(record)
            if header is not None:
                frame = ReviewHighlight(self.window, color)
                frame.setGeometry(header)
                self._overlays.append(frame)
                badge = ReviewBadge(
                    self.window,
                    text,
                    f"rgb({color.red()},{color.green()},{color.blue()})",
                    fg,
                )
                badge.move(header.topLeft() + QPoint(2, 2))
                self._overlays.append(badge)
                return
        widget = self._widget_from_record(record)
        if widget is None or not widget.isVisible():
            return
        self._place_overlay(widget, text, color, fg, highlight=True)

    def _place_overlay(
        self,
        widget: QWidget,
        text: str,
        color: QColor,
        fg: str,
        *,
        highlight: bool,
    ) -> None:
        origin = widget.mapTo(self.window, QPoint(0, 0))
        size = widget.size()
        if highlight and size.width() > 0 and size.height() > 0:
            frame = ReviewHighlight(self.window, color)
            frame.setGeometry(QRect(origin, size))
            self._overlays.append(frame)
        badge = ReviewBadge(
            self.window,
            text,
            f"rgb({color.red()},{color.green()},{color.blue()})",
            fg,
        )
        badge.move(origin + QPoint(2, 2))
        self._overlays.append(badge)

    def visible_region_map(self) -> List[Dict[str, Any]]:
        rows = []
        for node in self.region_nodes:
            rid = self.region_ids.get(id(node.widget))
            if not rid:
                continue
            fp = self.registry.fingerprint(node.widget, main_tab_key(self.window))
            rows.append(
                {
                    "id": rid,
                    "kind": node.kind,
                    "visible": bool(node.widget.isVisible()),
                    **fp,
                }
            )
        rows.sort(key=lambda row: row["id"])
        return rows

    def write_outputs(self) -> Dict[str, str]:
        selection_path = os.path.join(self.output_dir, SELECTION_NAME)
        screenshot_path = os.path.join(self.output_dir, SCREENSHOT_NAME)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git": git_identity(),
            "active_tab": main_tab_key(self.window),
            "preview_state": self.preview_state,
            "window_size": {
                "width": int(self.window.width()),
                "height": int(self.window.height()),
            },
            "visible_regions": self.visible_region_map(),
            "sources": self.sources,
            "anchor": self.anchor,
            "screenshot": os.path.join(REVIEW_DIRNAME, SCREENSHOT_NAME).replace("\\", "/"),
            "ambiguities": list(self.ambiguities),
        }
        with open(selection_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        QApplication.processEvents()
        pixmap = self.window.grab()
        pixmap.save(screenshot_path, "PNG")
        return {"selection": selection_path, "screenshot": screenshot_path}

    def overlay_widgets(self) -> List[QWidget]:
        return list(self._overlays)


def attach_ui_review(
    window: QMainWindow,
    *,
    enabled: bool = True,
    output_dir: Optional[str] = None,
) -> UIReviewController:
    existing = getattr(window, "_ui_review", None)
    if isinstance(existing, UIReviewController):
        controller = existing
        if output_dir:
            controller.output_dir = review_dir(output_dir)
            controller.registry = RegionRegistry(os.path.join(controller.output_dir, REGION_MAP_NAME))
    else:
        controller = UIReviewController(window, output_dir=output_dir)
        window._ui_review = controller
    controller.preview_state = str(getattr(window, "_ui_preview_state", "") or controller.preview_state)
    if enabled:
        controller.enable()
    return controller
