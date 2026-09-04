"""Time History Viewer catalog, filters, Add/Clear, and probe-file plot."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PyQt5.QtWidgets import QApplication

from models_2d import ProbePoint2D
from probes_model import ProbePoint
from tab_time_history import (
    ImportedSeries,
    TabTimeHistory,
    catalog_rows,
    latest_probe_field_file,
    padded_axis_limits,
    parse_external_timeseries,
    parse_probe_history,
    wrap_legend_name,
)


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class CatalogRowsTests(unittest.TestCase):
    def test_maps_1d_2d_3d_and_ignores_regions(self):
        rows = catalog_rows(
            gauges_1d=((1.5, "G1"),),
            probes_2d=(ProbePoint2D("P1", 2.0, 0.5),),
            probes_3d=(ProbePoint("Q1", 1.0, 2.0, 3.0),),
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].gauge_id, "1D-1")
        self.assertEqual(rows[0].x, 1.5)
        self.assertEqual(rows[0].y, 0.0)
        self.assertEqual(rows[0].z, 0.0)
        self.assertEqual(rows[1].gauge_id, "2D-1")
        self.assertEqual(rows[1].x, 2.0)
        self.assertEqual(rows[1].y, 0.5)
        self.assertEqual(rows[2].gauge_id, "3D-1")
        self.assertEqual((rows[2].x, rows[2].y, rows[2].z), (1.0, 2.0, 3.0))


class ProbeHistoryReaderTests(unittest.TestCase):
    def test_parse_and_latest_field_file(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            path = os.path.join(fo, "p")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.001 201325\n")
            found = latest_probe_field_file(td, "gauges1d", "p")
            self.assertEqual(os.path.normpath(found), os.path.normpath(path))
            locs, times, columns = parse_probe_history(path)
            self.assertEqual(locs[0], "1 0 0")
            self.assertEqual(times, [0.0, 0.001])
            self.assertEqual(columns[0], [101325.0, 201325.0])

    def test_parse_named_column_csv_timeseries(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "gauges.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Time,G1,G2\n")
                handle.write("0.0,1,2\n")
                handle.write("0.1,3,4\n")
            times, series = parse_external_timeseries(path)
            self.assertEqual(times, [0.0, 0.1])
            self.assertEqual(series, {"G1": [1.0, 3.0], "G2": [2.0, 4.0]})


class TabTimeHistoryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def setUp(self):
        self.tab = TabTimeHistory()
        self.tab.set_source_provider(
            gauges_1d=lambda: ((1.5, "G1"), (3.0, "G2")),
            probes_2d=lambda: (ProbePoint2D("P1", 2.0, 0.5),),
            probes_3d=lambda: (ProbePoint("Q1", 1.0, 2.0, 3.0),),
            case_dir=lambda dim: "",
            p_atm=lambda dim: 101325.0,
        )
        self.tab.refresh_catalog()

    def test_layout_defaults(self):
        self.assertEqual(
            [
                self.tab.workspace_tabs.tabText(index)
                for index in range(self.tab.workspace_tabs.count())
            ],
            ["Gauges", "Add Data", "Appearance"],
        )
        self.assertIs(self.tab._plot_splitter.widget(0), self.tab.workspace_tabs)
        self.assertIs(self.tab._plot_splitter.widget(1), self.tab.canvas)
        self.assertEqual(self.tab.chk_3d.text(), "3D")
        self.assertEqual(self.tab.chk_2d.text(), "2D")
        self.assertEqual(self.tab.chk_1d.text(), "1D")
        self.assertEqual(self.tab.chk_regions.text(), "Regions")
        self.assertTrue(self.tab.chk_3d.isChecked())
        self.assertTrue(self.tab.chk_2d.isChecked())
        self.assertTrue(self.tab.chk_1d.isChecked())
        self.assertFalse(self.tab.chk_regions.isChecked())
        self.assertEqual(self.tab.btn_add.text(), "Add")
        self.assertEqual(self.tab.btn_clear.text(), "Clear")
        self.assertTrue(self.tab.chk_pressure.isChecked())
        self.assertFalse(self.tab.chk_impulse.isChecked())
        headers = [
            self.tab.tbl_gauges.horizontalHeaderItem(i).text()
            for i in range(self.tab.tbl_gauges.columnCount())
        ]
        self.assertEqual(headers, ["ID", "X", "Y", "Z", "Label"])
        all_text = " ".join(
            widget.text() for widget in self.tab.findChildren(type(self.tab.btn_add))
        )
        self.assertNotIn("VIPER", all_text.upper())

    def test_each_table_column_can_be_resized(self):
        header = self.tab.tbl_gauges.horizontalHeader()
        for column in range(self.tab.tbl_gauges.columnCount()):
            self.assertEqual(
                header.sectionResizeMode(column), header.Interactive
            )
        before = header.sectionSize(2)
        header.resizeSection(2, before + 25)
        self.assertEqual(header.sectionSize(2), before + 25)

    def test_workspace_tab_switch_keeps_graph_geometry(self):
        self.tab.resize(1685, 900)
        self.tab.show()
        self.app.processEvents()
        initial = self.tab.canvas.geometry()
        for index in range(self.tab.workspace_tabs.count()):
            self.tab.workspace_tabs.setCurrentIndex(index)
            self.app.processEvents()
            self.assertEqual(self.tab.canvas.geometry(), initial)

    def test_regions_checkbox_sits_below_3d(self):
        self.tab.show()
        self.tab.resize(450, 700)
        self.app.processEvents()
        self.assertGreater(self.tab.chk_regions.y(), self.tab.chk_3d.y())
        self.assertLessEqual(abs(self.tab.chk_regions.x() - self.tab.chk_3d.x()), 8)
        self.assertGreaterEqual(
            self.tab.chk_regions.width(), self.tab.chk_regions.sizeHint().width()
        )

    def test_wrap_legend_name_breaks_long_labels(self):
        self.assertEqual(wrap_legend_name("G1", 20), "G1")
        wrapped = wrap_legend_name("VeryLongGaugeNameWithoutSpaces", 12)
        self.assertIn("\n", wrapped)
        self.assertLessEqual(max(len(part) for part in wrapped.split("\n")), 12)

    def test_catalog_and_dimension_filters(self):
        self.assertEqual(self.tab.tbl_gauges.rowCount(), 4)
        self.tab.chk_1d.setChecked(False)
        self.tab.chk_2d.setChecked(False)
        self.app.processEvents()
        self.assertEqual(self.tab.tbl_gauges.rowCount(), 1)
        self.assertEqual(self.tab.tbl_gauges.item(0, 0).text(), "3D-1")
        self.tab.chk_regions.setChecked(True)
        self.app.processEvents()
        self.assertEqual(self.tab.tbl_gauges.rowCount(), 1)

    def test_add_and_clear_series(self):
        self.tab.tbl_gauges.selectRow(0)
        self.tab.add_selected()
        self.assertEqual(self.tab.added_keys(), [("1d", 0)])
        self.assertTrue(self.tab.has_series())
        self.tab.clear_series()
        self.assertEqual(self.tab.added_keys(), [])
        self.assertFalse(self.tab.has_series())
        self.assertEqual(self.tab.tbl_gauges.rowCount(), 4)

    def test_add_with_legend_on_large_canvas_does_not_crash(self):
        self.tab.show()
        self.tab.resize(1685, 1060)
        self.app.processEvents()
        self.tab.tbl_gauges.selectRow(0)
        self.tab.add_selected()
        self.tab._redraw_plot()
        self.app.processEvents()
        pix = self.tab.canvas.pixmap()
        self.assertIsNotNone(pix)
        self.assertFalse(pix.isNull())
        legend = self.tab.canvas.axes.get_legend()
        self.assertIsNotNone(legend)

    def test_synthetic_gauges1d_file_plots_overpressure(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            with open(os.path.join(fo, "p"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.001 201325\n")
            self.tab.set_source_provider(case_dir=lambda dim: td if dim == "1d" else "")
            self.tab.tbl_gauges.selectRow(0)
            self.tab.add_selected()
            self.tab._redraw_plot()
            lines = self.tab.canvas.axes.lines
            self.assertEqual(len(lines), 1)
            xdata = list(lines[0].get_xdata())
            ydata = list(lines[0].get_ydata())
            self.assertEqual(xdata, [0.0, 0.001])
            self.assertAlmostEqual(ydata[0], 0.0)
            self.assertAlmostEqual(ydata[1], 100000.0)
            self.assertEqual(self.tab.canvas.axes.get_ylabel(), "Overpressure (Pa)")
            self.assertEqual(self.tab.canvas.axes.get_xlabel(), "Time (s)")
            legend = self.tab.canvas.axes.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual([text.get_text() for text in legend.get_texts()], ["G1"])
            xlim = self.tab.canvas.axes.get_xlim()
            ylim = self.tab.canvas.axes.get_ylim()
            self.assertLessEqual(xlim[0], 0.0)
            self.assertGreater(xlim[1], 0.001)
            self.assertLessEqual(ylim[0], 0.0)
            self.assertGreater(ylim[1], 100000.0)

    def test_begin_run_hides_existing_samples_and_shows_only_new_rows(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            path = os.path.join(fo, "p")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.001 201325\n")
            self.tab.tbl_gauges.selectRow(0)
            self.tab.add_selected()
            self.tab.begin_run("1D", td)
            self.tab._redraw_plot()
            line = self.tab.canvas.axes.lines[0]
            self.assertEqual(list(line.get_xdata()), [])
            self.assertEqual(list(line.get_ydata()), [])

            with open(path, "a", encoding="utf-8") as handle:
                handle.write("0.002 301325\n")
            self.tab.note_sim_progress("1D", 0.002)
            self.tab._redraw_plot()
            line = self.tab.canvas.axes.lines[0]
            self.assertEqual(list(line.get_xdata()), [0.002])
            self.assertEqual(list(line.get_ydata()), [200000.0])

    def test_load_completed_case_discovers_gauges_and_extrema(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            with open(os.path.join(fo, "p"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.1 201325\n")
            loaded = self.tab.load_completed_case(td)
            self.assertEqual(loaded, 1)
            self.assertEqual(len(self.tab._imported), 1)
            series = self.tab._imported[0]
            self.assertEqual(series.dim, "1d")
            self.assertEqual(series.field, "p")
            self.assertEqual(series.values, [0.0, 100000.0])
            self.assertEqual(series.extrema(), (0.0, 0.0, 100000.0, 0.1))
            self.assertEqual(self.tab.tbl_imported.rowCount(), 1)

    def test_csv_import_plots_on_right_axis_with_live_series(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "external.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Time,Imported Gauge\n")
                handle.write("0.0,0\n")
                handle.write("0.1,25\n")
            self.assertEqual(self.tab.load_external_file(path), 1)
            self.tab.tbl_imported.selectRow(0)
            self.tab._plot_import_selection("right")
            self.tab.tbl_gauges.selectRow(0)
            self.tab.add_selected()
            self.tab._redraw_plot()
            self.assertIsNotNone(self.tab._right_axes)
            self.assertEqual(len(self.tab._right_axes.lines), 1)
            self.assertEqual(
                list(self.tab._right_axes.lines[0].get_ydata()), [0.0, 25.0]
            )
            self.assertEqual(len(self.tab.canvas.axes.lines), 1)

    def test_impulse_import_uses_dashed_line(self):
        self.tab._imported.append(
            ImportedSeries(
                uid="impulse",
                source="CompletedCase",
                dim="1d",
                field="impulse",
                label="G1",
                times=[0.0, 0.1],
                values=[0.0, 10.0],
                plotted=True,
                color="#123456",
            )
        )
        self.tab._redraw_plot()
        self.assertEqual(self.tab.canvas.axes.lines[0].get_linestyle(), "--")
        self.assertEqual(self.tab.canvas.axes.lines[0].get_color(), "#123456")

    def test_appearance_controls_apply_to_all_series(self):
        self.tab._imported.append(
            ImportedSeries(
                uid="external",
                source="CSV",
                dim="1d",
                field="p",
                label="Loaded",
                times=[0.0, 0.1],
                values=[1.0, 2.0],
                plotted=True,
                color="#123456",
            )
        )
        self.tab.edit_plot_title.setText("Combined Histories")
        self.tab.cmb_legend_position.setCurrentText("Top")
        controls = self.tab._appearance_controls["x"]
        controls["title"].setText("Elapsed Time (s)")
        controls["minimum"].setText("-1")
        controls["maximum"].setText("2")
        self.tab._redraw_plot()
        axes = self.tab.canvas.axes
        self.assertEqual(axes.get_title(), "Combined Histories")
        self.assertEqual(axes.get_xlabel(), "Elapsed Time (s)")
        self.assertEqual(tuple(round(value, 6) for value in axes.get_xlim()), (-1.0, 2.0))
        self.assertIsNotNone(axes.get_legend())

    def test_added_gauges_have_distinct_colors_and_right_legend(self):
        self.tab.tbl_gauges.selectRow(0)
        self.tab.add_selected()
        self.tab.tbl_gauges.selectRow(1)
        self.tab.add_selected()
        self.tab._redraw_plot()
        lines = self.tab.canvas.axes.lines
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(lines[0].get_color(), lines[1].get_color())
        legend = self.tab.canvas.axes.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual([text.get_text() for text in legend.get_texts()], ["G1", "G2"])
        self.tab.show()
        self.tab.resize(1100, 700)
        self.app.processEvents()
        self.tab.canvas.draw_idle()
        ax_box = self.tab.canvas.axes.get_window_extent()
        legend_box = legend.get_window_extent()
        self.assertGreaterEqual(legend_box.x0, ax_box.x1 - 20)

    def test_canvas_fills_available_area_on_resize(self):
        self.tab.resize(1100, 700)
        self.tab.show()
        self.app.processEvents()
        canvas = self.tab.canvas
        self.assertGreaterEqual(canvas.width(), 500)
        self.assertGreaterEqual(canvas.height(), 400)
        canvas.draw_idle()
        pix = canvas.pixmap()
        self.assertIsNotNone(pix)
        self.assertFalse(pix.isNull())
        self.assertGreaterEqual(pix.width(), int(canvas.contentsRect().width() * 0.85))
        self.assertGreaterEqual(pix.height(), int(canvas.contentsRect().height() * 0.85))
        self.tab.resize(800, 500)
        self.app.processEvents()
        canvas.draw_idle()
        pix = canvas.pixmap()
        self.assertGreaterEqual(pix.width(), int(canvas.contentsRect().width() * 0.85))
        self.assertGreaterEqual(pix.height(), int(canvas.contentsRect().height() * 0.85))

    def test_long_gauge_name_wraps_inside_canvas(self):
        long_name = "VeryLongGaugeNameThatExceedsTheLegendColumnWidth"
        self.tab.set_source_provider(gauges_1d=lambda: ((1.5, long_name),))
        self.tab.refresh_catalog()
        self.tab.show()
        self.tab.resize(1100, 700)
        self.app.processEvents()
        self.tab.tbl_gauges.selectRow(0)
        self.tab.add_selected()
        self.tab._redraw_plot()
        legend = self.tab.canvas.axes.get_legend()
        self.assertIsNotNone(legend)
        texts = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(len(texts), 1)
        self.assertIn("\n", texts[0])
        self.tab.canvas.draw_idle()
        legend_box = legend.get_window_extent()
        fig_w = self.tab.canvas.figure.get_figwidth() * self.tab.canvas.figure.dpi
        self.assertLessEqual(legend_box.x1, fig_w + 8)

    def test_impulse_matches_pressure_color_and_is_dashed(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            with open(os.path.join(fo, "p"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.001 201325\n")
            with open(os.path.join(fo, "impulse"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 0.0\n")
                handle.write("0.001 12.5\n")
            self.tab.set_source_provider(case_dir=lambda dim: td if dim == "1d" else "")
            self.tab.chk_impulse.setChecked(True)
            self.tab.tbl_gauges.selectRow(0)
            self.tab.add_selected()
            self.tab._redraw_plot()
            lines = self.tab.canvas.axes.lines
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0].get_color(), lines[1].get_color())
            self.assertEqual(lines[0].get_linestyle(), "-")
            self.assertEqual(lines[1].get_linestyle(), "--")
            legend = self.tab.canvas.axes.get_legend()
            self.assertEqual([text.get_text() for text in legend.get_texts()], ["G1"])

    def test_live_sim_time_extends_x_axis_above_data(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            with open(os.path.join(fo, "p"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.001 201325\n")
            self.tab.set_source_provider(case_dir=lambda dim: td if dim == "1d" else "")
            self.tab.tbl_gauges.selectRow(0)
            self.tab.add_selected()
            self.tab._redraw_plot()
            before = self.tab.canvas.axes.get_xlim()[1]
            self.tab.note_sim_progress("1D", 0.05)
            self.tab._redraw_plot()
            after = self.tab.canvas.axes.get_xlim()[1]
            self.assertGreater(after, 0.05)
            self.assertGreater(after, before)
            _lo, hi = padded_axis_limits(0.0, 0.05)
            self.assertAlmostEqual(after, hi)


    def test_missing_gauge_column_does_not_plot_mismatched_xy(self):
        with tempfile.TemporaryDirectory() as td:
            fo = os.path.join(td, "postProcessing", "gauges1d", "0")
            os.makedirs(fo)
            with open(os.path.join(fo, "p"), "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1.5 0 0)\n")
                handle.write("0.0 101325\n")
            self.tab.set_source_provider(case_dir=lambda dim: td if dim == "1d" else "")
            self.tab.tbl_gauges.selectRow(1)
            self.tab.add_selected()
            self.tab._redraw_plot()
            lines = self.tab.canvas.axes.lines
            self.assertEqual(len(lines), 1)
            self.assertEqual(list(lines[0].get_xdata()), [])
            self.assertEqual(list(lines[0].get_ydata()), [])


class ProbeHistoryIncompleteTests(unittest.TestCase):
    def test_parse_skips_incomplete_trailing_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "p")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("# Probe 0 (1 0 0)\n")
                handle.write("0.0 101325\n")
                handle.write("0.002 201")
            _locs, times, columns = parse_probe_history(path)
            self.assertEqual(times, [0.0])
            self.assertEqual(columns[0], [101325.0])


class TabTimeHistoryAppWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def test_main_window_uses_real_tab_and_refreshes_from_1d_gauges(self):
        from main_new import BlastFoamApp

        win = BlastFoamApp()
        try:
            self.assertIsInstance(win.tab_time_history, TabTimeHistory)
            win.tab_1d.set_gauge_locations(((4.0, "LiveG"),))
            win.tab_time_history.refresh_catalog()
            ids = [
                win.tab_time_history.tbl_gauges.item(row, 0).text()
                for row in range(win.tab_time_history.tbl_gauges.rowCount())
            ]
            self.assertIn("1D-1", ids)
            labels = [
                win.tab_time_history.tbl_gauges.item(row, 4).text()
                for row in range(win.tab_time_history.tbl_gauges.rowCount())
            ]
            self.assertIn("LiveG", labels)
            self.assertEqual(win._time_history_case_dir("1d"), "")
        finally:
            win.close()
