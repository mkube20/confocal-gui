"""Qt main window: tabs Move, Laser, Count Rate, Alignment, XY Scan, Z Scan.

Run with:  python run_confocal_gui.py          (real hardware via pyscan)
           python run_confocal_gui.py --sim    (simulated microscope)
"""

import csv
import json
import os
import queue
import re
import threading
import time
import traceback

import numpy as np
import matplotlib

try:  # matplotlib >= 3.5
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
except (ImportError, ValueError):  # older matplotlib (PyQt5 only)
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import analysis, saving, scanning  # noqa: E402
from .alignment import (MARKER_LABELS, AlignmentSession, PiezoFrame, layout_offsets,  # noqa: E402
                        markers_from_bl)
from .config import AXES  # noqa: E402
from .hardware import Positioner, counter_time, make_counter  # noqa: E402

Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal
Qt = QtCore.Qt
YES = QtWidgets.QMessageBox.StandardButton.Yes
MONO = QtGui.QFont("Consolas", 9)


def fmt3(v, nd=3):
    return "(" + ", ".join(f"{x:.{nd}f}" for x in v) + ")"


# ----------------------------------------------------------------------------
# Hardware worker: one thread does all instrument I/O, jobs run in order.
# ----------------------------------------------------------------------------
class Worker(QtCore.QObject):
    finished = Signal(object)   # (callback, result)
    failed = Signal(object)     # (callback, message)
    point = Signal(object)      # live scan updates
    stage_progress = Signal(object)   # stage position while homing

    def __init__(self):
        super().__init__()
        self.jobs = queue.Queue()
        self.busy = False
        self.stop_event = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, fn, on_done=None, on_error=None):
        self.jobs.put((fn, on_done, on_error))

    def clear(self):
        try:
            while True:
                self.jobs.get_nowait()
        except queue.Empty:
            pass

    @property
    def idle(self):
        return not self.busy and self.jobs.empty()

    def _loop(self):
        while True:
            fn, on_done, on_error = self.jobs.get()
            self.busy = True
            try:
                result = fn()
                self.finished.emit((on_done, result))
            except Exception as e:  # report to the GUI thread
                traceback.print_exc()
                self.failed.emit((on_error, f"{type(e).__name__}: {e}"))
            finally:
                self.busy = False


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
class Canvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(4, 3), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setMinimumSize(200, 150)


class ImageCanvas(Canvas):
    """Scan image (u right, v up) with a contrast bar on the right. The bar spans the
    measured count range; drag its top / bottom handle to set the brightest / darkest
    displayed level (works during and after a scan). Double-click the bar, or use
    auto_contrast(), to go back to automatic (full range)."""

    def __init__(self, title, on_double_click=None):
        super().__init__()
        self.fig.clf()
        # fixed margins (tight_layout can't handle the bar)
        if hasattr(self.fig, "set_layout_engine"):      # matplotlib >= 3.6
            self.fig.set_layout_engine(None)
        else:
            self.fig.set_tight_layout(False)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[22, 1], wspace=0.06,
                                   left=0.1, right=0.86, bottom=0.11, top=0.93)
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.cax = self.fig.add_subplot(gs[0, 1])
        self.title = title
        self.labels = saving.axis_labels("piezo")
        self.on_double_click = on_double_click
        self.drag = None
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("motion_notify_event", self._motion)
        self.mpl_connect("button_release_event", self._release)
        self.clear()

    def clear(self):
        self.ax.clear()
        self.cax.clear()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])
        self.cax.set_xticks([])
        self.cax.yaxis.tick_right()
        self.cax.yaxis.set_label_position("right")
        self.cax.set_ylabel("counts/s  (drag the cyan lines)")
        self.im = self.bar = None
        self.manual = None                 # (low, high) set by dragging; None = automatic
        self.data_range = (0.0, 1.0)
        self.draw_idle()

    # ---- contrast
    def display_clim(self):
        """Colour limits in use: the dragged ones, or the full data range."""
        return self.manual if self.manual else self.data_range

    def auto_contrast(self):
        self.manual = None
        self._apply()
        self.draw_idle()

    def _apply(self):
        lo, hi = self.display_clim()
        if self.im is not None:
            self.im.set_clim(lo, hi)
        dlo, dhi = self.data_range
        bot, top = min(dlo, lo), max(dhi, hi)
        if top <= bot:
            top = bot + 1
        vals = np.linspace(bot, top, 256)[:, None]
        if self.bar is None:
            self.bar = self.cax.imshow(vals, aspect="auto", origin="lower", cmap="inferno",
                                       extent=[0, 1, bot, top], interpolation="nearest")
            self.lo_line = self.cax.axhline(lo, color="cyan", lw=3)
            self.hi_line = self.cax.axhline(hi, color="cyan", lw=3)
        else:
            self.bar.set_data(vals)
            self.bar.set_extent([0, 1, bot, top])
        self.bar.set_clim(lo, hi)
        self.cax.set_ylim(bot, top)
        self.lo_line.set_ydata([lo, lo])
        self.hi_line.set_ydata([hi, hi])

    def _bar_value(self, event):
        return float(self.cax.transData.inverted().transform((event.x, event.y))[1])

    def _press(self, event):
        if event.inaxes is self.ax:
            if event.dblclick and self.on_double_click and self.im is not None:
                self.on_double_click(event.xdata, event.ydata)
            return
        if event.inaxes is not self.cax or self.bar is None:
            return
        if event.dblclick:
            self.auto_contrast()
            return
        lo, hi = self.display_clim()
        px = lambda val: self.cax.transData.transform((0.5, val))[1]
        self.drag = "lo" if abs(px(lo) - event.y) < abs(px(hi) - event.y) else "hi"
        self._motion(event)

    def _motion(self, event):
        if not self.drag or self.bar is None:
            return
        lo, hi = self.display_clim()
        bot, top = self.cax.get_ylim()
        val = min(max(self._bar_value(event), bot), top)
        gap = (top - bot) * 0.01
        if self.drag == "lo":
            lo = min(val, hi - gap)
        else:
            hi = max(val, lo + gap)
        self.manual = (lo, hi)
        self._apply()
        self.draw_idle()

    def _release(self, event):
        self.drag = None

    # ---- data
    def show_result(self, res, best=None, fit_center=None):
        du = (res.u[1] - res.u[0]) / 2 if len(res.u) > 1 else 0.05
        dv = (res.v[1] - res.v[0]) / 2 if len(res.v) > 1 else 0.05
        data = np.ma.masked_invalid(res.counts)
        if self.im is None or self.im.get_array().shape != data.shape:
            manual = self.manual
            self.clear()
            self.manual = manual
            self.im = self.ax.imshow(data, origin="lower", cmap="inferno", aspect="equal",
                                     interpolation="nearest",
                                     extent=[res.u[0] - du, res.u[-1] + du, res.v[0] - dv, res.v[-1] + dv])
            (self.best_mark,) = self.ax.plot([], [], "c+", ms=14, mew=2)
            (self.fit_mark,) = self.ax.plot([], [], "x", color="deepskyblue", ms=10, mew=2)
        self.im.set_data(data)
        if data.count():
            lo, hi = float(data.min()), float(data.max())
            self.data_range = (lo, hi if hi > lo else lo + 1)
        self._apply()
        self.best_mark.set_data(([best[0]], [best[1]]) if best else ([], []))
        self.fit_mark.set_data(([fit_center[0]], [fit_center[1]]) if fit_center else ([], []))
        self.draw_idle()


class ZCanvas(Canvas):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.ax.clear()
        self.ax.set_title("Z sweep")
        self.ax.set_xlabel("piezo z (um)")
        self.ax.set_ylabel("counts/s")
        self.draw_idle()

    def show_result(self, res, best_index=None):
        self.ax.clear()
        self.best = saving.draw_zsweep(self.ax, res, best_index)
        self.ax.set_title("Z sweep")
        self.draw_idle()


class SweepCanvas(Canvas):
    """Count rate vs laser current (% of max)."""

    def __init__(self):
        super().__init__()
        self.clear()

    def _fresh(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

    def clear(self):
        self._fresh()
        empty = scanning.PowerSweepResult(np.array([]), 0.0)
        saving.draw_power_sweep(self.ax, empty)
        self.draw_idle()

    def show_result(self, res):
        self._fresh()
        saving.draw_power_sweep(self.ax, res)
        b = res.best()
        if b:
            self.ax.plot(b[1], b[2], "c*", ms=12)
        self.draw_idle()


class CountCanvas(Canvas):
    def __init__(self):
        super().__init__()
        self.lines = [self.ax.plot([], [], lw=1.2, label=n)[0] for n in ("detector 1", "detector 2", "sum")]
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("counts/s")
        self.ax.legend(loc="upper left", fontsize=8)

    def update_data(self, t, series, window):
        if not t:
            for ln in self.lines:
                ln.set_data([], [])
            self.draw_idle()
            return
        t = np.asarray(t)
        keep = t >= t[-1] - window if window else np.ones(len(t), bool)
        lo, hi = np.inf, -np.inf
        for ln, ys in zip(self.lines, series):
            y = np.asarray(ys)[keep]
            ln.set_data(t[keep], y)
            lo, hi = min(lo, y.min()), max(hi, y.max())
        self.ax.set_xlim(t[keep][0], max(t[keep][-1], t[keep][0] + 1))
        pad = 0.05 * (hi - lo) + 1
        self.ax.set_ylim(max(0, lo - pad), hi + pad)
        self.draw_idle()


# ----------------------------------------------------------------------------
# small widget helpers
# ----------------------------------------------------------------------------
class Spin(QtWidgets.QDoubleSpinBox):
    """Number box that emits ``entered`` when Enter is pressed (after taking the typed value)."""
    entered = Signal()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.entered.emit()


def enter_clicks(btn, *spins):
    """Pressing Enter in any of ``spins`` is the same as clicking ``btn`` (nothing if it's disabled)."""
    for s in spins:
        s.entered.connect(btn.click)


def spin(value, lo, hi, step, decimals=2, suffix=" um"):
    s = Spin()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


def colored(b, color):
    b.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; padding: 5px; }}"
                    "QPushButton:disabled { background-color: #d0d0d0; color: #888; }")
    return b


def button(text, slot, tip=None):
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(lambda _=False: slot())
    if tip:
        b.setToolTip(tip)
    return b


def label(text="", wrap=True):
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def hsplit(*widgets, sizes=None):
    s = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
    s.setChildrenCollapsible(False)
    for w in widgets:
        s.addWidget(w)
    if sizes:
        s.setSizes(sizes)
    return s


def scrollable(widget):
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def boxed(title, layout):
    g = QtWidgets.QGroupBox(title)
    g.setLayout(layout)
    return g


def parse_xyz(text):
    """Three numbers from typed or pasted text, e.g. "2.53, 1.18, 1.713" or "[2.53 1.18 1.713]"."""
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    return np.array([float(n) for n in nums]) if len(nums) == 3 else None


class JogPanel(QtWidgets.QWidget):
    """Position readout + stage (coarse) and piezo (fine) jog buttons."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        cfg = win.cfg
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stage_lab = label(wrap=False)
        self.piezo_lab = label(wrap=False)
        for lab in (self.stage_lab, self.piezo_lab):
            lab.setFont(MONO)
            lay.addWidget(lab)
        self.buttons = []
        stage = self._group("Stage (coarse)", "stage",
                            spin(cfg["stage_xy_step_um"], cfg["stage_xy_min_step_um"], 1000, 10, 0),
                            spin(cfg["stage_z_step_um"], cfg["stage_z_min_step_um"], 200, 1, 0))
        piezo = self._group("Piezo (fine)", "piezo",
                            spin(cfg["piezo_xy_step_um"], cfg["piezo_min_step_um"], 10, 0.1, 1),
                            spin(cfg["piezo_z_step_um"], cfg["piezo_min_step_um"], 10, 0.1, 1))
        rc = button("Recenter piezo", win.recenter_piezo,
                    "Move the stage so the piezo is back mid-range, without moving the spot")
        self.buttons.append(rc)
        piezo.layout().addWidget(rc, 5, 0, 1, 4)
        # positions are shown and typed in the GUI frame (x / y as on the buttons)
        self.stage_goto = self._goto_row(stage.layout(), 6, "Go to (mm)", lambda: win.user(win.pos.stage),
                                         lambda t: win.goto_stage(win.from_user(t)))
        self.piezo_goto = self._goto_row(piezo.layout(), 6, "Go to (um)", lambda: win.user(win.pos.piezo),
                                         lambda t: win.goto_piezo(win.from_user(t)))
        lay.addWidget(hsplit(stage, piezo))
        lay.addStretch(1)
        self.setMaximumWidth(760)

    def _group(self, title, kind, xy_spin, z_spin):
        gl = QtWidgets.QGridLayout()
        gl.addWidget(QtWidgets.QLabel("XY step"), 0, 0)
        gl.addWidget(xy_spin, 0, 1, 1, 3)
        gl.addWidget(QtWidgets.QLabel("Z step"), 1, 0)
        gl.addWidget(z_spin, 1, 1, 1, 3)
        cfg = self.win.cfg
        # Labels are fixed (right = +x, up = +y); which stage/piezo axis each
        # button actually moves is set by screen_right / screen_up in the config.
        rx, rs = AXES[cfg["screen_right"][0]], cfg["screen_right"][1]
        ux, us = AXES[cfg["screen_up"][0]], cfg["screen_up"][1]
        for text, r, c, axis, sgn, sp in (("\u25b2 +y", 2, 1, ux, us, xy_spin), ("\u25bc \u2212y", 4, 1, ux, -us, xy_spin),
                                          ("\u25c0 \u2212x", 3, 0, rx, -rs, xy_spin), ("\u25b6 +x", 3, 2, rx, rs, xy_spin),
                                          ("+z", 2, 3, 2, 1, z_spin), ("\u2212z", 4, 3, 2, -1, z_spin)):
            b = QtWidgets.QPushButton(text)
            b.setMinimumSize(40, 30)
            b.clicked.connect(lambda _=False, a=axis, s=sgn, sp=sp: self.win.jog(kind, a, s * sp.value()))
            gl.addWidget(b, r, c)
            self.buttons.append(b)
        return boxed(title, gl)

    def _goto_row(self, grid, row, title, current, action):
        """x, y, z boxes + Go. Empty boxes keep that axis where it is; pasting all
        three numbers (e.g. "2.53, 1.18, 1.713") into any box fills all three."""
        edits = []
        box = QtWidgets.QHBoxLayout()
        box.addWidget(QtWidgets.QLabel(title))
        for ax in "xyz":
            e = QtWidgets.QLineEdit()
            e.setPlaceholderText(ax)
            e.setMinimumWidth(50)
            box.addWidget(e, 1)
            edits.append(e)
        go = QtWidgets.QPushButton("Go")
        go.setMinimumWidth(40)
        box.addWidget(go)

        def submit():
            target = np.array(current(), float)
            for k, e in enumerate(edits):
                text = e.text().strip()
                if not text:
                    continue
                three = parse_xyz(text)
                if three is not None:          # pasted "x, y, z" into one box
                    for kk in range(3):
                        edits[kk].setText(f"{three[kk]:g}")
                    target = three
                    break
                try:
                    target[k] = float(text)
                except ValueError:
                    self.win.error(f"'{text}' is not a number ({'xyz'[k]})")
                    return
            action(target)
        go.clicked.connect(lambda _=False: submit())
        for e in edits:
            e.returnPressed.connect(submit)
        grid.addLayout(box, row, 0, 1, 4)
        self.buttons += edits + [go]
        return edits

    def show_stage(self, xyz):
        self.stage_lab.setText(f"Stage (mm) {fmt3(self.win.user(xyz), 4)}")

    def update_position(self, pos):
        self.stage_lab.setText(f"Stage (mm) {fmt3(self.win.user(pos.stage), 4)}")
        frac = pos.piezo_fraction()
        edge = np.any((frac < 0.05) | (frac > 0.95))
        self.piezo_lab.setText(f"Piezo (um) {fmt3(self.win.user(pos.piezo), 2)}" +
                               ("  near end of range" if edge else ""))
        self.piezo_lab.setStyleSheet("color: #c62828;" if edge else "")

    def set_enabled(self, on):
        for b in self.buttons:
            b.setEnabled(on)

class SaveDialog(QtWidgets.QDialog):
    """Pick a folder and type a folder name; results go in <folder>/<name>/."""

    def __init__(self, parent, title, folder, name):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        form = QtWidgets.QFormLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(folder)
        row.addWidget(self.folder, 1)
        row.addWidget(button("Browse...", self._browse))
        form.addRow("Save in", row)
        self.name = QtWidgets.QLineEdit(name)
        form.addRow("Folder name", self.name)
        self.preview = label()
        self.preview.setStyleSheet("color: gray")
        form.addRow(self.preview)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save |
                                         QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)
        self.folder.textChanged.connect(self._update)
        self.name.textChanged.connect(self._update)
        self._update()
        self.name.setFocus()
        self.name.selectAll()

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save in", self.folder.text())
        if d:
            self.folder.setText(d)

    def _update(self):
        n = self.name.text() or "NAME"
        self.preview.setText(f"Creates {os.path.join(self.folder.text(), n)} with {n}_image.png, "
                             f"{n}_data.csv and {n}_results.txt")

    def values(self):
        return self.folder.text().strip(), self.name.text()


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
MODE_TEXT = {
    "stage": "<span style='color:#c62828'>Larger than the piezo range (19.8 um): the stepper moves "
             "during the scan (about every 19 um along each row).</span>",
}
MODE_PLAIN = {"piezo": "piezo only (stepper fixed)",
              "stage": "stepper + piezo (area larger than the piezo range)"}
INFO_STYLE = "font-size: 11pt; padding: 6px; background: #e3f2fd; color: black; border: 1px solid #90caf9;"


class MainWindow(QtWidgets.QMainWindow):
    TAB_NAMES = ["Move", "Laser", "Count Rate", "Alignment", "XY Scan", "Z Scan"]

    def __init__(self, hw, cfg, lasers=None):
        super().__init__()
        self.hw, self.cfg = hw, cfg
        self.lasers = lasers or {}
        self.setWindowTitle("Confocal control" + ("  [SIMULATOR]" if hw.name == "simulator" else ""))
        self.pos = Positioner(hw, cfg)
        self.read = make_counter(hw, cfg)
        # GUI x / y are the hardware axes moved by the +x / +y buttons
        self.perm = [AXES[cfg["screen_right"][0]], AXES[cfg["screen_up"][0]], 2]
        self.worker = Worker()
        self.worker.finished.connect(self._job_done)
        self.worker.failed.connect(self._job_failed)
        self.worker.point.connect(self._scan_point)
        self.worker.stage_progress.connect(self._show_stage_progress)

        self.laser_on = None          # (name, pct) of the laser that is on
        self.count_t, self.count_data = [], ([], [], [])
        self.count_t0 = None
        self.count_running = True

        self.session = AlignmentSession(cfg["accept_tolerance_um"])
        self.saved_alignment = self._load_alignment()   # last completed alignment (file)
        self.frame = None             # marker frame once aligned
        self.target = None
        self.xy = None                # last XY scan: {"res", "frame", "fit", "target", "time"}
        self.z = None                 # last Z scan:  {"res", "best", "avg", "time"}
        self.save_dir = cfg["default_save_folder"] or os.path.expanduser("~")
        self.scanning = False
        self.scan_dirty = None
        self.scan_t0 = 0
        self.poll_count = 0
        self.pages = {}

        self.tabs = QtWidgets.QTabWidget()
        for builder, name in zip((self._build_move_tab, self._build_laser_tab, self._build_count_tab,
                                  self._build_align_tab, self._build_xy_tab, self._build_z_tab), self.TAB_NAMES):
            self.tabs.addTab(builder(), name)
        self.setCentralWidget(self.tabs)
        self.statusBar()

        self._update_alignment_ui()
        self._update_position()
        self._update_sweep_estimate()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start(200)
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.timeout.connect(self._redraw)
        self.redraw_timer.start(300)

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1150, int(screen.width() * 0.9)), min(720, int(screen.height() * 0.85)))
        QtCore.QTimer.singleShot(300, self._check_homed)

    # ------------------------------------------------- GUI coordinate frame
    def user(self, xyz):
        """Hardware (x, y, z) -> GUI (x, y, z), where GUI x/y follow the arrow buttons."""
        return np.asarray(xyz, float)[self.perm]

    def from_user(self, xyz):
        out = np.empty(3)
        out[self.perm] = np.asarray(xyz, float)
        return out

    # ---------------------------------------------------------------- Move tab
    def _build_move_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.move_jog = JogPanel(self)
        lay.addWidget(self.move_jog)
        self.home_btn = button("Home stage", self.home_stage,
                               "Send the stepper stage to its home switches (z first, away from the objective)")
        self.home_btn.setMaximumWidth(200)
        lay.addWidget(self.home_btn)
        lay.addStretch(1)
        return w

    # --------------------------------------------------------------- Laser tab
    def _build_laser_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()
        self.laser_name = next(iter(self.lasers), "")
        form.addRow("Laser", QtWidgets.QLabel(self.laser_name or "none configured"))
        row = QtWidgets.QHBoxLayout()
        self.laser_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.laser_slider.setRange(0, 100)
        self.laser_spin = spin(50, 0, 100, 1, 0, " %")
        self.laser_slider.setValue(50)
        self.laser_slider.valueChanged.connect(lambda v: self.laser_spin.setValue(v))
        self.laser_spin.valueChanged.connect(lambda v: self.laser_slider.setValue(int(round(v))))
        row.addWidget(self.laser_slider, 1)
        row.addWidget(self.laser_spin)
        form.addRow("Current", row)
        max_a = self.lasers[self.laser_name].spec.get("max_current_A") if self.laser_name else None
        form.addRow("", label("% of the diode controller's max current"
                              + (f" (100 % = {max_a:g} A)" if max_a else "") + ", not % of optical power."))
        row = QtWidgets.QHBoxLayout()
        on = button("Turn on / set current", self.set_laser)
        enter_clicks(on, self.laser_spin)
        colored(on, "#2e7d32")
        off = button("Off", self.laser_off)
        off.setStyleSheet("font-weight: bold; padding: 6px;")
        row.addWidget(on)
        row.addWidget(off)
        self.laser_buttons = [on, off, self.laser_slider, self.laser_spin]
        form.addRow(row)
        self.laser_status = QtWidgets.QLabel("Reading laser state...")
        self.laser_status.setStyleSheet("font-size: 13pt; font-weight: bold;")
        form.addRow(self.laser_status)
        g = boxed("Laser", form)
        lay.addWidget(g)
        lay.addWidget(self._build_sweep_box())
        lay.addStretch(1)
        if self.laser_name:
            self.run(self.lasers[self.laser_name].read_state, self._show_laser_state,
                     lambda m: self.laser_status.setText("Laser state unknown (not read)"))
        else:
            self.laser_status.setText("")
        return hsplit(scrollable(w), self.pages["laser"].canvas, sizes=[420, 640])

    def _build_sweep_box(self):
        cfg = self.cfg
        p = type("Page", (), {})()
        box = QtWidgets.QVBoxLayout()
        box.addWidget(label("Total count rate vs laser current (% of max) at the current position; the laser "
                            "goes back to its previous setting afterwards."))
        form = QtWidgets.QFormLayout()
        p.start_pct = spin(cfg["sweep_start_pct"], 0, 100, 1, 1, " %")
        p.stop_pct = spin(cfg["sweep_stop_pct"], 0, 100, 1, 1, " %")
        p.step_pct = spin(cfg["sweep_step_pct"], 0.1, 100, 1, 1, " %")
        p.time = spin(cfg["sweep_time_s"], 0.1, 60, 0.1, 1, " s")
        form.addRow("Start", p.start_pct)
        form.addRow("Stop", p.stop_pct)
        form.addRow("Step", p.step_pct)
        form.addRow("Counting per setting", p.time)
        for sp in (p.start_pct, p.stop_pct, p.step_pct, p.time):
            sp.valueChanged.connect(self._update_sweep_estimate)
        box.addLayout(form)
        p.est = label()
        p.est.setStyleSheet(INFO_STYLE)
        box.addWidget(p.est)
        p.start = button("Start current sweep", self._start_sweep)
        p.start.setMinimumHeight(36)
        colored(p.start, "#2e7d32")
        enter_clicks(p.start, p.start_pct, p.stop_pct, p.step_pct, p.time)
        box.addWidget(p.start)
        p.progress = QtWidgets.QProgressBar()
        box.addWidget(p.progress)
        p.stop = button("Stop", self.stop)
        colored(p.stop, "#c62828")
        p.stop.setEnabled(False)
        box.addWidget(p.stop)
        p.best = label()
        p.best.setStyleSheet("font-size: 11pt; font-weight: bold;")
        box.addWidget(p.best)
        p.save = button("Save results...", self._save_sweep,
                        "Saves NAME_image.png, NAME_data.csv and NAME_results.txt in a folder called NAME")
        p.save.setEnabled(False)
        box.addWidget(p.save)
        p.canvas = SweepCanvas()
        self.pages["laser"] = p
        if not self.laser_name:
            p.start.setEnabled(False)
        g = QtWidgets.QGroupBox("Laser current sweep")
        g.setLayout(box)
        return g

    def _sweep_values(self):
        p = self.pages["laser"]
        return scanning.sweep_values(p.start_pct.value(), p.stop_pct.value(), p.step_pct.value())

    def _update_sweep_estimate(self, *_):
        p = self.pages.get("laser")
        if p is None:
            return
        vals = self._sweep_values()
        per = self.cfg["sweep_settle_s"] + counter_time(self.cfg, p.time.value()) + 0.05
        p.est.setText(f"{len(vals)} settings: {vals[0]:g} % to {vals[-1]:g} % of max current<br>"
                      f"Estimated time: <b>{scanning.format_duration(len(vals) * per)}</b>")

    def _start_sweep(self):
        if not self.laser_name or self.scanning:
            return
        p = self.pages["laser"]
        vals = self._sweep_values()
        laser, prev = self.lasers[self.laser_name], self.laser_on
        t, settle, stop = p.time.value(), self.cfg["sweep_settle_s"], self.worker.stop_event
        p.canvas.clear()
        self.status("Running laser current sweep...")
        self._begin_scan("laser", len(vals))

        def on_point(res, k):
            self.worker.point.emit(("laser", res, k + 1, res.counts0[k], res.counts1[k]))

        def job():
            try:
                return scanning.run_power_sweep(laser, self.read, vals, t, settle, self.hw.wait, stop,
                                                self.pos, on_point)
            finally:                                   # back to the previous laser setting
                if prev:
                    laser.set_power(prev[1])
                else:
                    laser.off()

        def done(res):
            self.set_busy(False)
            self._show_laser_state((prev is not None, prev[1] if prev else None))
            self.sweep = {"res": res, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "restored": prev}
            self.scan_dirty = ("laser", res)
            self._redraw()
            p.save.setEnabled(True)
            self.status("Current sweep done" if res.completed else "Current sweep stopped", 5000)

        def failed(msg):
            self.set_busy(False)
            self.error(f"Current sweep failed: {msg}")

        self.run(job, done, failed)

    def _save_sweep(self):
        d = getattr(self, "sweep", None)
        if not d:
            return
        res = d["res"]

        def write(files, name):
            saving.save_power_sweep_png(res, files["image.png"], f"Laser current sweep {name}")
            saving.save_power_sweep_csv(res, files["data.csv"])
            ok = ~np.isnan(res.counts)
            max_a = self.lasers[self.laser_name].spec.get("max_current_A")
            lines = [f"Laser current sweep {name}   {d['time']}", "",
                     f"Laser: {self.laser_name}",
                     f"Current (% of max{f', 100 % = {max_a:g} A' if max_a else ''}): {res.pct[0]:g} % to "
                     f"{res.pct[-1]:g} %, step {abs(res.pct[1] - res.pct[0]) if len(res.pct) > 1 else 0:g} % "
                     f"({len(res.pct)} settings)",
                     f"Counting {res.time_s:g} s per setting, after {self.cfg['sweep_settle_s']:g} s settle",
                     f"Points measured: {int(ok.sum())} of {len(res.pct)}" + ("" if res.completed else " (stopped)"),
                     ""]
            if res.stage_mm is not None:
                lines += self._pos_lines("Measured at:", res.stage_mm, res.piezo_um)
            b = res.best()
            if b:
                lines += ["", f"Highest count rate: {b[2] / 1000:,.2f} kcps at {b[1]:g} % current"]
            restored = d["restored"]
            lines += ["", "Laser afterwards: " + (f"back to {restored[1]:.0f} %" if restored else "off"),
                      "Coordinates are in the GUI frame (x/y as on the arrow buttons)."]
            with open(files["results.txt"], "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        self._save_to_folder("Save current sweep", time.strftime("current_sweep_%Y%m%d_%H%M"), write)

    def _show_laser_state(self, state):
        """state = (on, pct) read from the laser, or None if it can't be read."""
        if state is None:
            self.laser_status.setText("Laser state unknown")
            return
        on, pct = state
        if pct is not None:
            self.laser_spin.setValue(round(pct))
        self.laser_on = (self.laser_name, pct if pct is not None else 0) if on else None
        self.laser_status.setText(f"{self.laser_name} ON" + (f" at {pct:.0f} % current" if pct is not None else "")
                                  if on else f"{self.laser_name} off")

    def set_laser(self):
        if not self.laser_name:
            return
        name, pct = self.laser_name, self.laser_spin.value()
        self.run(lambda: self.lasers[name].set_power(pct),
                 lambda _: (self._show_laser_state((pct > 0, pct)),
                            self.status(f"{name} set to {pct:.0f} % current", 3000)))

    def laser_off(self):
        if not self.laser_name:
            return
        name = self.laser_name
        self.run(self.lasers[name].off,
                 lambda _: (self._show_laser_state((False, None)), self.status(f"{name} off", 3000)))

    def _laser_text(self):
        return (f"{self.laser_on[0]} at {self.laser_on[1]:.0f} % of max current" if self.laser_on
                else "not set from this program")

    # ---------------------------------------------------------- Count Rate tab
    def _build_count_tab(self):
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        self.count_labels = []
        for name in ("Detector 1", "Detector 2", "Sum"):
            ll.addWidget(QtWidgets.QLabel(name))
            lab = QtWidgets.QLabel("-")
            lab.setStyleSheet("font-size: 20pt; font-weight: bold;")
            ll.addWidget(lab)
            self.count_labels.append(lab)
        self.count_btn = button("Pause", self._toggle_counts)
        ll.addWidget(self.count_btn)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Show last"))
        self.count_window = QtWidgets.QComboBox()
        self.count_window.addItems(["30 s", "60 s", "5 min", "all"])
        self.count_window.setCurrentIndex(1)
        row.addWidget(self.count_window, 1)
        ll.addLayout(row)
        ll.addWidget(button("Clear", self._clear_counts))
        ll.addWidget(button("Save data + plot...", self._save_counts))
        ll.addStretch(1)
        self.count_canvas = CountCanvas()
        return hsplit(left, self.count_canvas, sizes=[220, 800])

    def _toggle_counts(self):
        self.count_running = not self.count_running
        self.count_btn.setText("Pause" if self.count_running else "Resume")

    def _clear_counts(self):
        self.count_t, self.count_data, self.count_t0 = [], ([], [], []), None
        self.count_canvas.update_data([], [], None)

    def _add_counts(self, c0, c1):
        now = time.time()
        if self.count_t0 is None:
            self.count_t0 = now
        self.count_t.append(now - self.count_t0)
        for lst, v in zip(self.count_data, (c0, c1, c0 + c1)):
            lst.append(v)
        for lab, v in zip(self.count_labels, (c0, c1, c0 + c1)):
            lab.setText(f"{v / 1000:,.1f} kcps")

    def _save_counts(self):
        if not self.count_t:
            return
        t, (c0, c1, cs) = list(self.count_t), [list(x) for x in self.count_data]

        def write(files, name):
            with open(files["data.csv"], "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["time_s", "detector1_cps", "detector2_cps", "sum_cps"])
                wr.writerows(zip(t, c0, c1, cs))
            win = {0: 30, 1: 60, 2: 300, 3: None}[self.count_window.currentIndex()]
            self.count_canvas.update_data(t, (c0, c1, cs), None)      # whole record in the image
            self.count_canvas.ax.set_title(f"Count rate {name}")
            self.count_canvas.fig.savefig(files["image.png"], dpi=150)
            self.count_canvas.ax.set_title("")
            self.count_canvas.update_data(self.count_t, self.count_data, win)
            lines = [f"Count rate {name}   saved {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                     f"Recording: {len(t)} readings over {t[-1] - t[0]:.1f} s"]
            for lab, ys in (("Detector 1", c0), ("Detector 2", c1), ("Sum", cs)):
                a = np.asarray(ys, float)
                lines.append(f"  {lab}: mean {a.mean() / 1000:,.2f} kcps, std {a.std() / 1000:.2f}, "
                             f"min {a.min() / 1000:,.2f}, max {a.max() / 1000:,.2f}")
            lines += ["", f"Laser: {self._laser_text()}", ""]
            lines += self._pos_lines("Position when saved:", self.pos.stage, self.pos.piezo)
            lines += ["", "Coordinates are in the GUI frame (x/y as on the arrow buttons)."]
            with open(files["results.txt"], "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        self._save_to_folder("Save count rate", time.strftime("count_rate_%Y%m%d_%H%M"), write)

    # ----------------------------------------------------------- Alignment tab
    def _build_align_tab(self):
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        self.align_instr = label()
        self.align_instr.setStyleSheet("font-size: 11pt; padding: 6px; background: #fff8e1; color: black;"
                                       " border: 1px solid #ffcc80;")
        ll.addWidget(self.align_instr)
        self.save_marker_btn = button("Save marker position", self._save_manual_marker)
        self.accept_btn = button("Accept", self._accept_marker)
        self.adjust_btn = button("Save adjusted position", self._save_adjusted_marker)
        for b, color in ((self.save_marker_btn, "#1565c0"), (self.accept_btn, "#2e7d32"), (self.adjust_btn, "#1565c0")):
            b.setMinimumHeight(38)
            colored(b, color)
        ll.addWidget(self.save_marker_btn)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.accept_btn)
        row.addWidget(self.adjust_btn)
        ll.addLayout(row)
        self.regoto_btn = button("Return to predicted position", self._goto_prediction)
        ll.addWidget(self.regoto_btn)
        form = QtWidgets.QFormLayout()
        self.marker_status = {}
        for k in ("TR", "TL", "BL", "BR"):
            self.marker_status[k] = QtWidgets.QLabel("")
            form.addRow(MARKER_LABELS[k].capitalize() + ":", self.marker_status[k])
        ll.addWidget(boxed("Markers", form))
        self.layout_btn = button("New grid: I'm on its BL marker (use saved layout)", self._new_grid_from_bl,
                                 "The other three markers are placed relative to where you are now, using the "
                                 "layout from the last alignment; then accept / adjust each one")
        self.recheck_btn = button("Re-check the saved markers", self._recheck_markers,
                                  "Go to the marker positions from the last alignment and accept / adjust them")
        ll.addWidget(self.layout_btn)
        ll.addWidget(self.recheck_btn)
        ll.addWidget(button("Restart alignment", self._restart_alignment))
        form = QtWidgets.QFormLayout()
        ox, oy = self.cfg["scan_start_offset_um"]
        self.goto_u = spin(ox, -4000, 4000, 5, 2)
        self.goto_v = spin(oy, -4000, 4000, 5, 2)
        form.addRow("x from BL marker", self.goto_u)
        form.addRow("y from BL marker", self.goto_v)
        self.goto_marker_btn = button("Go there (in focus)", self._goto_marker_pos,
                                      "Along the marker edges (x: BL\u2192BR, y: BL\u2192TL), not the stage axes")
        form.addRow(self.goto_marker_btn)
        enter_clicks(self.goto_marker_btn, self.goto_u, self.goto_v)
        self.goto_box = boxed("Go to a position on the sample", form)
        ll.addWidget(self.goto_box)
        ll.addStretch(1)
        self.align_jog = JogPanel(self)
        return hsplit(left, self.align_jog, sizes=[380, 620])

    # ------------------------------------------------------------- XY Scan tab
    def _scan_controls(self, p, ll, kind):
        p.est = label()
        p.est.setStyleSheet(INFO_STYLE)
        ll.addWidget(p.est)
        p.start = button("Start XY scan" if kind == "xy" else "Start z scan",
                         self._start_xy if kind == "xy" else self._start_z)
        p.start.setMinimumHeight(38)
        colored(p.start, "#2e7d32")
        ll.addWidget(p.start)
        p.progress = QtWidgets.QProgressBar()
        ll.addWidget(p.progress)
        p.stop = button("Stop", self.stop)
        colored(p.stop, "#c62828")
        p.stop.setEnabled(False)
        ll.addWidget(p.stop)
        p.best = label()
        p.best.setStyleSheet("font-size: 11pt; font-weight: bold;")
        ll.addWidget(p.best)
        p.goto_opt = button("Go to optimum", self._goto_xy_optimum if kind == "xy" else self._goto_z_optimum,
                            "After a scan the stage and piezo go back to where the scan started; "
                            "this moves to the optimum (piezo first)")
        p.goto_opt.setMinimumHeight(34)
        colored(p.goto_opt, "#1565c0")
        p.goto_opt.setEnabled(False)
        ll.addWidget(p.goto_opt)
        p.where = label()
        p.where.setFont(MONO)
        ll.addWidget(p.where)
        if kind == "xy":
            p.auto = button("Auto contrast", lambda: self.pages["xy"].canvas.auto_contrast(),
                            "Reset the colour scale to the full range (or double-click the colour bar)")
            ll.addWidget(p.auto)
        p.save = button("Save results...", lambda k=kind: self._save_scan(k),
                        "Saves NAME_image.png, NAME_data.csv and NAME_results.txt in a folder called NAME")
        p.save.setEnabled(False)
        ll.addWidget(p.save)
        ll.addStretch(1)

    def _build_xy_tab(self):
        cfg = self.cfg
        p = type("Page", (), {})()
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        p.axes = label()
        ll.addWidget(p.axes)
        p.use_markers = QtWidgets.QCheckBox("Adjust z for the sample tilt (from the markers)")
        p.use_markers.setEnabled(False)
        p.use_markers.toggled.connect(self._update_estimates)
        ll.addWidget(p.use_markers)
        ll.addWidget(label("Centred on the current position (within the piezo's reach). Scans like the "
                           "notebook: piezo x / y, hardware x is the fast axis, each line starts from the same "
                           "side, one PicoHarp reading per point after the dwell."))
        form = QtWidgets.QFormLayout()
        p.rx = spin(cfg["coarse_range_um"][0], 0.1, 3000, 1.0, 2)
        p.ry = spin(cfg["coarse_range_um"][1], 0.1, 3000, 1.0, 2)
        p.step = spin(cfg["coarse_step_um"], cfg["min_scan_step_um"], 100, 0.1, 2)
        form.addRow("X range", p.rx)
        form.addRow("Y range", p.ry)
        form.addRow("Step", p.step)
        for s in (p.rx, p.ry, p.step):
            s.valueChanged.connect(self._update_estimates)
        ll.addLayout(form)
        self._scan_controls(p, ll, "xy")
        enter_clicks(p.start, p.rx, p.ry, p.step)
        p.canvas = ImageCanvas("XY scan", self._goto_uv)
        self.pages["xy"] = p
        return hsplit(scrollable(left), p.canvas, sizes=[360, 700])

    # -------------------------------------------------------------- Z Scan tab
    def _build_z_tab(self):
        cfg = self.cfg
        p = type("Page", (), {})()
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(label("Sweeps z at fixed x, y (centred on the current z, or starting from it), then "
                           "goes back to where it started. 'Go to optimum' moves to the best z and measures "
                           "the count rate there."))
        form = QtWidgets.QFormLayout()
        p.range = spin(cfg["z_range_um"], 0.1, 100, 0.5, 2)
        p.step = spin(cfg["z_step_um"], cfg["min_scan_step_um"], 10, 0.1, 2)
        p.zmode = QtWidgets.QComboBox()
        p.zmode.addItems(["Centred on the current z", "From the current z, going up (+z)",
                          "From the current z, going down (\u2212z)"])
        p.zmode.currentIndexChanged.connect(self._update_estimates)
        form.addRow("Sweep", p.zmode)
        form.addRow("Z range", p.range)
        form.addRow("Step", p.step)
        for s in (p.range, p.step):
            s.valueChanged.connect(self._update_estimates)
        ll.addLayout(form)
        self._scan_controls(p, ll, "z")
        enter_clicks(p.start, p.range, p.step)
        p.zmode.setToolTip("Centred: current z - range/2 to current z + range/2 (shifted to fit the piezo). "
                           "From the current z: starts exactly where you are.")
        p.canvas = ZCanvas()
        self.pages["z"] = p
        return hsplit(scrollable(left), p.canvas, sizes=[360, 700])
    # -------------------------------------------------------------- utilities
    def status(self, msg, ms=5000):
        self.statusBar().showMessage(msg, ms)

    def error(self, msg):
        QtWidgets.QMessageBox.warning(self, "Error", msg)

    def ask(self, title, text):
        return QtWidgets.QMessageBox.question(self, title, text) == YES

    def run(self, fn, on_done=None, on_error=None):
        self.worker.submit(fn, on_done, on_error)

    def _job_done(self, payload):
        cb, result = payload
        self._update_position()
        if cb:
            cb(result)

    def _job_failed(self, payload):
        cb, msg = payload
        self._update_position()
        (cb or self.error)(msg)

    @property
    def dwell(self):
        return self.cfg["dwell_s"]

    def set_busy(self, busy):
        self.scanning = busy
        self.home_btn.setEnabled(not busy)
        self.move_jog.set_enabled(not busy)
        self.align_jog.set_enabled(not busy)
        for b in (self.save_marker_btn, self.accept_btn, self.adjust_btn, self.regoto_btn,
                  self.layout_btn, self.recheck_btn):
            b.setEnabled(not busy)
        self.goto_box.setEnabled(not busy and self.frame is not None)
        data = {"xy": self.xy, "z": self.z, "laser": getattr(self, "sweep", None)}
        for kind, p in self.pages.items():
            p.start.setEnabled(not busy and (kind != "laser" or bool(self.laser_name)))
            p.stop.setEnabled(busy)
            p.save.setEnabled(not busy and data.get(kind) is not None)
            if hasattr(p, "goto_opt"):
                p.goto_opt.setEnabled(not busy and bool(data.get(kind)) and data[kind].get("target") is not None)
        for b in getattr(self, "laser_buttons", []):
            b.setEnabled(not busy)

    def stop(self):
        self.worker.stop_event.set()
        self.worker.clear()
        self.status("Stopping...")

    # --------------------------------------------------------------- position
    def _update_position(self):
        self.move_jog.update_position(self.pos)
        self.align_jog.update_position(self.pos)
        self._remember_stage()
        if not self.scanning:
            self._update_estimates()

    # ------------------------------------------------------------ stage homing
    def _last_stage_path(self):
        return self.cfg["last_stage_file"] or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".last_stage_position.json")

    def _load_last_stage(self):
        """Last good stage position saved by a previous session: (xyz, time) or None."""
        try:
            with open(self._last_stage_path(), encoding="utf-8") as f:
                last = json.load(f)
            return np.array(last["stage_mm"], float), last.get("time", "?")
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _remember_stage(self):
        """Keep the last good stage position (in memory and on disk) to offer after homing."""
        if not hasattr(self, "_last_good"):
            self._last_good = self._load_last_stage()
        if not self.pos.stage_homed() or np.any(self.pos.stage <= 1e-6):
            return                        # only remember positions with every coordinate > 0
        xyz = self.pos.stage.copy()
        if self._last_good is not None and np.array_equal(self._last_good[0], xyz):
            return
        self._last_good = (xyz, time.strftime("%Y-%m-%d %H:%M"))
        try:
            with open(self._last_stage_path(), "w", encoding="utf-8") as f:
                json.dump({"stage_mm": [float(v) for v in xyz], "time": self._last_good[1]}, f)
        except OSError:
            pass

    def _check_homed(self):
        """Start-up check of the stepper. Reads a fresh position first (the very first
        reading after connecting can be stale), then decides."""
        if getattr(self, "_closed", False):
            return

        def job():
            self.pos.refresh()
            self.hw.wait(0.3)
            return self.pos.refresh()[0]

        self.run(job, self._startup_stage_check,
                 lambda m: self.error(f"Could not read the stepper position ({m}).\n\n"
                                      "Restart the stepper motors (switch the controller off and on again), "
                                      "then restart the GUI."))

    def _startup_stage_check(self, stage):
        if getattr(self, "_closed", False):
            return
        stage = np.asarray(stage, float)
        print(f"Start-up stepper reading (hardware x, y, z): {stage.tolist()} mm")
        shown = fmt3(self.user(stage), 4)
        restart = ("Restart the stepper motors (switch the controller off and on again), then restart the GUI.")
        tiny = np.abs(stage) < 5e-5                   # shows as 0.0000 / -0.0000
        neg_zero = tiny & np.signbit(stage)          # "-0": controller needs a restart (homing won't work)
        zero = tiny & ~np.signbit(stage)
        if np.any(neg_zero):
            self.error(f"The stepper reads {shown} mm: a coordinate reads -0, so the stepper controller is not "
                       f"in a usable state (homing will not work).\n\n{restart}")
            return
        if np.all(zero):
            # freshly powered controller: home it, then go back to where we were
            self.status("The stepper reads (0, 0, 0): homing it, then returning to the last position...", 0)
            self.home_stage(confirm=False, auto_return=True)
            return
        if np.any(zero):
            self.error(f"The stepper reads {shown} mm: one or more coordinates are exactly 0, so the stepper "
                       f"controller is not in a usable state.\n\n{restart}")
            return
        if np.any(stage < 0) or not self.pos.stage_homed():
            if self.ask("Stage not homed",
                        f"The stage reads {shown} mm. Negative coordinates mean the stepper controller lost its "
                        "home position, so the stage can't be moved reliably.\n\n"
                        "Home it now? Z goes down to 0 first (away from the objective), then X and Y go to 0, and "
                        "then it offers to return to the last position. Watch the coordinates on the Move tab; "
                        f"if it doesn't work within {self.cfg['stage_home_timeout_s']:g} s it stops."):
                self.home_stage(confirm=False)

    def home_stage(self, confirm=True, auto_return=False):
        if self.scanning:
            return
        if confirm and not self.ask("Home stage", "Send the stage to its home position? Z goes down to 0 "
                                                  "first (away from the objective), then X and Y go to 0."):
            return
        before = getattr(self, "_last_good", None)   # where to offer to return afterwards
        self.set_busy(True)
        self.status("Homing stage (z, then x, then y)...", 0)

        def done(_):
            self.set_busy(False)
            self.status("Stage homed", 5000)
            if auto_return:
                self._return_to_last(before)
            else:
                self._offer_last_stage(before)

        def failed(msg):
            self.set_busy(False)
            self.statusBar().clearMessage()
            self.error(f"Homing failed.\n\n{msg}")

        progress = lambda xyz: self.worker.stage_progress.emit(np.array(xyz, float))
        self.run(lambda: self.pos.home_stage(progress), done, failed)

    def _show_stage_progress(self, xyz):
        self.move_jog.show_stage(xyz)
        self.align_jog.show_stage(xyz)

    def _usable_last(self, last):
        """The last position if it is worth going back to (all coordinates > 0, in range)."""
        if last is None:
            return None
        xyz, when = last
        lim = self.pos.stage_lim
        if np.any(xyz <= 1e-6) or np.any(xyz > lim[:, 1]) or np.allclose(xyz, self.pos.stage):
            return None
        return xyz, when

    def _return_to_last(self, last):
        last = self._usable_last(last)
        if last is None:
            return
        xyz, when = last
        self.run(lambda: self.pos.move_stage(xyz),
                 lambda _: self.status(f"Back at the last position {fmt3(self.user(xyz), 4)} mm (from {when})", 8000))

    def _offer_last_stage(self, last):
        last = self._usable_last(last)
        if last is None:
            return
        xyz, when = last
        if self.ask("Return to last position",
                    f"Move the stage back to its last position {fmt3(self.user(xyz), 4)} mm (from {when})?"):
            self.goto_stage(xyz)

    def _poll(self):
        """Read the detectors continuously (any tab), and refresh positions now and then."""
        if self.scanning or not self.worker.idle:
            return
        self.poll_count += 1
        refresh = self.tabs.currentIndex() in (0, 3) and self.poll_count % 8 == 0
        read = self.count_running
        if not (read or refresh):
            return

        def job():
            if refresh:
                self.pos.refresh()
            return self.hw.get_count_rates() if read else None

        def done(r):
            if r is not None:
                self._add_counts(*r)
        self.run(job, done, lambda m: self.status("Count read failed: " + m))

    # ----------------------------------------------------------------- moving
    def jog(self, kind, axis, delta):
        if self.scanning:
            return
        if kind == "stage":
            self.run(lambda: self.pos.jog_stage(axis, delta))
        else:
            self.run(lambda: self.pos.jog_piezo(axis, delta),
                     on_error=lambda m: self.error(m + "\nUse 'Recenter piezo' or the stage."))

    def goto_stage(self, xyz_mm):
        """Move the stage to ``xyz_mm`` (hardware order)."""
        if self.scanning:
            return
        self.run(lambda: self.pos.move_stage(xyz_mm),
                 lambda _: self.status(f"Moved stage to {fmt3(self.user(self.pos.stage), 4)} mm", 3000))

    def goto_piezo(self, xyz_um):
        """Move the piezo to ``xyz_um`` (hardware order)."""
        if self.scanning:
            return
        self.run(lambda: self.pos.move_piezo_axes(xyz_um),
                 lambda _: self.status(f"Moved piezo to {fmt3(self.user(self.pos.piezo), 2)} um", 3000))

    def recenter_piezo(self):
        self.run(self.pos.recenter_piezo,
                 lambda n: self.status("Piezo centered" if n else "Piezo already centered", 3000))

    def move_to(self, target, then=None, what="target", piezo_first=False):
        target = np.asarray(target, float)
        try:
            self.pos.check_reachable(target)
        except Exception as e:
            self.error(str(e))
            return

        def done(r):
            self.status(f"Moved to {what}", 3000)
            if then:
                then(r)
        mover = self.pos.move_piezo_first if piezo_first else self.pos.move_abs
        self.run(lambda: mover(target), done)

    def _goto_uv(self, u, v):
        """Double-click on the XY image: move there (piezo only if possible)."""
        if self.xy is None or self.scanning:
            return
        frame = self.xy["frame"]
        dz = frame.to_grid(self.pos.abs)[2]
        if self.ask("Move", f"Move to x = {u:.2f}, y = {v:.2f} um?"):
            self.move_to(frame.to_abs(u, v, dz), what=f"clicked point ({u:.2f}, {v:.2f})", piezo_first=True)

    # -------------------------------------------------------------- alignment
    def _update_alignment_ui(self):
        s = self.session
        name = s.current
        manual, verify = s.phase == "manual", s.phase == "verify"
        self.save_marker_btn.setVisible(manual)
        for b in (self.accept_btn, self.adjust_btn, self.regoto_btn):
            b.setVisible(verify)
        first = manual and s.manual_index == 0
        have_layout = self.saved_alignment is not None
        self.layout_btn.setVisible(first and have_layout)
        self.recheck_btn.setVisible(first and have_layout)
        if first and have_layout:
            self.align_instr.setText(
                "Move to the <b>BOTTOM LEFT</b> marker and focus with Z. Then either save it and align by hand, "
                f"or use the layout saved on {self.saved_alignment['time']} (same marker layout, any grid).")
            self.save_marker_btn.setText("Save bottom left marker (align by hand)")
        elif manual:
            self.align_instr.setText(f"Move to the <b>{MARKER_LABELS[name].upper()}</b> marker, focus with Z, "
                                     "then save it.")
            self.save_marker_btn.setText(f"Save {MARKER_LABELS[name]} marker")
        elif verify:
            self.align_instr.setText(f"Check the <b>{MARKER_LABELS[name].upper()}</b> marker "
                                     f"(predicted position). Accept it, or adjust and save.<br>"
                                     f"Accepted in a row: <b>{s.accepts_in_row} / 4</b>")
        else:
            self.align_instr.setText("<b>Alignment complete.</b> XY scans can now follow the marker edges.")
        for k in ("TR", "TL", "BL", "BR"):
            last = [h["action"] for h in s.history if h["marker"] == k]
            text = {"manual": "saved"}.get(last[-1], last[-1]) if last else ""
            if k == name:
                text = "\u25b6 current" + (f" (last: {text})" if text else "")
            self.marker_status[k].setText(text)
        self.goto_box.setEnabled(self.frame is not None and not self.scanning)
        p = self.pages.get("xy")
        if p is not None:
            p.use_markers.setEnabled(self.frame is not None)
            if self.frame is None:
                p.use_markers.setChecked(False)

    def _capture(self, then):
        def job():
            self.pos.refresh()
            return self.pos.abs, self.pos.stage.copy(), self.pos.piezo.copy()
        self.run(job, then)

    def _save_manual_marker(self):
        if self.session.phase != "manual":
            return

        def done(r):
            p, st, pz = r
            name = self.session.record_manual(p)
            self.status(f"Saved {MARKER_LABELS[name]} marker")
            self._update_alignment_ui()
            if self.session.phase == "verify":
                self._goto_prediction()
        self._capture(done)

    def _goto_prediction(self):
        if self.session.phase != "verify":
            return
        name = self.session.current
        self.target = self.session.prediction()
        try:
            self.pos.check_reachable(self.target)
        except Exception as e:
            self.error(f"Predicted {MARKER_LABELS[name]} position is unreachable: {e}")
            return

        def arrived(_):
            self.session.set_arrival(self.pos.abs)
            self.status(f"Moved to predicted {MARKER_LABELS[name]} marker", 3000)
            self._update_alignment_ui()

        target = self.target.copy()
        self.run(lambda: self.pos.move_abs(target), arrived)

    def _confirm_marker(self, adjusted_button):
        if self.session.phase != "verify":
            return

        def done(r):
            p, st, pz = r
            ref = self.session.arrival if self.session.arrival is not None else self.target
            dist = float(np.linalg.norm(p - ref))
            if not adjusted_button and dist > self.session.tol and not self.ask(
                    "Position changed", f"You moved {dist:.2f} um from the predicted position.\n"
                                        "Save this as an adjusted position?"):
                return
            name = self.session.current
            adjusted, dist = self.session.confirm(p)
            self.status(f"{MARKER_LABELS[name]} marker {'adjusted' if adjusted else 'accepted'}")
            self.target = None
            if self.session.done:
                self.frame = self.session.frame()
                self.pages["xy"].use_markers.setChecked(True)
                self._save_alignment(self.session.measured)
                self.status("Alignment complete (saved for next time)", 5000)
            self._update_alignment_ui()
            if not self.session.done:
                self._goto_prediction()
        self._capture(done)

    def _accept_marker(self):
        self._confirm_marker(adjusted_button=False)

    def _save_adjusted_marker(self):
        self._confirm_marker(adjusted_button=True)

    def _goto_marker_pos(self):
        """Move to (x, y) from the BL marker along the marker edges, in focus."""
        if self.frame is None or self.scanning:
            return
        u, v = self.goto_u.value(), self.goto_v.value()
        self.move_to(self.frame.to_abs(u, v), what=f"x = {u:g}, y = {v:g} um from the BL marker")

    # ---- saved alignment / layout
    def _alignment_path(self):
        return self.cfg["alignment_file"] or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".alignment.json")

    def _load_alignment(self):
        """Last completed alignment: {"markers", "offsets", "time"} or None."""
        try:
            with open(self._alignment_path(), encoding="utf-8") as f:
                d = json.load(f)
            markers = {k: np.array(d["markers_abs_um"][k], float) for k in ("TR", "TL", "BL", "BR")}
            return {"markers": markers, "offsets": layout_offsets(markers), "time": d.get("time", "?")}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_alignment(self, markers):
        markers = {k: np.asarray(markers[k], float) for k in ("TR", "TL", "BL", "BR")}
        stamp = time.strftime("%Y-%m-%d %H:%M")
        data = {"time": stamp,
                "markers_abs_um": {k: v.tolist() for k, v in markers.items()},
                "offsets_from_BL_um": {k: v.tolist() for k, v in layout_offsets(markers).items()},
                "note": "abs = 1000 * stage_mm + piezo_sign * piezo_um (hardware axis order)"}
        try:
            with open(self._alignment_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
        except OSError as e:
            self.status(f"Could not save the alignment: {e}")
        self.saved_alignment = {"markers": markers, "offsets": layout_offsets(markers), "time": stamp}

    def _start_from_markers(self, markers, start, note):
        """Go straight to accept / adjust with all four markers known."""
        self.session = AlignmentSession.from_markers(markers, self.cfg["accept_tolerance_um"], start, note)
        self.frame = self.target = None
        self._update_alignment_ui()
        self._update_estimates()
        self._goto_prediction()

    def _known_markers(self):
        """The most recent full set of markers: this session's, else the saved ones."""
        if len(self.session.measured) == 4:
            return self.session.measured
        return self.saved_alignment["markers"] if self.saved_alignment else None

    def _recheck_markers(self):
        markers = self._known_markers()
        if markers is None or self.scanning:
            return
        self._start_from_markers(markers, "BL", "previous")

    def _new_grid_from_bl(self):
        """The user is on the BL marker of a grid with the saved layout."""
        if self.saved_alignment is None or self.scanning:
            return
        offsets = self.saved_alignment["offsets"]

        def done(r):
            self._start_from_markers(markers_from_bl(r[0], offsets), "BR", "from layout")
        self._capture(done)

    def _ask_restart_choice(self):
        """'recheck', 'newgrid', 'full' or None (cancel)."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Restart alignment")
        box.setText("How do you want to restart the alignment?")
        role = QtWidgets.QMessageBox.ButtonRole
        choices = {}
        if self._known_markers() is not None:
            choices[box.addButton("Re-check the same markers", role.ActionRole)] = "recheck"
        if self.saved_alignment is not None:
            choices[box.addButton("New grid (I'm on its BL marker)", role.ActionRole)] = "newgrid"
        choices[box.addButton("Full restart (find BL, BR, TR by hand)", role.ActionRole)] = "full"
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        return choices.get(box.clickedButton())

    def _restart_alignment(self):
        if self.scanning:
            return
        choice = self._ask_restart_choice()
        if choice == "recheck":
            self._recheck_markers()
        elif choice == "newgrid":
            self._new_grid_from_bl()
        elif choice == "full":
            self.session = AlignmentSession(self.cfg["accept_tolerance_um"])
            self.frame = self.target = None
            self._update_alignment_ui()
            self._update_estimates()
            self.status("Alignment restarted")

    # ---------------------------------------------------------- time estimates
    def _timing_path(self):
        return self.cfg["scan_timing_file"] or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".scan_timing.json")

    def overhead(self, kind):
        """Seconds per point on top of the dwell for this scan type: measured on
        the last finished scans of that type, or the config default."""
        if not hasattr(self, "_overhead"):
            try:
                with open(self._timing_path(), encoding="utf-8") as f:
                    self._overhead = {k: float(v) for k, v in json.load(f).items()}
            except (OSError, ValueError, TypeError, AttributeError):
                self._overhead = {}
        if kind in self._overhead:
            return self._overhead[kind]
        if kind == "xy" and "coarse" in self._overhead:      # measured by the older version
            return self._overhead["coarse"]
        if kind == "xy_notebook":
            return self.cfg["xy_overhead_per_point_s"]
        return self.cfg["overhead_per_point_s"]

    def _learn_timing(self, kind, res, n_points):
        """Update the per-point overhead from a finished scan (half old, half new)."""
        if not res.completed or n_points < 10 or res.finished <= res.started:
            return
        self.overhead(kind)
        if kind == "xy_notebook":
            lines = len(res.v) if self.perm[0] == 0 else len(res.u)
            busy = res.finished - res.started - self.cfg["xy_start_settle_s"] - lines * res.dwell_s
            measured = max(0.0, busy / n_points - res.dwell_s)
        else:
            measured = max(0.0, (res.finished - res.started) / n_points - counter_time(self.cfg, res.dwell_s))
        old = self._overhead.get(kind)
        self._overhead[kind] = measured if old is None else 0.5 * (old + measured)
        try:
            with open(self._timing_path(), "w", encoding="utf-8") as f:
                json.dump(self._overhead, f)
        except OSError:
            pass
        self._update_estimates()

    # ------------------------------------------------------------- XY planning
    def scan_frame(self):
        """Frame the next XY scan uses: the piezo x / y axes as shown in the GUI, with z
        fixed, or (aligned and ticked) z following the tilt of the marker plane."""
        p = self.pages["xy"]
        plane = self.frame if self.frame is not None and p.use_markers.isChecked() else None
        return PiezoFrame(self.pos.stage, self.pos.sign, self.pos.abs[2], self.perm[0], self.perm[1],
                          plane=plane, at=self.pos.abs)

    def _xy_plan(self):
        """(frame, u values, v values, dz) for an XY scan centred on the current position."""
        p = self.pages["xy"]
        frame = self.scan_frame()
        u0, v0, dz = frame.to_grid(self.pos.abs)
        dz = 0.0
        u = scanning.centered_values(u0, p.rx.value(), p.step.value())
        v = scanning.centered_values(v0, p.ry.value(), p.step.value())
        if self._xy_piezo_only():
            u, v = scanning.fit_window_in_piezo(self.pos, frame, u, v, dz)
        return frame, u, v, dz

    def _z_values(self):
        """Absolute z values: centred on the current focus (shifted to fit the piezo if
        needed), or starting at the current z going up / down."""
        p = self.pages["z"]
        z0, rng, step = self.pos.abs[2], p.range.value(), p.step.value()
        mode = p.zmode.currentIndex()
        if mode:
            return scanning.directed_values(z0, rng, step, 1 if mode == 1 else -1)
        z = scanning.centered_values(z0, rng, step)
        return scanning.fit_z_in_piezo(self.pos, z) if self._z_piezo_only() else z

    def piezo_span(self, axes):
        lim = self.pos.piezo_lim
        return float(min(lim[k, 1] - lim[k, 0] for k in axes))

    def _xy_piezo_only(self):
        p = self.pages["xy"]
        span = self.piezo_span(self.perm[:2])
        return p.rx.value() <= span + 1e-9 and p.ry.value() <= span + 1e-9

    def _z_piezo_only(self):
        return self.pages["z"].range.value() <= self.piezo_span([2]) + 1e-9

    @staticmethod
    def _mode_text(piezo_only):
        return "" if piezo_only else "<br>" + MODE_TEXT["stage"]

    def _update_estimates(self, *_):
        if "xy" not in self.pages or "z" not in self.pages:
            return
        d = self.dwell
        p = self.pages["xy"]
        aligned = self.frame is not None and p.use_markers.isChecked()
        p.axes.setText("Axes: piezo x / y (as on the Move tab); z follows the tilt of the marker plane."
                       if aligned else
                       "Axes: piezo x / y (as on the Move tab); z stays fixed."
                       + ("" if self.frame is not None else " Align the markers to let z follow the sample tilt."))
        frame, u, v, dz = self._xy_plan()
        labels = saving.axis_labels(frame.kind)
        if p.canvas.im is None and p.canvas.labels != labels:
            p.canvas.labels = labels
            p.canvas.clear()
        n = len(u) * len(v)
        p.est.setText(f"{n} points ({len(u)} x {len(v)})<br>Estimated time: "
                      f"<b>{scanning.format_duration(self._xy_time(len(u), len(v)))}</b>"
                      f"{self._mode_text(self._xy_piezo_only())}")
        z = self.pages["z"]
        zv = self._z_values()
        warn = (z.zmode.currentIndex() and self._z_piezo_only() and not scanning.z_in_piezo(self.pos, zv))
        z.est.setText(f"{len(zv)} points<br>Estimated time: "
                      f"<b>{scanning.format_duration(len(zv) * (counter_time(self.cfg, d) + self.overhead('z')))}</b>"
                      f"{self._mode_text(self._z_piezo_only())}"
                      + ("<br><span style='color:#c62828'>Runs past the end of the piezo z range from here.</span>"
                         if warn else ""))

    def _xy_time(self, nu, nv):
        n, d = nu * nv, self.dwell
        if self._xy_piezo_only():          # notebook method: dwell per point, 2 x dwell per new line
            lines = nv if self.perm[0] == 0 else nu
            return (self.cfg["xy_start_settle_s"] + n * (d + self.overhead("xy_notebook")) + lines * d)
        return n * (counter_time(self.cfg, d) + self.overhead("xy"))

    def _where_text(self, title, stage, piezo):
        pad = "&nbsp;" * (len(title) + 1)
        return (f"{title} stage {fmt3(self.user(stage), 4)} mm<br>"
                f"{pad}piezo {fmt3(self.user(piezo), 2)} um")

    # ------------------------------------------------------------ running scans
    def _begin_scan(self, kind, total):
        self.worker.stop_event.clear()
        self.set_busy(True)
        self.scan_t0 = time.time()
        p = self.pages[kind]
        p.progress.setRange(0, total)
        p.progress.setValue(0)
        p.best.setText("")
        p.save.setEnabled(False)
        if hasattr(p, "goto_opt"):
            p.goto_opt.setEnabled(False)

    def _back_to_start(self, start):
        """(Worker thread) stepper, then piezo, back to where a scan started.
        Returns an error text, or None if it worked."""
        try:
            target = self.pos.to_abs(*start)
            self.pos.refresh()
            self.pos.move_stage(start[0])
            self.pos.refresh()
            # the stepper lands within ~1 um; take that out with the piezo when it can
            p = self.pos.piezo_for(target)
            self.pos.move_piezo_axes(p if np.all(self.pos._piezo_ok(p, 0.0)) else start[1])
            self.pos.refresh()
            return None
        except Exception as e:
            return str(e)

    def _scan_job(self, scan, start):
        """Job that runs ``scan()`` and then returns to ``start`` (also if it stops or fails)."""
        def job():
            try:
                res = scan()
            except Exception:
                self._back_to_start(start)
                raise
            return res, self._back_to_start(start)
        return job

    def _returned(self, err):
        if err:
            self.error(f"Could not move back to the start position of the scan: {err}")

    def _scan_point(self, payload):
        kind, res, n, c0, c1 = payload
        self.scan_dirty = (kind, res)
        p = self.pages[kind]
        p.progress.setValue(n)
        elapsed = time.time() - self.scan_t0
        left = elapsed / max(n, 1) * (p.progress.maximum() - n)
        p.progress.setFormat(f"%v / %m   ~{scanning.format_duration(left)} left")
        # The hardware thread is busy with the scan, so the Count Rate tab can't poll;
        # show the detector readings the scan just took instead (values travel with
        # the signal, so a busy screen can't show a later point twice).
        if self.count_running and c0 == c0 and c1 == c1:
            self._add_counts(float(c0), float(c1))

    def _redraw(self):
        if self.tabs.currentIndex() == 2:
            win = {0: 30, 1: 60, 2: 300, 3: None}[self.count_window.currentIndex()]
            self.count_canvas.update_data(self.count_t, self.count_data, win)
        if not self.scan_dirty:
            return
        kind, res = self.scan_dirty
        self.scan_dirty = None
        p = self.pages[kind]
        b = res.best()
        if kind == "laser":
            p.canvas.show_result(res)
            if b:
                p.best.setText(f"Highest so far: {b[2] / 1000:,.1f} kcps at {b[1]:g} % current")
            return
        if kind == "z":
            p.canvas.show_result(res)
            if b:
                p.best.setText(f"Best so far: {b[2] / 1000:,.1f} kcps at piezo z = {res.piezo_um[b[0], 2]:.3f} um")
            return
        p.canvas.show_result(res, (b[2], b[3]) if b else None)
        if b:
            j, i = b[0], b[1]
            p.best.setText(f"Best so far: {b[4] / 1000:,.1f} kcps at x = {b[2]:.2f}, y = {b[3]:.2f} um")

    def _start_xy(self):
        frame, u, v, dz = self._xy_plan()
        try:
            scanning.check_raster_reachable(self.pos, frame, u, v, dz)
        except Exception as e:
            self.error(str(e))
            return
        p = self.pages["xy"]
        p.canvas.labels = saving.axis_labels(frame.kind)
        p.canvas.clear()
        self.status("Running XY scan...")
        self._begin_scan("xy", len(u) * len(v))
        p.where.setText(self._where_text("Start:  ", self.pos.stage, self.pos.piezo))
        dwell, stop, piezo_only = self.dwell, self.worker.stop_event, self._xy_piezo_only()
        start = (self.pos.stage.copy(), self.pos.piezo.copy())

        def on_point(res, j, i):
            self.worker.point.emit(("xy", res, len(res.order), res.counts0[j, i], res.counts1[j, i]))

        def done(r):
            res, err = r
            self.set_busy(False)
            self._xy_done(res, frame)
            self._returned(err)

        def failed(msg):
            self.set_busy(False)
            self.error(f"XY scan failed: {msg}")

        if piezo_only:          # the notebook's scan
            scan = lambda: scanning.run_xy_notebook(self.pos, self.hw, frame, u, v, dz, dwell, stop, on_point,
                                                    self.cfg["counts_channel"], self.cfg["xy_start_settle_s"])
        else:                   # larger than the piezo range: the stepper has to move
            scan = lambda: scanning.run_raster("xy", self.pos, self.read, frame, u, v, dz, dwell, stop, on_point,
                                               piezo_only=False)
        self.run(self._scan_job(scan, start), done, failed)

    def _xy_done(self, res, frame):
        self._learn_timing("xy_notebook" if res.method == "notebook" else "xy", res, res.n_points)
        self.scan_dirty = ("xy", res)
        self._redraw()
        if not res.order:
            self.status("XY scan stopped")
            return
        j, i, u, v, c = res.best()
        # centroid of the 3 x 3 pixels around the brightest one (within half a step)
        fit = {"ok": True, "center": analysis.refine_peak_2d(res.u, res.v, res.counts)} \
            if res.completed else {"ok": False}
        target = fit["center"] if fit["ok"] else (u, v)
        self.xy = {"res": res, "frame": frame, "fit": fit, "target": target,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        p = self.pages["xy"]
        p.canvas.show_result(res, (u, v), fit["center"] if fit["ok"] else None)
        p.save.setEnabled(True)
        p.goto_opt.setEnabled(True)
        p.where.setText(self._where_text("Start:  ", res.start_stage, res.start_piezo))
        if not res.completed:
            self.status(f"XY scan stopped after {len(res.order)} of {res.n_points} points; back at the start "
                        "position", 8000)
            return
        self.status("XY scan done; back at the start position. 'Go to optimum' moves to the brightest spot.", 8000)

    def _goto_xy_optimum(self):
        if not self.xy or self.scanning:
            return
        res, frame, target = self.xy["res"], self.xy["frame"], self.xy["target"]
        p = self.pages["xy"]

        def moved(_):
            self.xy["final"] = (self.pos.stage.copy(), self.pos.piezo.copy())
            p.where.setText(self._where_text("Start:  ", res.start_stage, res.start_piezo)
                            + "<br>" + self._where_text("Optimum:", *self.xy["final"]))
        self.move_to(frame.to_abs(target[0], target[1], res.z_offset), then=moved,
                     what="the optimum (brightest point)", piezo_first=True)

    def _start_z(self):
        p = self.pages["z"]
        xyz = self.pos.abs.copy()
        z = self._z_values()
        try:
            for zz in (z[0], z[-1]):
                self.pos.check_reachable(np.r_[xyz[:2], zz])
        except Exception as e:
            self.error(str(e))
            return
        if p.zmode.currentIndex() and self._z_piezo_only() and not scanning.z_in_piezo(self.pos, z):
            lo, hi = self.pos.piezo_lim[2]
            self.error(f"Starting at the current z, the sweep would run past the end of the piezo z range "
                       f"({lo:g}\u2013{hi:g} um). Use a smaller range, the other direction, or 'Centred'.")
            return
        p.canvas.clear()
        self.status("Running z scan...")
        self._begin_scan("z", len(z))
        p.where.setText(self._where_text("Start:  ", self.pos.stage, self.pos.piezo))
        dwell, stop, piezo_only = self.dwell, self.worker.stop_event, self._z_piezo_only()
        start = (self.pos.stage.copy(), self.pos.piezo.copy())

        def on_point(res, k):
            self.worker.point.emit(("z", res, k + 1, res.counts0[k], res.counts1[k]))

        def done(r):
            res, err = r
            self.set_busy(False)
            self._z_done(res)
            self._returned(err)

        def failed(msg):
            self.set_busy(False)
            self.error(f"Z scan failed: {msg}")

        self.run(self._scan_job(lambda: scanning.run_z_sweep(self.pos, self.read, xyz[:2], z, dwell, stop,
                                                             on_point, piezo_only=piezo_only), start),
                 done, failed)

    def _z_done(self, res):
        self._learn_timing("z", res, len(res.z_abs))
        self.scan_dirty = ("z", res)
        self._redraw()
        if res.best() is None:
            self.status("Z scan stopped")
            return
        k, zbest, c = res.best()     # the measured best point (same as "Best so far")
        p = self.pages["z"]
        p.canvas.show_result(res, k)
        self.z = {"res": res, "best": k, "avg": None, "target": np.r_[res.xy_abs, zbest],
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        p.save.setEnabled(True)
        p.goto_opt.setEnabled(True)
        p.where.setText(self._where_text("Start:  ", res.start_stage, res.start_piezo))
        if not res.completed:
            self.status(f"Z scan stopped after {int(np.sum(~np.isnan(res.counts)))} points; back at the start "
                        "position", 8000)
            return
        self.status("Z scan done; back at the start position. 'Go to optimum' moves to the best z.", 8000)

    def _goto_z_optimum(self):
        """Move to the best z of the last z scan, then average the count rate there."""
        if not self.z or self.scanning:
            return
        res, k = self.z["res"], self.z["best"]
        c = res.counts[k]
        p = self.pages["z"]
        target = self.z["target"]
        avg_s, dwell = self.cfg["final_count_average_s"], self.dwell

        def job():
            self.pos.move_piezo_first(target)
            self.pos.refresh()
            return scanning.average_counts(self.read, avg_s, dwell)

        def done(r):
            self.set_busy(False)
            self.z["avg"] = r
            self.z["final"] = (self.pos.stage.copy(), self.pos.piezo.copy())
            p.where.setText(self._where_text("Start:  ", res.start_stage, res.start_piezo)
                            + "<br>" + self._where_text("Optimum:", *self.z["final"]))
            p.best.setText(f"Best: {c / 1000:,.1f} kcps at piezo z = {res.piezo_um[k, 2]:.3f} um<br>"
                           f"Count rate there ({avg_s:g} s average): {r[0] / 1000:,.2f} kcps")
            self.status("Moved to the optimum (best z)", 3000)

        def failed(msg):
            self.set_busy(False)
            self.error(msg)

        try:
            self.pos.check_reachable(target)
        except Exception as e:
            self.error(str(e))
            return
        self.set_busy(True)
        self.run(job, done, failed)

    # ------------------------------------------------------------------ saving
    def _get_save_target(self, title, default_name):
        """Ask where to save: (parent folder, folder name) or None."""
        dlg = SaveDialog(self, title, self.save_dir, default_name)
        return dlg.values() if dlg.exec() else None

    def _save_to_folder(self, title, default_name, write):
        """Ask for 'Save in' + 'Folder name', then call write(files, name) with the paths of
        NAME_image.png, NAME_data.csv and NAME_results.txt inside the new folder."""
        target = self._get_save_target(title, default_name)
        if not target:
            return False
        parent, name = target
        problem = saving.check_folder_name(name)
        if problem:
            self.error(problem)
            return False
        folder = os.path.join(parent, name)
        files = {k: os.path.join(folder, f"{name}_{k}") for k in ("image.png", "data.csv", "results.txt")}
        if any(os.path.exists(f) for f in files.values()) and not self.ask(
                "Replace files?", f"{folder} already contains results called {name}. Replace them?"):
            return False
        try:
            os.makedirs(folder, exist_ok=True)
            write(files, name)
        except OSError as e:
            self.error(f"Could not save: {e}")
            return False
        self.save_dir = parent
        self.status(f"Saved {name}_image.png, {name}_data.csv and {name}_results.txt in {folder}", 8000)
        return True

    def _save_scan(self, kind):
        d = self.xy if kind == "xy" else self.z
        if not d:
            return

        def write(files, name):
            res = d["res"]
            if kind == "xy":
                saving.save_raster_png(res, files["image.png"], f"XY scan {name}", d["target"],
                                       clim=self.pages["xy"].canvas.display_clim())
                saving.save_raster_csv(res, files["data.csv"], self.perm)
                text = self._xy_summary(name)
            else:
                saving.save_zsweep_png(res, files["image.png"], d["best"])
                saving.save_zsweep_csv(res, files["data.csv"], self.perm)
                text = self._z_summary(name)
            with open(files["results.txt"], "w", encoding="utf-8") as f:
                f.write(text)

        stem = "xy_scan" if kind == "xy" else "z_scan"
        self._save_to_folder(f"Save {'XY' if kind == 'xy' else 'Z'} scan",
                             f"{stem}_{time.strftime('%Y%m%d_%H%M')}", write)

    def _pos_lines(self, title, stage, piezo):
        return [title, f"  stage (mm): {fmt3(self.user(stage), 6)}", f"  piezo (um): {fmt3(self.user(piezo), 4)}"]

    def _xy_summary(self, name=""):
        d = self.xy
        res = d["res"]
        j, i, u, v, c = res.best()
        step = abs(res.u[1] - res.u[0]) if len(res.u) > 1 else 0
        cu, cv = 0.5 * (res.u[0] + res.u[-1]), 0.5 * (res.v[0] + res.v[-1])
        marker = res.frame_kind == "marker"
        lines = [f"XY scan {name}   {d['time']}", ""]
        lines += self._pos_lines("Start position (go back here to repeat the scan):",
                                 res.start_stage, res.start_piezo)
        if marker:
            lines.append(f"  centre from the BL marker: x = {cu:.3f}, y = {cv:.3f} um, "
                         f"focus offset from the marker plane {res.z_offset:+.3f} um")
        lines += ["",
                  "Scan settings:",
                  "  axes: " + ("along the marker edges (x: BL to BR, y: BL to TL), z following the marker plane"
                               if marker else "piezo x / y, " + ("z following the tilt of the marker plane"
                                                                 if res.tilt else "z fixed")),
                  "  method: " + ("as the notebook (hardware piezo x fast, every line from the same side, "
                                  f"{res.dwell_s:g} s wait then one PicoHarp reading per point)"
                                  if res.method == "notebook" else "stepper + piezo raster"),
                  f"  x range {res.u[-1] - res.u[0]:.3f} um, y range {res.v[-1] - res.v[0]:.3f} um, "
                  f"step {step:.3f} um ({len(res.u)} x {len(res.v)} points)",
                  f"  x {res.u[0]:.3f} to {res.u[-1]:.3f} um, y {res.v[0]:.3f} to {res.v[-1]:.3f} um",
                  (f"  counting {res.dwell_s:g} s per point (after a {self.cfg['count_gate_s']:g} s settle)"
                   if res.method != "notebook" else f"  dwell {res.dwell_s:g} s per point"),
                  f"  motion: {MODE_PLAIN.get(res.mode, res.mode)}"
                  + (f"; {res.clamped} edge point(s) held at the piezo limit" if res.clamped else ""),
                  f"  stepper during the scan (mm): {fmt3(self.user(res.scan_stage), 6)}",
                  f"  laser: {self._laser_text()}",
                  f"  points measured: {len(res.order)} of {res.n_points}" + ("" if res.completed else " (stopped early)"),
                  ""]
        tu, tv = d["target"]
        how = "centroid around the brightest point" if d["fit"].get("ok") else "brightest point"
        if d.get("final"):
            lines += self._pos_lines("Optimum (moved here with 'Go to optimum'):", *d["final"])
        else:
            lines += self._pos_lines("Optimum (brightest pixel as measured; not moved there):",
                                     res.stage_mm[j, i], res.piezo_um[j, i])
        lines += [f"  x = {tu:.3f}, y = {tv:.3f} um ({how}); brightest count rate {c / 1000:,.2f} kcps"]
        lines += ["After the scan the stage and piezo went back to the start position.",
                  "", "Coordinates are in the GUI frame (x/y as on the arrow buttons)."]
        return "\n".join(lines) + "\n"

    def _z_summary(self, name=""):
        d = self.z
        res = d["res"]
        k = d["best"]
        pz = res.piezo_um[:, 2]
        step = abs(pz[1] - pz[0]) if len(pz) > 1 else 0
        lines = [f"Z scan {name}   {d['time']}", ""]
        lines += self._pos_lines("Start position (go back here to repeat the scan):",
                                 res.start_stage, res.start_piezo)
        lines += ["",
                  "Scan settings:",
                  f"  z range {res.z_abs[-1] - res.z_abs[0]:.3f} um, step {step:.3f} um ({len(res.z_abs)} points)",
                  f"  piezo z {np.nanmin(pz):.3f} to {np.nanmax(pz):.3f} um",
                  f"  counting {res.dwell_s:g} s per point (after a {self.cfg['count_gate_s']:g} s settle)",
                  f"  motion: {MODE_PLAIN.get(res.mode, res.mode)}",
                  f"  stepper during the scan (mm): {fmt3(self.user(res.scan_stage), 6)}",
                  f"  laser: {self._laser_text()}",
                  ""]
        if d.get("final"):
            lines += self._pos_lines("Optimum (moved here with 'Go to optimum'):", *d["final"])
        else:
            lines += self._pos_lines("Optimum (best z as measured; not moved there):",
                                     res.stage_mm[k], res.piezo_um[k])
        lines.append(f"  piezo z = {pz[k]:.4f} um; count rate during the scan {res.counts[k] / 1000:,.2f} kcps")
        if d["avg"]:
            lines.append(f"  count rate there afterwards ({self.cfg['final_count_average_s']:g} s average): "
                         f"{d['avg'][0] / 1000:,.2f} kcps (std {d['avg'][1] / 1000:.2f})")
        lines += ["", "Coordinates are in the GUI frame (x/y as on the arrow buttons)."]
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------- closing
    def closeEvent(self, event):
        if self.scanning and not self.ask("Quit", "A scan is running. Stop it and quit?"):
            event.ignore()
            return
        self.stop()
        self._closed = True
        self.poll_timer.stop()
        self.redraw_timer.stop()
        if self.laser_on and self.ask("Laser", f"{self.laser_on[0]} is on. Turn it off before quitting?"):
            try:
                self.lasers[self.laser_on[0]].off()
            except Exception:
                pass
        for obj in list(self.lasers.values()) + [self.hw]:
            try:
                obj.close()
            except Exception:
                pass
        event.accept()