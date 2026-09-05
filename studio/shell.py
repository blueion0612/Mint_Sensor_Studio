# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
shell.py
The window: the source picker, the lesson rail, and the panel that explains
what is on screen.

The shell owns four things and no more. It owns the open source, so that
switching from a cable to a radio to a simulation to a recording changes one
object and every page carries on. It owns the session, so that whatever is
derived from the signal is derived once rather than once per page. It owns the
clock, so pages are drawn on one timer instead of competing. And it owns the
recorder, so that any source can be written to a file with one button.

Pages own their own contents. The shell never reaches into a page except to
say "you are visible now" and "here is another frame".
"""

from __future__ import annotations

import math
import os
import time
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import catalogue, core, modality, sources, theme

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
LOGO = os.path.join(HERE, "..", "img", "wordmark.png")
RECORDINGS = os.path.join(APP_DIR, "recordings")

# Twenty-two frames a second. A signal trace is smooth well below thirty, and
# every curve on screen costs about two and a half milliseconds to paint
# whatever is in it, so the frame rate is the budget the pages are written
# against. The Signals page draws nine curves; at thirty frames it would have
# spent five sixths of every frame painting and the window would feel sticky.
REFRESH_MS = 45


# ---------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------
def label(text, kind="", wrap=False, align=None):
    w = QtWidgets.QLabel(text)
    if kind:
        w.setObjectName(kind)
    w.setWordWrap(wrap)
    if align is not None:
        w.setAlignment(align)
    w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    return w


def badge(text, colour):
    """A small colored pill with white text: the modality, worn in the toolbar."""
    w = QtWidgets.QLabel(text)
    w.setObjectName("Badge")
    w.setStyleSheet("background: %s;" % colour)
    w.setAlignment(QtCore.Qt.AlignCenter)
    return w


def card(*widgets, vertical=True, margin=theme.PAD, gap=theme.GAP):
    """A white panel with a border. Every page is built out of these."""
    f = QtWidgets.QFrame()
    f.setObjectName("Card")
    lay = QtWidgets.QVBoxLayout(f) if vertical else QtWidgets.QHBoxLayout(f)
    lay.setContentsMargins(margin, margin, margin, margin)
    lay.setSpacing(gap)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return f, lay


def row(*widgets, gap=theme.GAP):
    lay = QtWidgets.QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(gap)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return lay


def clear_layout(lay):
    """Take everything out of a layout and let Qt delete it."""
    while lay.count():
        item = lay.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class Pace:
    """
    Keep the drawing inside its own frame, on whatever laptop this is.

    The pages were written against a budget: at twenty-two frames a second
    there are forty-five milliseconds, and a page that spends more than about
    two thirds of them painting makes the window feel sticky. A machine three
    times slower than the one this was built on would do exactly that.

    So the cost is measured instead of assumed. If drawing is taking too much
    of the frame, the frames are spaced further apart and the curves are drawn
    with fewer points, until it fits. It recovers the same way when the load
    drops, so moving to a cheaper page does not leave the window sluggish.
    """

    FLOOR, CEILING = 45, 120        # ms between frames

    def __init__(self):
        self.interval = self.FLOOR
        self.points = 320           # how many a curve may be drawn with
        self.cost = 0.0             # a smoothed frame cost, in ms

    def measure(self, ms) -> bool:
        """Take one frame's cost. True if the pacing changed."""
        self.cost = ms if self.cost == 0.0 else 0.9 * self.cost + 0.1 * ms
        share = self.cost / self.interval
        if share > 0.66 and self.interval < self.CEILING:
            self.interval = min(self.CEILING, int(self.interval * 1.4))
            self.points = max(120, int(self.points * 0.75))
            return True
        if share < 0.25 and self.interval > self.FLOOR:
            self.interval = max(self.FLOOR, int(self.interval / 1.4))
            self.points = min(320, int(self.points * 1.25))
            return True
        return False


class Every:
    """
    True once every n calls.

    Two things on these pages are expensive out of all proportion to how often
    they need doing. Rich text costs a layout pass, and an axis whose range has
    moved costs a whole new set of ticks. Neither is worth thirty times a
    second, and a number that changes thirty times a second cannot be read
    anyway.
    """

    def __init__(self, n):
        self.n = int(n)
        self.i = 0

    def __call__(self) -> bool:
        self.i += 1
        return self.i % self.n == 0


def steps(value, base=2.0, floor=1e-9):
    """Round up to the next power of `base`, so a range grows in jumps."""
    return float(base ** math.ceil(math.log(max(value, floor), base)))


class YRange:
    """
    Hold a plot's vertical range on a ladder of fixed steps.

    Letting a live plot autoscale looks harmless and is the single most
    expensive thing on a page: the range moves by a hair on every frame, the
    axis decides its ticks are stale, and it rebuilds the whole tick picture
    thirty times a second. On one page here that was thirteen milliseconds of
    a seventeen millisecond frame.

    Snapping the range to powers of two costs nothing and fixes it, and the
    plot is easier to read as well, because a trace that grows now moves up
    the screen instead of the screen shrinking around it.
    """

    def __init__(self, plot_item, base=2.0, pad=1.3, symmetric=True, floor=1e-3):
        self.pi = plot_item
        self.base, self.pad, self.symmetric, self.floor = base, pad, symmetric, floor
        self.now = None
        plot_item.disableAutoRange(axis="y")

    def fit(self, *arrays):
        top = 0.0
        for a in arrays:
            if len(a):
                top = max(top, float(np.nanmax(np.abs(a))))
        want = steps(max(top * self.pad, self.floor), self.base)
        if want == self.now:
            return
        self.now = want
        self.pi.setYRange(-want if self.symmetric else 0.0, want, padding=0)


class Band:
    """
    Like YRange, but for a signal that sits around a level rather than zero:
    a raw PPG at 120,000 counts, say. The range is centered on the recent mean
    and its half-width climbs the same ladder of steps.
    """

    def __init__(self, plot_item, pad=1.4, floor=1.0):
        self.pi = plot_item
        self.pad, self.floor = pad, floor
        self.now = None
        plot_item.disableAutoRange(axis="y")

    def fit(self, a):
        if not len(a):
            return
        mid = float(np.nanmean(a))
        half = steps(max(float(np.nanmax(np.abs(a - mid))) * self.pad, self.floor))
        # the center is quantized too, or the axis would move every frame
        mid = round(mid / (half / 4.0)) * (half / 4.0)
        want = (mid, half)
        if want == self.now:
            return
        self.now = want
        self.pi.setYRange(mid - half, mid + half, padding=0)


LIMIT = [320]              # what decimate() uses, lowered by Pace on a slow machine


def decimate(*arrays, limit=None):
    """
    Thin arrays to at most `limit` points each, keeping the last one.

    A plot two hundred pixels tall and a thousand wide cannot show more, and
    the path costs what it costs whether or not anyone can see the difference.
    Measured on this window: nine curves of 700 points take 31 ms a frame, the
    same nine at 250 take 20, and nobody can tell them apart.
    """
    n = len(arrays[0])
    s = max(1, n // (LIMIT[0] if limit is None else limit))
    if s == 1:
        return arrays
    return tuple(a[::s] for a in arrays)


def envelope_decimate(t, x, limit=None):
    """
    Thin a fast signal without losing its peaks.

    Plain decimation of a kilohertz EMG throws away nine samples in ten, and
    the ones thrown away are as likely as any to be the spikes. This keeps the
    minimum and the maximum of every stride, drawn as a zig-zag, which is what
    a real oscilloscope does at slow sweep speeds.
    """
    n = len(x)
    lim = LIMIT[0] if limit is None else limit
    s = max(1, n // lim)
    if s < 4:
        return t[::max(1, s)], x[::max(1, s)]
    m = (n // s) * s
    blocks = x[:m].reshape(-1, s)
    tt = t[:m].reshape(-1, s)
    lo, hi = blocks.min(axis=1), blocks.max(axis=1)
    out_t = np.column_stack([tt[:, 0], tt[:, s // 2]]).ravel()
    out_x = np.column_stack([lo, hi]).ravel()
    return out_t, out_x


def thin(t, x, hz, limit=None):
    """Decimate a trace the right way for its rate: by envelope when it is fast."""
    if hz > 400.0:
        return envelope_decimate(t, x, limit)
    return decimate(t, x, limit=limit)


def draw(curve, t, x, hz, limit=None):
    """
    Put a trace on a curve, thinned the right way for its rate, with a pen
    that paints fast.

    A kilohertz signal is drawn as the minimum and maximum of every stride, a
    zig-zag that keeps its spikes. Qt strokes a zig-zag with an antialiased
    pen wider than one pixel through its slow, general path: measured here,
    the same six hundred points cost 148 ms at width 1.4 and 2 ms at width
    1.0, which is the difference between a window that feels dead and one
    that feels live. So the pen is thinned along with the data, and put back
    when the data is not thinned that way.
    """
    import pyqtgraph as pg
    lim = LIMIT[0] if limit is None else limit
    zig = hz > 400.0 and len(x) // lim >= 4
    if zig:
        tt, xx = envelope_decimate(t, x, lim)
        # A smooth signal at a kilohertz, an envelope say, has nothing inside
        # a stride worth keeping. Its zig-zag would be one line drawn twice
        # with the thin pen; draw it once, with its own.
        spread = float(np.abs(xx[1::2] - xx[0::2]).mean()) if len(xx) > 3 else 0.0
        span = float(np.ptp(xx)) if len(xx) else 0.0
        if spread < 0.02 * span:
            zig = False
            tt, xx = tt[1::2], 0.5 * (xx[0::2] + xx[1::2])
    else:
        tt, xx = decimate(t, x, limit=lim)
    pens = getattr(curve, "_pens", None)
    if pens is None:
        wide = curve.opts["pen"]
        narrow = pg.mkPen(wide)
        narrow.setWidthF(1.0)
        pens = curve._pens = (wide, narrow)
    want = pens[1] if zig else pens[0]
    if curve.opts.get("pen") is not want:
        curve.setPen(want)
    curve.setData(tt, xx)


class Picker:
    """
    A channel box that offers whatever the current session has.

    The processing pages work on one signal at a time, and which signals there
    are depends on what is plugged in: an IMU has accelerometer axes and their
    length, an EMG board has raw, filtered and envelope, a pulse oximeter has
    two colors. The session says (`menu()`), and this keeps the box in step.
    """

    def __init__(self, on_change=None):
        self.box = QtWidgets.QComboBox()
        self.menu = []
        self._sig = None
        if on_change is not None:
            self.box.currentIndexChanged.connect(lambda _i: on_change())

    def follow(self, session):
        menu = session.menu() if session is not None else []
        sig = tuple(m[0] for m in menu)
        if sig == self._sig:
            return
        self._sig = sig
        self.menu = menu
        was = self.box.currentText()
        self.box.blockSignals(True)
        self.box.clear()
        self.box.addItems([m[0] for m in menu])
        if was in sig:
            self.box.setCurrentIndex(sig.index(was))
        self.box.blockSignals(False)

    def get(self, d):
        i = self.box.currentIndex()
        if 0 <= i < len(self.menu):
            return self.menu[i][2](d)
        return d[:, 1] if d.shape[1] > 1 else d[:, 0]

    def unit(self) -> str:
        i = self.box.currentIndex()
        return self.menu[i][1] if 0 <= i < len(self.menu) else ""

    def name(self) -> str:
        return self.box.currentText()


class Slider(QtWidgets.QWidget):
    """A labeled slider that reads out its own value. Used all over the pages."""

    changed = QtCore.Signal(float)

    def __init__(self, text, lo, hi, value, step=1.0, unit="", decimals=0, width=190):
        super().__init__()
        self._lo, self._step, self._dec, self._unit = lo, step, decimals, unit
        self._name = label(text, "Caption")
        self._value = label("", "Readout")
        self._value.setAlignment(QtCore.Qt.AlignRight)
        self._bar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._bar.setRange(0, int(round((hi - lo) / step)))
        self._bar.setFixedWidth(width)
        self._bar.valueChanged.connect(self._moved)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(row(self._name, None, self._value))
        lay.addWidget(self._bar)
        self.set(value)

    def _moved(self, _):
        v = self.value()
        self._value.setText(("%%.%df %%s" % self._dec) % (v, self._unit))
        self.changed.emit(v)

    def value(self) -> float:
        return self._lo + self._bar.value() * self._step

    def set(self, v):
        self._bar.setValue(int(round((v - self._lo) / self._step)))
        self._moved(0)


# ---------------------------------------------------------------
# choosing where the measurements come from
# ---------------------------------------------------------------
class _Scan(QtCore.QThread):
    done = QtCore.Signal(list)

    def run(self):
        self.done.emit(sources.list_ble_devices())


SPEEDS = (("real time", 1.0), ("2x", 2.0), ("4x", 4.0), ("as fast as possible", 0.0))


class SourceDialog(QtWidgets.QDialog):
    """
    Four ways to get measurements, one at a time, each with only its own
    controls on screen. A cable is not configured the way a radio is, a
    simulation is configured by what it should pretend, and a recording only
    needs a file.
    """

    def __init__(self, parent=None, current="SIM", sim_modality="imu"):
        super().__init__(parent)
        self.setWindowTitle("Source")
        self.setMinimumWidth(560)
        self.result_kind = None
        self.result_target = ""
        self.result_kw = {}

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._usb_tab(), "USB cable")
        self.tabs.addTab(self._ble_tab(), "Bluetooth")
        self.tabs.addTab(self._sim_tab(sim_modality), "Simulated")
        self.tabs.addTab(self._file_tab(), "Recording")
        self.tabs.setCurrentIndex({"USB": 0, "BLE": 1, "SIM": 2, "FILE": 3}.get(current, 2))

        connect = QtWidgets.QPushButton("Connect")
        connect.setObjectName("Primary")
        connect.setDefault(True)
        connect.clicked.connect(self._accept)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.PAD)
        lay.addWidget(self.tabs)
        lay.addLayout(row(None, cancel, connect))

    # ---- tabs ----
    def _usb_tab(self):
        w = QtWidgets.QWidget()
        self.usb_list = QtWidgets.QComboBox()
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self._fill_ports)
        self.usb_expect = QtWidgets.QComboBox()
        for text, key in (("what the board announces", "auto"), ("IMU", "imu"),
                          ("EMG", "emg"), ("PPG", "ppg")):
            self.usb_expect.addItem(text, key)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP)
        lay.addWidget(label("A board on a serial port. Its header line says what it "
                            "sends. A charge-only cable carries power but no data.",
                            "Caption", wrap=True))
        lay.addLayout(row(self.usb_list, refresh, None))
        lay.addLayout(row(label("treat it as", "Caption"), self.usb_expect, None))
        lay.addStretch(1)
        self._fill_ports()
        return w

    def _ble_tab(self):
        w = QtWidgets.QWidget()
        self.ble_list = QtWidgets.QComboBox()
        self.ble_note = label("", "Caption", wrap=True)
        self.ble_scan = QtWidgets.QPushButton("Scan")
        self.ble_scan.clicked.connect(self._scan)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP)
        lay.addWidget(label("The IMU board with no cable. Each advertises as IMU-XXXX, "
                            "from its own Bluetooth address.", "Caption", wrap=True))
        lay.addLayout(row(self.ble_list, self.ble_scan, None))
        lay.addWidget(self.ble_note)
        lay.addStretch(1)
        return w

    def _sim_tab(self, which):
        w = QtWidgets.QWidget()
        self.sim_kind = QtWidgets.QComboBox()
        for key in ("imu", "emg", "ppg"):
            m = modality.MODALITIES[key]
            self.sim_kind.addItem("%s   %s" % (m.name, m.long_name), key)
        self.sim_blurb = label("", "Caption", wrap=True)
        self.sim_pages = QtWidgets.QStackedWidget()

        # IMU
        p = QtWidgets.QWidget()
        self.sim_motion = QtWidgets.QComboBox()
        self.sim_motion.addItems(sources.MOTIONS)
        self.sim_gyro = Slider("gyroscope bias", 0.0, 5.0, 0.6, 0.1, "deg/s", 1)
        self.sim_accel = Slider("accelerometer bias", 0.0, 40.0, 5.0, 0.5, "mg", 1)
        self.sim_noise = Slider("noise", 0.0, 5.0, 1.0, 0.1, "x", 1)
        pl = QtWidgets.QVBoxLayout(p)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(theme.GAP)
        pl.addLayout(row(label("motion", "Caption"), self.sim_motion, None))
        pl.addLayout(row(self.sim_gyro, self.sim_accel, self.sim_noise, None))
        self.sim_pages.addWidget(p)

        # EMG
        p = QtWidgets.QWidget()
        self.emg_pattern = QtWidgets.QComboBox()
        self.emg_pattern.addItems(sources.EMG_PATTERNS)
        self.emg_pattern.setCurrentIndex(1)
        self.emg_amp = Slider("contraction", 0.1, 5.0, 1.0, 0.1, "mV rms", 1)
        self.emg_hum = Slider("mains hum", 0.0, 0.5, 0.05, 0.01, "mV", 2)
        self.emg_noise = Slider("noise", 0.0, 5.0, 1.0, 0.1, "x", 1)
        self.emg_mains = QtWidgets.QComboBox()
        self.emg_mains.addItems(["50 Hz", "60 Hz"])
        self.emg_count = QtWidgets.QComboBox()
        self.emg_count.addItems(["one muscle", "two muscles, with crosstalk"])
        pl = QtWidgets.QVBoxLayout(p)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(theme.GAP)
        pl.addLayout(row(label("what the muscle does", "Caption"), self.emg_pattern,
                         label("mains", "Caption"), self.emg_mains, None))
        pl.addLayout(row(label("electrodes", "Caption"), self.emg_count, None))
        pl.addLayout(row(self.emg_amp, self.emg_hum, self.emg_noise, None))
        self.sim_pages.addWidget(p)

        # PPG
        p = QtWidgets.QWidget()
        self.ppg_bpm = Slider("heart rate", 40.0, 180.0, 72.0, 1.0, "BPM", 0)
        self.ppg_hrv = Slider("variability", 0.0, 10.0, 3.0, 0.5, "%", 1)
        self.ppg_spo2 = Slider("SpO₂", 85.0, 100.0, 97.0, 0.5, "%", 1)
        self.ppg_motion = Slider("movement", 0.0, 1.0, 0.0, 0.05, "x", 2)
        self.ppg_hum = Slider("mains hum", 0.0, 0.3, 0.0, 0.01, "x", 2)
        self.ppg_noise = Slider("noise", 0.0, 5.0, 1.0, 0.1, "x", 1)
        pl = QtWidgets.QVBoxLayout(p)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(theme.GAP)
        pl.addLayout(row(self.ppg_bpm, self.ppg_hrv, self.ppg_spo2, None))
        pl.addLayout(row(self.ppg_motion, self.ppg_hum, self.ppg_noise, None))
        self.sim_pages.addWidget(p)

        self.sim_kind.currentIndexChanged.connect(self._sim_changed)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP)
        lay.addWidget(label("A sensor that does not exist. Its faults are set here, so "
                            "its truth is known and an estimate's error can be measured.",
                            "Caption", wrap=True))
        lay.addLayout(row(label("sensor", "Caption"), self.sim_kind, None))
        lay.addWidget(self.sim_blurb)
        lay.addWidget(self.sim_pages)
        lay.addStretch(1)
        self.sim_kind.setCurrentIndex({"imu": 0, "emg": 1, "ppg": 2}.get(which, 0))
        self._sim_changed()
        return w

    def _sim_changed(self, *_):
        i = self.sim_kind.currentIndex()
        self.sim_pages.setCurrentIndex(i)
        self.sim_blurb.setText(modality.MODALITIES[self.sim_kind.currentData()].blurb)

    def _file_tab(self):
        w = QtWidgets.QWidget()
        self.file_path = QtWidgets.QLineEdit()
        self.file_path.setPlaceholderText("a CSV with a header row: n,micros,...")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        self.file_speed = QtWidgets.QComboBox()
        for text, v in SPEEDS:
            self.file_speed.addItem(text, v)
        self.file_loop = QtWidgets.QCheckBox("start again at the end")
        self.file_loop.setChecked(True)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP)
        lay.addWidget(label("A CSV played back as if it were arriving now: from Record, "
                            "from the Week 2 scripts, or any file whose first row is "
                            "n, micros, then the channels.", "Caption", wrap=True))
        lay.addLayout(row(self.file_path, browse))
        lay.addLayout(row(label("play at", "Caption"), self.file_speed, self.file_loop, None))
        lay.addStretch(1)
        return w

    # ---- filling in ----
    def _fill_ports(self):
        self.usb_list.clear()
        for dev, desc, is_board in sources.list_serial_ports():
            tag = "   (an Arduino board)" if is_board else ""
            self.usb_list.addItem("%s   %s%s" % (dev, desc[:40], tag), dev)
        if not self.usb_list.count():
            self.usb_list.addItem("no serial ports on this computer", "")

    def _scan(self):
        self.ble_scan.setEnabled(False)
        self.ble_note.setText("scanning for about five seconds...")
        self._thread = _Scan()
        self._thread.done.connect(self._scanned)
        self._thread.start()

    def _scanned(self, names):
        self.ble_list.clear()
        self.ble_list.addItems(names)
        self.ble_scan.setEnabled(True)
        self.ble_note.setText("found %d" % len(names) if names else
                              "nothing found. Is the board powered, and is USE_BLE set to 1 "
                              "in the sketch?")

    def _browse(self):
        start = self.file_path.text() or (RECORDINGS if os.path.isdir(RECORDINGS) else "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a recording", start,
                                                        "Recordings (*.csv);;All files (*)")
        if path:
            self.file_path.setText(path)

    def _accept(self):
        i = self.tabs.currentIndex()
        if i == 0:
            self.result_kind, self.result_target = "USB", self.usb_list.currentData() or "auto"
            self.result_kw = dict(expect=self.usb_expect.currentData())
        elif i == 1:
            self.result_kind, self.result_target = "BLE", self.ble_list.currentText()
        elif i == 2:
            self.result_kind, self.result_target = "SIM", ""
            which = self.sim_kind.currentData()
            if which == "emg":
                self.result_kw = dict(modality="emg",
                                      pattern=self.emg_pattern.currentText(),
                                      amplitude_mv=self.emg_amp.value(),
                                      hum_mv=self.emg_hum.value(),
                                      noise=self.emg_noise.value(),
                                      mains_hz=50.0 if self.emg_mains.currentIndex() == 0 else 60.0,
                                      channels=1 + self.emg_count.currentIndex())
            elif which == "ppg":
                self.result_kw = dict(modality="ppg",
                                      bpm=self.ppg_bpm.value(), hrv=self.ppg_hrv.value(),
                                      spo2=self.ppg_spo2.value(),
                                      motion=self.ppg_motion.value(),
                                      hum=self.ppg_hum.value(), noise=self.ppg_noise.value())
            else:
                self.result_kw = dict(modality="imu",
                                      motion=self.sim_motion.currentText(),
                                      bias_dps=self.sim_gyro.value(),
                                      accel_bias_mg=self.sim_accel.value(),
                                      noise=self.sim_noise.value())
        else:
            path = self.file_path.text().strip()
            if not path:
                self._browse()
                path = self.file_path.text().strip()
                if not path:
                    return
            self.result_kind, self.result_target = "FILE", path
            self.result_kw = dict(speed=self.file_speed.currentData(),
                                  loop=self.file_loop.isChecked())
        self.accept()


# ---------------------------------------------------------------
# the panel along the bottom
# ---------------------------------------------------------------
class Explain(QtWidgets.QFrame):
    """
    The equation behind the page, its symbols, and where to read more.

    The term responsible for the trouble is red. The references are links, so
    nothing here has to stand in for a textbook, and a line in the accent
    color says what to try with the controls above.
    """

    def __init__(self):
        super().__init__()
        self.setObjectName("Explain")
        self.setMinimumHeight(100)
        self.setMaximumHeight(theme.EXPLAIN_H)

        self.head = label("", "Section")
        self.eq = label("", wrap=True)
        self.eq.setTextFormat(QtCore.Qt.RichText)
        self.terms = label("", wrap=True)
        self.terms.setTextFormat(QtCore.Qt.RichText)
        self.why = label("", wrap=True)
        self.why.setTextFormat(QtCore.Qt.RichText)
        self.why.setAlignment(QtCore.Qt.AlignTop)
        self.tip = label("", "Tip", wrap=True)
        self.tip.setTextFormat(QtCore.Qt.RichText)
        self.link = label("", wrap=True)
        self.link.setTextFormat(QtCore.Qt.RichText)
        self.link.setOpenExternalLinks(True)
        self.link.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self.head)
        left.addWidget(self.eq)
        left.addWidget(self.terms)
        left.addStretch(1)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setStyleSheet("color: %s;" % theme.LINE)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(self.why)
        right.addWidget(self.tip)
        right.addWidget(self.link)
        right.addStretch(1)

        inner = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(inner)
        box.setContentsMargins(theme.PAD, 12, theme.PAD, 12)
        box.setSpacing(theme.PAD)
        box.addLayout(left, 3)
        box.addWidget(divider)
        box.addLayout(right, 2)

        # On a short screen there is not room for four lines of algebra and a
        # paragraph. Scrolling is the only honest answer: clipping would hide
        # the term the whole panel exists to point at.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.show_nothing()

    def show_nothing(self):
        self.head.setText("")
        self.eq.setText("")
        self.terms.setText("")
        self.tip.setText("")
        self.link.setText("")
        self.why.setText("<span style='color:%s'>Choose a page on the left.</span>"
                         % theme.INK_FAINT)

    def show(self, head="", equation="", terms=(), why="", live="", source=None,
             sources=(), tip=""):
        self.head.setText(head.upper())
        self.eq.setText(equation)
        bits = []
        for sym, meaning in terms:
            bits.append("<b>%s</b> &nbsp;%s" % (sym, meaning))
        if live:
            bits.append(live)
        self.terms.setText("<span style='color:%s'>%s</span>"
                           % (theme.INK_SOFT, "<br>".join(bits)))
        self.why.setText(why)
        self.tip.setText(("<b>Try</b> &nbsp;" + tip) if tip else "")
        self.tip.setVisible(bool(tip))
        links = []
        if source:
            links.append(source)
        links += list(sources)
        self.link.setText("<br>".join(
            "<a href='%s' style='color:%s'>%s</a>" % (url, theme.ACCENT, title)
            for title, url in links))


# ---------------------------------------------------------------
# a page
# ---------------------------------------------------------------
class Page(QtWidgets.QWidget):
    """What the shell expects of every lesson."""

    title = "Page"
    subtitle = ""
    modalities = ()                     # which sensors it is for; empty means any
    explain_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Page")

    def body(self, *widgets):
        """Standard page frame: a heading, then whatever the page is."""
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP)
        head = QtWidgets.QVBoxLayout()
        head.setSpacing(1)
        self._title = label(self.title)
        self._title.setStyleSheet("font-size: %dpt; font-weight: 600;" % theme.SIZE_TITLE)
        head.addWidget(self._title)
        self._subtitle = label(self.subtitle, "Tagline")
        self._subtitle.setVisible(bool(self.subtitle))
        head.addWidget(self._subtitle)
        lay.addLayout(head)
        for w in widgets:
            if isinstance(w, QtWidgets.QLayout):
                lay.addLayout(w, 1)
            else:
                lay.addWidget(w, 1)
        return lay

    def set_subtitle(self, text):
        if hasattr(self, "_subtitle"):
            self._subtitle.setText(text)
            self._subtitle.setVisible(bool(text))

    def fits(self, session) -> bool:
        """Can this page run on that session? Empty means any signal, "*" means always."""
        if "*" in self.modalities:
            return True
        return not self.modalities or getattr(session, "modality", "generic") in self.modalities

    def on_show(self, session, source):
        pass

    def on_hide(self):
        pass

    def tick(self, session, source):
        pass

    def explain(self) -> dict:
        return {}


# ---------------------------------------------------------------
# the window
# ---------------------------------------------------------------
class Studio(QtWidgets.QMainWindow):
    def __init__(self, entries=None):
        super().__init__()
        self.setWindowTitle("MINT Sensor Studio")
        self.setMinimumSize(theme.MIN_W, theme.MIN_H)
        self._size_to_screen()

        self.source = None
        self.session = core.Session()
        self._kind = "SIM"
        self._sim_modality = "imu"
        self.pace = Pace()
        self.recorder = None
        # A page that raises puts a modal box on the screen, which is right in
        # front of a class and wrong in front of a test: a check that trips it
        # would sit there for ever waiting for a button nobody is going to
        # press. The checks set this, read `last_error`, and carry on.
        self.quiet_errors = False
        self.last_error = None
        self._seen = 0                       # samples counted at the last frame
        self._quiet_since = None             # when the source last fell silent
        self._dropped_at = (0, 0.0)          # (dropped count, when it last grew)
        self._cost_tick = Every(20)

        self._build_toolbar()
        self._build_body(catalogue.build() if entries is None else entries)
        self._shortcuts()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._frame)
        self.timer.start(self.pace.interval)

    def _size_to_screen(self):
        """Open large, but never larger than the screen it has been given."""
        screen = QtGui.QGuiApplication.primaryScreen()
        free = screen.availableGeometry() if screen else None
        w, h = theme.WANT_W, theme.WANT_H
        if free is not None:
            w = max(theme.MIN_W, min(w, free.width() - 40))
            h = max(theme.MIN_H, min(h, free.height() - 60))
        self.resize(w, h)

        self.statusBar().showMessage("Not connected.  Pick a source, or start with a "
                                     "simulated sensor and no hardware at all.")
        self.cost = label("", "Caption")
        self.statusBar().addPermanentWidget(self.cost)

    # ---- chrome ----
    def _build_toolbar(self):
        bar = QtWidgets.QToolBar()
        bar.setObjectName("TopBar")
        bar.setMovable(False)
        bar.setIconSize(QtCore.QSize(1, 1))
        self.addToolBar(bar)

        if os.path.isfile(LOGO):
            mark = QtWidgets.QLabel()
            pix = QtGui.QPixmap(LOGO)
            mark.setPixmap(pix.scaledToHeight(19, QtCore.Qt.SmoothTransformation))
            bar.addWidget(mark)
            bar.addWidget(self._spacer(12))
        bar.addWidget(label("MINT Sensor Studio", "Product"))
        bar.addWidget(self._spacer(12))
        self.badge = badge("IMU", theme.MODALITY["imu"])
        # A widget on a toolbar is shown and hidden through the action the
        # toolbar wraps it in. Hiding the widget itself leaves the action in
        # place, and showing the widget again does nothing at all.
        self.badge_action = bar.addWidget(self.badge)
        self.badge_action.setVisible(False)

        spring = QtWidgets.QWidget()
        spring.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        bar.addWidget(spring)

        self.dot = label("●", "Caption")
        self.dot.setStyleSheet("color: %s; font-size: 14pt;" % theme.INK_FAINT)
        self.dot.setToolTip("link: nothing connected")
        bar.addWidget(self.dot)
        bar.addWidget(self._spacer(6))
        self.where = label("no source", "Readout")
        self.rate = label("", "Caption")
        block = QtWidgets.QVBoxLayout()
        block.setSpacing(0)
        block.setContentsMargins(0, 0, 0, 0)
        block.addWidget(self.where, 0, QtCore.Qt.AlignRight)
        block.addWidget(self.rate, 0, QtCore.Qt.AlignRight)
        holder = QtWidgets.QWidget()
        holder.setLayout(block)
        bar.addWidget(holder)
        bar.addWidget(self._spacer(theme.PAD))

        self.pick = QtWidgets.QPushButton("Source...")
        self.pick.setToolTip("Choose where the measurements come from  (Ctrl+K)")
        self.pick.clicked.connect(self.choose_source)
        self.go = QtWidgets.QPushButton("Connect")
        self.go.setObjectName("Primary")
        self.go.clicked.connect(self.toggle)
        self.rec = QtWidgets.QPushButton("● Record")
        self.rec.setObjectName("Record")
        self.rec.setCheckable(True)
        self.rec.setToolTip("Write everything that arrives to a CSV  (Ctrl+R)")
        self.rec.clicked.connect(self.toggle_record)
        bar.addWidget(self.pick)
        bar.addWidget(self.go)
        bar.addWidget(self._spacer(6))
        bar.addWidget(self.rec)

    @staticmethod
    def _spacer(w):
        s = QtWidgets.QWidget()
        s.setFixedWidth(w)
        return s

    def _shortcuts(self):
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+K"), self, self.choose_source)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, self.rec.click)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, lambda: self.choose_source("FILE"))

    def _build_body(self, entries):
        """
        The rail: a tree of groups, one per sensor and one of functions that
        work on any signal. A page for another sensor than the one connected
        is greyed, with the reason in its tooltip.
        """
        self.rail = QtWidgets.QTreeWidget()
        self.rail.setObjectName("Rail")
        self.rail.setHeaderHidden(True)
        self.rail.setRootIsDecorated(False)
        self.rail.setIndentation(16)
        self.rail.setExpandsOnDoubleClick(False)
        self.rail.setFocusPolicy(QtCore.Qt.NoFocus)
        self.rail.setFixedWidth(theme.RAIL_W)
        self.rail.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.stack = QtWidgets.QStackedWidget()
        self.entries = list(entries)
        self.pages = []
        self.groups = []
        self._items = []
        self._group_items = {}
        for group, key, page in self.entries:
            top = self._group_items.get(group.key)
            if top is None:
                top = QtWidgets.QTreeWidgetItem([group.name])
                top.setFlags(QtCore.Qt.ItemIsEnabled)          # a heading, not a page
                top.setData(0, QtCore.Qt.UserRole, None)
                top.setData(0, QtCore.Qt.UserRole + 1, group.key)
                top.setToolTip(0, group.blurb)
                self.rail.addTopLevelItem(top)
                self._group_items[group.key] = top
            index = self.stack.count()
            self.pages.append(page)
            self.groups.append(group)
            self.stack.addWidget(page)
            item = QtWidgets.QTreeWidgetItem([page.title])
            item.setData(0, QtCore.Qt.UserRole, index)
            item.setToolTip(0, page.subtitle)
            top.addChild(item)
            self._items.append(item)
            page.explain_changed.connect(self._refresh_explain)
        for key, top in self._group_items.items():
            self._label_group(top, key in ("imu", "signal"))
            top.setExpanded(key in ("imu", "signal"))
        self.rail.itemExpanded.connect(lambda it: self._label_group(it, True))
        self.rail.itemCollapsed.connect(lambda it: self._label_group(it, False))
        self.rail.itemClicked.connect(self._clicked)
        self.rail.currentItemChanged.connect(self._went_to)

        self.explain = Explain()

        upper = QtWidgets.QWidget()
        up = QtWidgets.QHBoxLayout(upper)
        up.setContentsMargins(0, 0, 0, 0)
        up.setSpacing(0)
        up.addWidget(self.rail)
        up.addWidget(self.stack, 1)

        centre = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(centre)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(upper, 1)
        lay.addWidget(self.explain)
        self.setCentralWidget(centre)
        self.goto(0)
        # After the window is up, not before: the cost being paid here is font
        # substitution, and Qt only goes looking for a face when it actually
        # has to paint the glyph.
        QtCore.QTimer.singleShot(0, self._warm_up)

    @staticmethod
    def _label_group(item, expanded):
        key = item.data(0, QtCore.Qt.UserRole + 1)
        name = {g.key: g.name for g in catalogue.GROUPS}.get(key, key or "")
        item.setText(0, ("\u25be  " if expanded else "\u25b8  ") + name.upper())

    def _clicked(self, item, _column):
        if item.data(0, QtCore.Qt.UserRole) is None:      # a heading: fold or unfold
            item.setExpanded(not item.isExpanded())

    def goto(self, index):
        """Show page number `index`, unfolding its group."""
        item = self._items[index]
        item.parent().setExpanded(True)
        self.rail.setCurrentItem(item)
        if self.stack.currentIndex() != index:
            self._went_to(item, None)

    def _warm_up(self):
        """
        Paint every equation once, before anyone is waiting for one.

        The equations use glyphs the interface font does not have, and the
        first time Qt has to paint one it searches every installed font for a
        face that does. Measured here: the first press of an estimator button
        cost 460 ms and every press after it cost 6. This moves that cost to
        start-up, where a third of a second is invisible.
        """
        for page in self.pages:
            info = page.explain()
            if info:
                self.explain.show(**info)
                self.explain.repaint()
        self._refresh_explain()

    def _gate(self):
        """Gray the pages that cannot run on what is connected, unfold the group that can."""
        kind = getattr(self.session, "modality", "generic")
        current = self.stack.currentIndex()
        fallback = None
        for i, (page, item) in enumerate(zip(self.pages, self._items)):
            ok = self.source is None or page.fits(self.session)
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                          if ok else QtCore.Qt.NoItemFlags)
            if ok:
                item.setToolTip(0, page.subtitle)
                if fallback is None:
                    fallback = i
            else:
                item.setToolTip(0, "%s is for %s. %s is connected."
                                % (page.title, ", ".join(m.upper() for m in page.modalities),
                                   kind.upper()))
        if self.source is not None:
            for key, top in self._group_items.items():
                if key == kind:
                    top.setExpanded(True)
        if (self.source is not None and not self.pages[current].fits(self.session)
                and fallback is not None):
            self.goto(fallback)

    # ---- source ----
    def choose_source(self, tab=None):
        d = SourceDialog(self, tab or self._kind, self._sim_modality)
        if d.exec() != QtWidgets.QDialog.Accepted:
            return
        self._kind = d.result_kind
        if d.result_kind == "SIM":
            self._sim_modality = d.result_kw.get("modality", "imu")
        self._open(d.result_kind, d.result_target, d.result_kw)

    def toggle(self):
        if self.source is not None:
            self._close()
        elif self._kind == "SIM":
            self._open("SIM", "", dict(modality=self._sim_modality))
        else:
            self.choose_source()

    def _open(self, kind, target, kw):
        self._close()
        self.statusBar().showMessage("Opening %s..." % kind)
        QtWidgets.QApplication.processEvents()
        try:
            src = sources.open_source(kind, target, **dict(kw))
            src.wait(20.0 if kind == "BLE" else 6.0)
        except Exception as e:                                  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Cannot open that source", str(e))
            self.statusBar().showMessage("Not connected.")
            return
        self.source = src
        self.session = core.session_for(src)
        self._seen = 0
        self._quiet_since = None
        self._dropped_at = (0, time.time())
        for p in self.pages:
            p.on_hide()
        self.go.setText("Disconnect")
        self.go.setObjectName("")
        self.go.setStyleSheet("")
        self.style().polish(self.go)
        self.badge.setText(src.badge)
        self.badge.setStyleSheet("background: %s;" % theme.MODALITY.get(src.modality,
                                                                          theme.MODALITY["generic"]))
        self.badge_action.setVisible(True)
        self.statusBar().showMessage(src.info or src.where())
        self._gate()
        self.pages[self.stack.currentIndex()].on_show(self.session, self.source)
        self._refresh_explain()

    def _close(self):
        if self.recorder is not None:
            self.rec.setChecked(False)
            self.toggle_record()
        if self.source is not None:
            self.source.close()
            self.source = None
        self.session.reset()
        self.go.setText("Connect")
        self.go.setObjectName("Primary")
        self.style().polish(self.go)
        self.where.setText("no source")
        self.rate.setText("")
        self.badge_action.setVisible(False)
        self.dot.setStyleSheet("color: %s; font-size: 14pt;" % theme.INK_FAINT)
        self.dot.setToolTip("link: nothing connected")
        self._gate()

    # ---- recording ----
    def toggle_record(self):
        if self.recorder is None:
            if self.source is None:
                self.rec.setChecked(False)
                self.statusBar().showMessage("Connect a source first; there is nothing to record.")
                return
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RECORDINGS, "%s_%s" % (self.source.modality, stamp) + ".csv")
            try:
                self.recorder = sources.Recorder(path, self.session.channels)
            except OSError as e:
                self.rec.setChecked(False)
                QtWidgets.QMessageBox.warning(self, "Cannot record", str(e))
                return
            self.rec.setChecked(True)
            self.statusBar().showMessage("Recording to %s" % path)
        else:
            rec, self.recorder = self.recorder, None
            rec.close()
            self.rec.setChecked(False)
            self.rec.setText("● Record")
            self.statusBar().showMessage("Saved %d samples, %.0f s, to %s"
                                         % (rec.n, rec.seconds, rec.path))

    # ---- pages ----
    def _went_to(self, item, _previous=None):
        if item is None:
            return
        index = item.data(0, QtCore.Qt.UserRole)
        if index is None:
            return
        for j, p in enumerate(self.pages):
            if j != index:
                p.on_hide()
        self.stack.setCurrentIndex(index)
        self.pages[index].on_show(self.session, self.source)
        self._refresh_explain()

    def _refresh_explain(self):
        if not self.pages:
            return
        info = self.pages[self.stack.currentIndex()].explain()
        if info:
            self.explain.show(**info)
        else:
            self.explain.show_nothing()

    def _link_health(self, src):
        """The dot beside the rate: green, amber when losing, red when silent."""
        now = time.time()
        if src.count > self._seen:
            self._seen = src.count
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = now
        dropped, when = self._dropped_at
        if src.dropped > dropped:
            self._dropped_at = (src.dropped, now)
            when = now
        if self._quiet_since is not None and now - self._quiet_since > 1.5:
            colour, text = theme.GUILTY, "link: nothing has arrived for a while"
        elif now - when < 3.0 and src.dropped:
            colour, text = theme.WARN, "link: samples are being lost"
        else:
            colour, text = theme.GOOD, "link: samples arriving, none lost recently"
        if getattr(self, "_dot_state", None) != colour:
            self._dot_state = colour
            self.dot.setStyleSheet("color: %s; font-size: 14pt;" % colour)
            self.dot.setToolTip(text)

    # ---- the clock ----
    def _frame(self):
        started = time.perf_counter()
        src = self.source
        if src is not None:
            if src.error is not None:
                err, src.error = src.error, None
                QtWidgets.QMessageBox.warning(self, "The source stopped", str(err))
                self._close()
                return
            samples = src.read()
            if samples:
                self.session.add(samples)
                if self.recorder is not None:
                    self.recorder.feed(samples)
            self.where.setText(src.where())
            self.rate.setText(src.rate_text())
            self._link_health(src)
            if self.recorder is not None and self._cost_tick.i % 10 == 0:
                s = int(self.recorder.seconds)
                self.rec.setText("■ %d:%02d" % (s // 60, s % 60))
        page = self.pages[self.stack.currentIndex()]
        try:
            # A page for another sensor is never asked to draw this one. The
            # rail keeps it out of reach; this keeps it out of trouble.
            if src is None or page.fits(self.session):
                page.tick(self.session, src)
        except Exception:                                        # noqa: BLE001
            self.last_error = traceback.format_exc()
            self.timer.stop()
            if not self.quiet_errors:
                QtWidgets.QMessageBox.critical(self, "That page stopped",
                                               self.last_error)
            return
        cost = (time.perf_counter() - started) * 1e3
        if self.pace.measure(cost):
            LIMIT[0] = self.pace.points
            self.timer.setInterval(self.pace.interval)
        if self._cost_tick():
            self.cost.setText("drawing %.0f ms of %d" % (self.pace.cost, self.pace.interval))

    def closeEvent(self, event):
        self.timer.stop()
        self._close()
        super().closeEvent(event)
