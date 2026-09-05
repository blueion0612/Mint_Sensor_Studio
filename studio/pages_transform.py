# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_transform.py
The three pages that do something to the signal rather than to the sensor.

    Convolution   an impulse response, and the sum that applies it
    Spectrum      the same seconds seen as frequencies
    Filters       keeping part of the band, and what that costs in time

Each one runs on whatever is arriving now, from whichever sensor is connected:
a shaken board, a contracting muscle and a fingertip tell different stories on
the same page. The channel box offers what the session has.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import core, dsp, theme
from .shell import Every, Page, Picker, Slider, YRange, card, decimate, draw, label, row

WINDOW = 8.5           # seconds the time plots look back over


class ConvolutionPage(Page):
    title = "Convolution"
    subtitle = "Impulse response and the convolution sum"

    KERNELS = ("moving average", "exponential", "difference", "raised cosine")

    def __init__(self):
        super().__init__()
        self.kernel_box = QtWidgets.QComboBox()
        self.kernel_box.addItems(self.KERNELS)
        self.kernel_box.currentIndexChanged.connect(self._redraw_kernel)
        self.length = Slider("length N", 2, 120, 15, 1, "samples", 0, width=230)
        self.length.changed.connect(lambda _: self._redraw_kernel())
        self.pick = Picker(self._picked)

        self.kplot = pg.PlotWidget()
        kp = theme.plot(self.kplot, "k  (samples)", "h[k]")
        theme.title(kp, "The impulse response h[k]")
        self.stems = kp.plot(pen=None, symbol="o", symbolSize=5,
                             symbolBrush=theme.ACCENT, symbolPen=None)
        self.drops = kp.plot(pen=theme.pen(theme.ACCENT_LINE, 1.0))
        self.kplot.setMaximumHeight(210)

        self.tplot = pg.PlotWidget()
        self.tp = theme.plot(self.tplot, "seconds before now", "")
        theme.title(self.tp, "x[n] and y[n] = x * h")
        self.tp.setXRange(-WINDOW + 0.5, 0, padding=0)
        theme.legend(self.tp)
        self.raw = self.tp.plot(pen=theme.pen(theme.LINE, 1.4), name="x[n], as measured")
        self.out = self.tp.plot(pen=theme.pen(theme.ACCENT, 1.9), name="y[n], after h")
        self.ty = YRange(self.tp)
        self.ky = YRange(kp)
        self.slow = Every(6)

        self.readout = label("", "Caption", wrap=True)

        top, _ = card(self.kplot,
                      row(label("kernel", "Caption"), self.kernel_box, self.length,
                          label("channel", "Caption"), self.pick.box, None),
                      self.readout)
        bottom, _ = card(self.tplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 0)
        col.addWidget(bottom, 1)
        self.body(col)
        self._session = None
        self._redraw_kernel()

    def _picked(self):
        theme.axis_label(self.tp, "left", self.pick.name(), self.pick.unit())

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)
        self._picked()

    def kernel(self) -> np.ndarray:
        n = int(self.length.value())
        name = self.kernel_box.currentText()
        if name == "moving average":
            h = np.ones(n)
        elif name == "exponential":
            h = np.exp(-np.arange(n) / max(n / 4.0, 1.0))
        elif name == "difference":
            h = np.zeros(max(n, 2))
            h[0], h[-1] = 1.0, -1.0
            return h                                  # already sums to zero, do not scale
        else:
            h = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 1) / (n + 1))
        s = h.sum()
        return h / s if s else h

    def _redraw_kernel(self):
        h = self.kernel()
        k = np.arange(len(h))
        self.stems.setData(k, h)
        self.ky.fit(h)
        gap = np.full(len(h), np.nan)
        xs = np.column_stack([k, k, gap]).ravel()
        ys = np.column_stack([np.zeros(len(h)), h, gap]).ravel()
        self.drops.setData(xs, ys)
        self.explain_changed.emit()

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        d = session.last(WINDOW)
        if len(d) < 40:
            return
        h = self.kernel()
        x = self.pick.get(d)
        y = np.convolve(x, h, mode="same")
        t = d[:, core.T] - d[-1, core.T]
        self.ty.fit(x, y)
        draw(self.raw, t, x, session.hz)
        draw(self.out, t, y, session.hz)
        if not self.slow():
            return

        hz = session.hz or 100.0
        n = len(h)
        self.readout.setText(
            "<span style='color:%s'>N = %d at %.0f Hz is a window of <b>%.0f ms</b>. "
            "A moving average of that length is a low-pass whose first null sits at "
            "<b>%.1f Hz</b>, and it delays everything by <b>%.0f ms</b>.</span>"
            % (theme.INK_SOFT, n, hz, 1000.0 * n / hz, hz / n, 1000.0 * (n - 1) / (2 * hz)))

    def explain(self):
        return dict(
            head="Convolution",
            equation="<span style='font-size:16pt'>y[n] &nbsp;=&nbsp; "
                     "&#8721;<sub>k</sub> h[k] &#183; x[n&minus;k]</span>",
            terms=(("h[k]", "impulse response"),
                   ("<span style='color:%s'>(N&minus;1)/2</span>" % theme.GUILTY,
                    "samples of delay, for a moving average of length N")),
            why="An LTI system is its impulse response: the output for any input is "
                "this sum.",
            tip="Lengthen the moving average until the trace goes smooth, then see how "
                "far behind the grey it runs. Smoothing costs delay.",
            source=("Linear time-invariant system — Wikipedia",
                    "https://en.wikipedia.org/wiki/Linear_time-invariant_system"),
            sources=(("Convolution, Wikipedia", "https://en.wikipedia.org/wiki/Convolution"),))


class SpectrumPage(Page):
    title = "Spectrum"
    subtitle = "Discrete Fourier transform, windowing"

    def __init__(self):
        super().__init__()
        self.pick = Picker(self._picked)
        self.window = QtWidgets.QComboBox()
        self.window.addItems(["Hann", "rectangular", "Hamming", "Blackman"])
        self.window.currentIndexChanged.connect(lambda _: self.explain_changed.emit())
        self.span = Slider("window length", 1.0, 20.0, 8.0, 0.5, "s", 1, width=220)
        self.span.changed.connect(lambda _: self.explain_changed.emit())
        self.logy = QtWidgets.QCheckBox("decibels")
        self.logy.toggled.connect(self._scale)

        self.tplot = pg.PlotWidget()
        self.tp = theme.plot(self.tplot, "seconds before now", "")
        theme.title(self.tp, "The window being transformed")
        self.wave = self.tp.plot(pen=theme.pen(theme.INK_SOFT, 1.3))
        self.tplot.setMaximumHeight(200)
        self._span = None

        self.fplot = pg.PlotWidget()
        self.fp = theme.plot(self.fplot, "frequency", "magnitude", "Hz", "")
        theme.title(self.fp, "Magnitude spectrum")
        self.spec = self.fplot.plot(pen=theme.pen(theme.ACCENT, 1.7),
                                    fillLevel=0, brush=theme.fill(theme.ACCENT, 40))
        self.peak = pg.InfiniteLine(angle=90, pen=theme.pen(theme.X_AXIS, 1.4, dash=True),
                                    label="{value:.2f} Hz",
                                    labelOpts={"position": 0.9, "color": theme.X_AXIS})
        self.fp.addItem(self.peak)

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 3))
        self.ty = YRange(self.tplot.getPlotItem())
        self.fy = YRange(self.fp, symmetric=False, floor=1e-4)
        self.fx = None
        self.slow = Every(6)

        top, _ = card(self.tplot,
                      row(label("channel", "Caption"), self.pick.box,
                          label("window", "Caption"), self.window,
                          self.span, self.logy, None),
                      self.readout)
        bottom, _ = card(self.fplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 0)
        col.addWidget(bottom, 1)
        self.body(col)
        self._session = None

    def _picked(self):
        theme.axis_label(self.tp, "left", self.pick.name(), self.pick.unit())

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)
        self._picked()

    def _scale(self, on):
        self.fp.setLogMode(False, False)
        theme.axis_label(self.fp, "left", "magnitude, dB" if on else "magnitude")
        self.explain_changed.emit()

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        span = self.span.value()
        d = session.last(span)
        if len(d) < 64:
            return
        t = d[:, core.T]
        hz = len(t) / max(t[-1] - t[0], 1e-6)
        x = self.pick.get(d)
        x = x - x.mean()
        self.ty.fit(x)
        if span != self._span:
            self._span = span
            self.tp.setXRange(-span, 0, padding=0)
        draw(self.wave, t - t[-1], x, session.hz)

        name = {"Hann": "hann", "rectangular": "rect", "Hamming": "hamming",
                "Blackman": "blackman"}[self.window.currentText()]
        freq, spec = dsp.spectrum(x, hz, name)
        if self.logy.isChecked():
            shown = 20 * np.log10(np.maximum(spec, 1e-9))
            floor = -120.0
            self.spec.setFillLevel(floor)
            if self.fy.now != floor:
                self.fy.now = floor
                self.fp.setYRange(floor, 20.0, padding=0)
        else:
            shown = spec
            self.spec.setFillLevel(0.0)
            self.fy.fit(spec)
        # A kilohertz record has thousands of bins; the plot has a few hundred pixels.
        self.spec.setData(*decimate(freq, shown, limit=1200))
        top_f = round(freq[-1])
        if top_f != self.fx:
            self.fx = top_f
            self.fp.setXRange(0, top_f, padding=0)

        if not self.slow():
            return
        lo = np.searchsorted(freq, 0.3)               # ignore what is left of the mean
        if len(spec) > lo + 2:
            i = lo + int(np.argmax(spec[lo:]))
            self.peak.setValue(float(freq[i]))
            self.readout.setText(
                "<span style='color:%s'>peak </span><b>%.2f Hz</b>"
                "<span style='color:%s'>&nbsp; = %.0f per minute &nbsp;·&nbsp; "
                "resolution %.3f Hz</span>"
                % (theme.INK_FAINT, freq[i], theme.INK_FAINT, freq[i] * 60,
                   hz / len(x)))

    def explain(self):
        return dict(
            head="Discrete Fourier transform",
            equation="<span style='font-size:16pt'>X[k] &nbsp;=&nbsp; "
                     "&#8721;<sub>n=0</sub><sup>N&minus;1</sup> "
                     "<span style='color:%s'>w[n]</span> &#183; x[n] &#183; "
                     "e<sup>&minus;j2&#960;kn/N</sup></span>" % theme.ACCENT,
            terms=(("&#916;f = f<sub>s</sub>/N", "frequency resolution"),
                   ("<span style='color:%s'>w[n]</span>" % theme.ACCENT,
                    "window function")),
            why="The DFT treats the window as one period. Ends that do not join up "
                "spread energy everywhere; a taper at the ends limits it.",
            tip="Switch to a rectangular window with a shaken board and watch the peak "
                "grow skirts. On a pulse, read the heart rate off the first peak.",
            source=("Window function — Wikipedia",
                    "https://en.wikipedia.org/wiki/Window_function"),
            sources=(("Discrete Fourier transform, Wikipedia",
                      "https://en.wikipedia.org/wiki/Discrete_Fourier_transform"),))


class Causal:
    """
    Run a filter over the session as it arrives, keeping the output beside the
    clock, so that only the new samples are filtered on each frame.

    Filtering the whole window again every frame is fine at a hundred hertz
    and is thirty milliseconds a frame at a thousand without scipy. A filter
    with state does not need to: it takes what it has not seen and carries on.
    """

    def __init__(self):
        self.sos = None
        self.f = None
        self.t = np.zeros(0)
        self.y = np.zeros(0)
        self.last = None

    def design(self, sos):
        self.sos = sos
        self.f = dsp.Sos(sos, 1)
        self.t = np.zeros(0)
        self.y = np.zeros(0)
        self.last = None

    def update(self, t, x, keep):
        if self.f is None:
            return self.t, self.y
        new = slice(0, len(t)) if self.last is None else slice(
            int(np.searchsorted(t, self.last, side="right")), len(t))
        if new.stop > new.start:
            if self.last is None:
                self.f.prime(x[new][0])          # no swing from zero at the start
            yn = self.f.process(x[new])
            self.t = np.concatenate([self.t, t[new]])
            self.y = np.concatenate([self.y, yn])
            self.last = float(t[-1])
        cut = self.t > (self.t[-1] - keep if len(self.t) else 0.0)
        self.t, self.y = self.t[cut], self.y[cut]
        return self.t, self.y


class FilterPage(Page):
    title = "Filters"
    subtitle = "Butterworth low-pass, high-pass, band-pass"

    KINDS = ("low-pass", "high-pass", "band-pass", "notch")

    def __init__(self):
        super().__init__()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(self.KINDS)
        self.kind.currentIndexChanged.connect(self._changed)
        self.pick = Picker(self._changed)
        self.cut = Slider("cut-off", 0.2, 25.0, 3.0, 0.1, "Hz", 1, width=200)
        self.cut.changed.connect(self._changed)
        self.width = Slider("band width", 0.2, 20.0, 4.0, 0.1, "Hz", 1, width=180)
        self.width.changed.connect(self._changed)
        self.order = Slider("order", 1, 8, 4, 1, "", 0, width=140)
        self.order.changed.connect(self._changed)
        self.zero_phase = QtWidgets.QCheckBox("zero phase (only possible offline)")
        self.zero_phase.toggled.connect(self._changed)

        self.tplot = pg.PlotWidget()
        self.tp = theme.plot(self.tplot, "seconds before now", "")
        theme.title(self.tp, "Before and after")
        self.tp.setXRange(-WINDOW + 0.5, 0, padding=0)
        theme.legend(self.tp)
        self.raw = self.tp.plot(pen=theme.pen(theme.LINE, 1.5), name="in")
        self.out = self.tp.plot(pen=theme.pen(theme.ACCENT, 1.9), name="out")

        self.fplot = pg.PlotWidget()
        self.fp = theme.plot(self.fplot, "frequency", "", "Hz", "")
        theme.title(self.fp, "What the filter keeps, over what the signal contains")
        theme.legend(self.fp)
        self.sig_spec = self.fplot.plot(pen=theme.pen(theme.LINE, 1.4),
                                        fillLevel=0, brush=theme.fill(theme.LINE, 120),
                                        name="signal")
        self.resp = self.fplot.plot(pen=theme.pen(theme.ACCENT, 2.0), name="|H(f)|")
        self.edge = pg.InfiniteLine(angle=90, pen=theme.pen(theme.X_AXIS, 1.2, dash=True))
        self.fp.addItem(self.edge)
        self.ty = YRange(self.tp)
        self.fp.setYRange(0, 1.05, padding=0)
        self.fp.disableAutoRange(axis="y")
        self.fx = None
        self.slow = Every(6)
        self.causal = Causal()
        self._design_key = None
        self._sos = None

        self.note = label("", "Caption", wrap=True)

        controls = row(label("kind", "Caption"), self.kind, self.cut, self.width,
                       self.order, None)
        controls2 = row(label("channel", "Caption"), self.pick.box,
                        self.zero_phase, None)
        top, _ = card(self.tplot, controls, controls2, self.note)
        bottom, _ = card(self.fplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(col)
        self._session = None
        self._changed()

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)
        # The cut-off slider spans what makes sense for the rate: a few hertz
        # for a hundred hertz IMU, hundreds for a kilohertz EMG.
        top = max(5.0, min(0.45 * session.hz, 450.0))
        if abs(self.cut._bar.maximum() * self.cut._step + self.cut._lo - top) > 1e-6:
            v = min(self.cut.value(), top)
            self.cut._bar.setRange(0, int(round((top - self.cut._lo) / self.cut._step)))
            self.cut.set(v)
            self.width._bar.setRange(0, int(round((top * 0.8 - self.width._lo)
                                                  / self.width._step)))
        theme.axis_label(self.tp, "left", self.pick.name(), self.pick.unit())
        self._design_key = None

    def _changed(self, *_):
        self.width.setVisible(self.kind.currentText() in ("band-pass", "notch"))
        self._design_key = None
        self.explain_changed.emit()

    def _design(self, hz):
        ny = hz / 2.0
        kind = self.kind.currentText()
        n = int(self.order.value())
        f = self.cut.value()
        try:
            if kind == "notch":
                if f >= ny * 0.98:
                    return None
                return dsp.notch(f, hz, q=max(f / max(self.width.value(), 0.05), 0.5))
            if kind == "band-pass":
                half = self.width.value() / 2.0
                lo, hi = max(f - half, 0.05), min(f + half, ny * 0.98)
                if hi <= lo:
                    return None
                return dsp.butterworth(n, "bandpass", lo, hz, hi)
            if f >= ny * 0.98:
                return None
            return dsp.butterworth(n, "lowpass" if kind == "low-pass" else "highpass", f, hz)
        except ValueError:
            return None

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        d = session.last(WINDOW)
        if len(d) < 64:
            return
        t = d[:, core.T]
        hz = len(t) / max(t[-1] - t[0], 1e-6)
        x = self.pick.get(d)
        key = (round(hz), self.kind.currentText(), int(self.order.value()),
               round(self.cut.value(), 2), round(self.width.value(), 2),
               self.pick.name())
        if key != self._design_key:
            self._design_key = key
            self._sos = self._design(hz)
            if self._sos is not None:
                self.causal.design(self._sos)
        sos = self._sos
        if sos is None:
            self.note.setText("that cut-off is at or past the Nyquist frequency "
                              "(%.1f Hz), where there is nothing left to keep." % (hz / 2))
            return

        if self.zero_phase.isChecked():
            # Offline by definition, so the whole window every frame. A
            # kilohertz signal gets four seconds of it, which is enough to look at.
            keep = len(x) if hz < 400 else int(4 * hz)
            y = dsp.filtfilt(sos, x[-keep:])
            ty = t[-keep:]
        else:
            ty, y = self.causal.update(t, x, WINDOW)
        self.ty.fit(x, y)
        draw(self.raw, t - t[-1], x, session.hz)
        if len(ty):
            draw(self.out, ty - t[-1], y, session.hz)
        if not self.slow():
            return

        w, mag, phase = dsp.response(sos, hz)
        freq, spec = dsp.spectrum(x, hz)
        top = max(float(spec.max()), 1e-9)
        self.sig_spec.setData(*decimate(freq, spec / top, limit=1200))
        self.resp.setData(w, mag)
        self.edge.setValue(self.cut.value())
        top_f = round(freq[-1])
        if top_f != self.fx:
            self.fx = top_f
            self.fp.setXRange(0, top_f, padding=0)

        if self.zero_phase.isChecked():
            how = ("Run forwards and then backwards, so nothing comes out late. That "
                   "needs the end of the record before it can produce the beginning, so "
                   "it cannot be done while the data is still arriving.")
        else:
            how = ("Run once, forwards, the way it would run on the board. Everything "
                   "comes out about <b>%.0f ms</b> late, and the steeper the filter the "
                   "later it is." % (1000.0 * self._delay_s(w, phase)))
        engine = "scipy" if dsp._sp is not None else "the biquad cascade in dsp.py"
        self.note.setText("<span style='color:%s'>Butterworth, order %d, sampled at "
                          "%.0f Hz, designed with %s. %s</span>"
                          % (theme.INK_SOFT, int(self.order.value()), hz, engine, how))

    @staticmethod
    def _delay_s(w, phase):
        """
        Group delay near direct current, in seconds, read off the phase slope.

        That is where the signal the student is watching actually sits for a
        low-pass, and it is the number that matches what the eye sees between
        the gray trace and the green one.
        """
        i = max(1, len(w) // 40)
        d = -(phase[i] - phase[0]) / (2 * np.pi * (w[i] - w[0]))
        return float(max(d, 0.0))

    def explain(self):
        return dict(
            head="Butterworth filter",
            equation="<span style='font-size:16pt'>y[n] &nbsp;=&nbsp; "
                     "&#8721;<sub>i</sub> b<sub>i</sub> x[n&minus;i] &nbsp;&minus;&nbsp; "
                     "&#8721;<sub>j&gt;0</sub> a<sub>j</sub> y[n&minus;j]"
                     "&nbsp;&nbsp;&nbsp;&nbsp;"
                     "<span style='color:%s'>&#8736;H(f) &#8800; 0</span></span>"
                     % theme.GUILTY,
            terms=(("|H(f)|", "magnitude response, drawn below"),
                   ("<span style='color:%s'>&#8736;H(f)</span>" % theme.GUILTY,
                    "phase response, which is the delay")),
            why="Maximally flat in the pass band, and never on time: a causal filter "
                "delays what it keeps, more so the steeper it is.",
            tip="Raise the order: the green trace falls further behind the grey. Tick "
                "zero phase and the delay vanishes, with any chance of doing it live.",
            source=("Butterworth filter — Wikipedia",
                    "https://en.wikipedia.org/wiki/Butterworth_filter"),
            sources=(("Digital biquad filter, Wikipedia",
                      "https://en.wikipedia.org/wiki/Digital_biquad_filter"),))
