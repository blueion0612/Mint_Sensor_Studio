# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_analysis.py
Four functions that work on whichever signal is connected.

    Periodicity   does it repeat, how often, and which harmonics make it up
    Spectrogram   the spectrum frame by frame
    Denoise       four smoothers, and what each one throws away
    Separation    principal and independent components of the channels
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import analysis, core, dsp, theme
from .shell import Every, Page, Picker, Slider, YRange, card, decimate, draw, label, row

FAINT = None   # filled at use; theme colors are read when a page is built


def _cell(name, value, unit=""):
    return ("<tr><td style='color:%s'>%s</td><td align=right><b>%s</b></td>"
            "<td style='color:%s'>%s</td></tr>" % (theme.INK_FAINT, name, value,
                                                    theme.INK_FAINT, unit))


class PeriodicityPage(Page):
    title = "Periodicity"
    subtitle = "Fundamental frequency by autocorrelation, and the Fourier series it implies"

    def __init__(self):
        super().__init__()
        self.pick = Picker(self._picked)
        self.span = Slider("window", 2.0, 20.0, 8.0, 0.5, "s", 1, width=180)
        self.span.changed.connect(lambda _: self.explain_changed.emit())
        self.harm = Slider("harmonics N", 1, 12, 5, 1, "", 0, width=150)
        self.harm.changed.connect(lambda _: self.explain_changed.emit())

        self.tplot = pg.PlotWidget()
        self.tp = theme.plot(self.tplot, "seconds before now", "")
        theme.title(self.tp, "The signal, and its Fourier series with N harmonics")
        theme.legend(self.tp)
        self.raw = self.tp.plot(pen=theme.pen(theme.LINE, 1.4), name="signal")
        self.series = self.tp.plot(pen=theme.pen(theme.ACCENT, 2.0), name="series")
        self.ty = YRange(self.tp)
        # Three plots on one page: each may be shorter than the house minimum,
        # or the page does not fit a 640-pixel window.
        self.tplot.setMinimumHeight(100)

        self.aplot = pg.PlotWidget()
        ap = theme.plot(self.aplot, "lag", "r", "s", "")
        self.aplot.setMinimumHeight(90)
        theme.title(ap, "Autocorrelation")
        ap.setYRange(-1.0, 1.05, padding=0)
        ap.disableAutoRange(axis="y")
        self.acf = ap.plot(pen=theme.pen(theme.INK_SOFT, 1.4))
        self.peak = pg.InfiniteLine(angle=90, pen=theme.pen(theme.GUILTY, 1.4, dash=True),
                                    label="T₀ = {value:.3f} s",
                                    labelOpts={"position": 0.9, "color": theme.GUILTY})
        ap.addItem(self.peak)
        self.ap = ap

        self.hplot = pg.PlotWidget()
        hp = theme.plot(self.hplot, "harmonic k", "|cₖ|", "", "")
        self.hplot.setMinimumHeight(90)
        theme.title(hp, "Harmonic amplitudes")
        self.bars = pg.BarGraphItem(x=[1], height=[0], width=0.6,
                                    brush=pg.mkBrush(theme.ACCENT_LINE), pen=None)
        hp.addItem(self.bars)
        self.hy = YRange(hp, symmetric=False, floor=1e-3)
        self.hp = hp

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("REPETITION", "Section"), self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(330)

        # two rows: a channel box and two sliders do not fit one row at 1024 px
        top, _ = card(self.tplot, row(label("channel", "Caption"), self.pick.box, None),
                      row(self.span, self.harm, None))
        left, _ = card(self.aplot)
        right, _ = card(self.hplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addLayout(row(left, right), 2)
        self.body(row(col, side))
        self.slow = Every(4)
        self._session = None
        self._span = None
        self.f0, self.r, self.explained = 0.0, 0.0, 0.0

    def _picked(self):
        theme.axis_label(self.tp, "left", self.pick.name(), self.pick.unit())

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)
        self._picked()

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        span = self.span.value()
        d = session.last(span)
        if len(d) < 64 or not self.slow():
            return
        t = d[:, core.T]
        x = self.pick.get(d)
        hz = session.hz
        if span != self._span:
            self._span = span
            self.tp.setXRange(-span, 0, padding=0)
            self.ap.setXRange(0, span / 2.0, padding=0)
        # at least two periods must fit the window for the peak to be trusted
        self.f0, self.r, acf, lags = analysis.fundamental(x, hz, f_lo=2.0 / span,
                                                          f_hi=min(hz / 4.0, 30.0))
        k, amp, _ph, recon = analysis.fourier_series(t - t[0], x, self.f0,
                                                     int(self.harm.value()))
        self.ty.fit(x - x.mean(), recon - x.mean())
        draw(self.raw, t - t[-1], x - x.mean(), hz)
        draw(self.series, t - t[-1], recon - x.mean(), hz)
        keep = lags <= span / 2.0
        self.acf.setData(*decimate(lags[keep], acf[keep], limit=600))
        self.peak.setValue(1.0 / self.f0 if self.f0 > 0 else 0.0)
        self.bars.setOpts(x=k, height=amp)
        self.hy.fit(amp)
        self.hp.setXRange(0.4, len(k) + 0.6, padding=0)
        var = float(np.var(x))
        self.explained = 1.0 - float(np.var(x - recon)) / var if var > 0 else 0.0

        rows = [_cell("f₀", "%.2f" % self.f0, "Hz"),
                _cell("period", "%.0f" % (1000.0 / self.f0) if self.f0 > 0 else "&#8212;", "ms"),
                _cell("per minute", "%.0f" % (60.0 * self.f0), ""),
                _cell("r at the peak", "%.2f" % self.r, ""),
                _cell("explained by N", "%.0f" % (100.0 * max(self.explained, 0.0)), "%")]
        self.readout.setText("<table cellspacing=5>%s</table>" % "".join(rows))
        if self.r < 0.3 or self.f0 <= 0:
            self.state.setText("The autocorrelation has no clear peak: this window does "
                               "not repeat. Shake the board, or pick a channel that moves.")
        else:
            self.state.setText("Repeats every %.0f ms. The first %d harmonics account for "
                               "%.0f %% of its variance; the rest is noise and whatever "
                               "does not repeat." % (1000.0 / self.f0, int(self.harm.value()),
                                                     100.0 * max(self.explained, 0.0)))
        self.explain_changed.emit()

    def explain(self):
        return dict(
            head="Fourier series",
            equation="<span style='font-size:16pt'>x(t) &nbsp;&#8776;&nbsp; a<sub>0</sub> + "
                     "&#8721;<sub>k=1</sub><sup>N</sup> |c<sub>k</sub>| cos(2&#960; k "
                     "<span style='color:%s'>f<sub>0</sub></span> t + &#966;<sub>k</sub>)"
                     "</span>" % theme.ACCENT,
            terms=(("<span style='color:%s'>f<sub>0</sub></span>" % theme.ACCENT,
                    "the fundamental: the first peak of the autocorrelation, %.2f Hz now"
                    % self.f0),
                   ("c<sub>k</sub>", "the k-th harmonic, projected out of the window"),
                   ("N", "%d harmonics kept" % int(self.harm.value()))),
            why="A signal that repeats every T₀ is a sum of sinusoids at multiples of "
                "1/T₀. The autocorrelation finds T₀; projecting onto those multiples gives "
                "each harmonic's size.",
            tip="Shake the board and raise N one step at a time: the green series grows "
                "into the grey signal, harmonic by harmonic.",
            source=("Fourier series, Wikipedia", "https://en.wikipedia.org/wiki/Fourier_series"),
            sources=(("Autocorrelation, Wikipedia",
                      "https://en.wikipedia.org/wiki/Autocorrelation"),))


class SpectrogramPage(Page):
    title = "Spectrogram"
    subtitle = "The spectrum frame by frame: short-time Fourier transform"

    def __init__(self):
        super().__init__()
        self.pick = Picker(self._picked)
        self.win = Slider("frame", 50.0, 2000.0, 500.0, 50.0, "ms", 0, width=200)
        self.win.changed.connect(lambda _: self.explain_changed.emit())
        self.span = Slider("history", 5.0, 60.0, 20.0, 5.0, "s", 0, width=160)
        self.db = QtWidgets.QCheckBox("decibels")
        self.db.setChecked(True)

        self.plot = pg.PlotWidget()
        pi = theme.plot(self.plot, "seconds before now", "frequency", "", "Hz")
        theme.title(pi, "|X(t, f)|")
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("viridis"))
        pi.addItem(self.img)
        pi.setMouseEnabled(x=False, y=False)
        self.pi = pi

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("RESOLUTION", "Section"), self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(330)

        top, _ = card(self.plot, row(label("channel", "Caption"), self.pick.box, self.db, None),
                      row(self.win, self.span, None))
        self.body(row(top, side))
        self.slow = Every(10)
        self._session = None
        self._top = 0.0

    def _picked(self):
        pass

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        if not self.slow():
            return
        span = self.span.value()
        d = session.last(span)
        hz = session.hz
        window = int(max(16, min(self.win.value() * 1e-3 * hz, len(d) // 2)))
        if len(d) < 4 * window:
            return
        t = d[:, core.T]
        x = self.pick.get(d)
        freq, times, mag = analysis.stft(x, hz, window)
        if mag.size == 0:
            return
        if self.db.isChecked():
            S = 20.0 * np.log10(mag + 1e-9)
            top = float(S.max())
            levels = (top - 60.0, top)
        else:
            S = mag
            levels = (0.0, float(S.max()) or 1.0)
        self.img.setImage(S, levels=levels, autoLevels=False)
        x0 = float(times[0] - (t[-1] - t[0]))
        self.img.setRect(QtCore.QRectF(x0, 0.0, float(times[-1] - times[0]) or 1e-3,
                                       float(freq[-1])))
        self.pi.setXRange(-span, 0, padding=0)
        top_f = min(float(freq[-1]), 60.0 if session.modality != "emg" else float(freq[-1]))
        if top_f != self._top:
            self._top = top_f
            self.pi.setYRange(0, top_f, padding=0)
        last = mag[:, -1]
        strongest = float(freq[int(np.argmax(last[1:])) + 1]) if len(last) > 1 else 0.0
        rows = [_cell("frame", "%d" % window, "samples"),
                _cell("&#916;t", "%.0f" % (1000.0 * window / 4.0 / hz), "ms"),
                _cell("&#916;f", "%.2f" % (hz / window), "Hz"),
                _cell("strongest now", "%.1f" % strongest, "Hz")]
        self.readout.setText("<table cellspacing=5>%s</table>" % "".join(rows))
        self.state.setText("A frame of %d samples resolves %.2f Hz and %.0f ms. Lengthen "
                           "it and the bands sharpen while the changes blur; shorten it "
                           "and the reverse." % (window, hz / window, 1000.0 * window / hz))

    def explain(self):
        return dict(
            head="Short-time Fourier transform",
            equation="<span style='font-size:16pt'>X(m, k) &nbsp;=&nbsp; &#8721;<sub>n</sub> "
                     "x[n] <span style='color:%s'>w[n &minus; mH]</span> "
                     "e<sup>&minus;j2&#960;kn/N</sup></span>" % theme.ACCENT,
            terms=(("<span style='color:%s'>w</span>" % theme.ACCENT,
                    "a window of N samples, slid along by H"),
                   ("<span style='color:%s'>&#916;f &#183; &#916;t</span>" % theme.GUILTY,
                    "fixed: N samples cost N/f<sub>s</sub> of time and buy f<sub>s</sub>/N "
                    "of frequency")),
            why="One spectrum describes the whole record; a spectrogram describes it "
                "frame by frame. A frame can be short or sharp in frequency, not both.",
            tip="Pick the simulated muscle's fatigue pattern: the bright band walks down "
                "over forty seconds. On the IMU, sweep the shake.",
            source=("Short-time Fourier transform, Wikipedia",
                    "https://en.wikipedia.org/wiki/Short-time_Fourier_transform"),
            sources=(("Spectrogram, Wikipedia", "https://en.wikipedia.org/wiki/Spectrogram"),))


class DenoisePage(Page):
    title = "Denoise"
    subtitle = "Moving average, median, exponential, Savitzky–Golay"

    METHODS = ("moving average", "median", "exponential", "Savitzky–Golay")

    def __init__(self):
        super().__init__()
        self.pick = Picker(self._picked)
        self.method = QtWidgets.QComboBox()
        self.method.addItems(self.METHODS)
        self.method.currentIndexChanged.connect(self._changed)
        # two boxes on one row have to fit beside each other at 1024 px
        self.method.setStyleSheet("QComboBox { min-width: 118px; }")
        self.pick.box.setStyleSheet("QComboBox { min-width: 110px; }")
        self.win = Slider("window", 10.0, 500.0, 100.0, 10.0, "ms", 0, width=200)
        self.win.changed.connect(self._changed)
        self.order = Slider("polynomial order", 1, 5, 3, 1, "", 0, width=140)
        self.order.changed.connect(self._changed)

        self.tplot = pg.PlotWidget()
        self.tp = theme.plot(self.tplot, "seconds before now", "")
        theme.title(self.tp, "Before and after")
        theme.legend(self.tp)
        self.raw = self.tp.plot(pen=theme.pen(theme.LINE, 1.4), name="raw")
        self.out = self.tp.plot(pen=theme.pen(theme.ACCENT, 2.0), name="smoothed")
        self.ty = YRange(self.tp)

        self.rplot = pg.PlotWidget()
        rp = theme.plot(self.rplot, "seconds before now", "removed")
        theme.title(rp, "What was taken out: raw minus smoothed")
        self.res = rp.plot(pen=theme.pen(theme.WARN, 1.2))
        self.ry = YRange(rp)
        self.rp = rp

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("SMOOTHING", "Section"), self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(330)

        top, _ = card(self.tplot, row(label("method", "Caption"), self.method,
                                      label("channel", "Caption"), self.pick.box, None),
                      row(self.win, self.order, None))
        bottom, _ = card(self.rplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, side))
        self.slow = Every(3)
        self._session = None
        self._changed()

    def _picked(self):
        theme.axis_label(self.tp, "left", self.pick.name(), self.pick.unit())
        theme.axis_label(self.rp, "left", "removed", self.pick.unit())

    def _changed(self, *_):
        self.order.setVisible(self.method.currentText() == "Savitzky–Golay")
        self.explain_changed.emit()

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)
        self._picked()

    def smooth(self, x, hz):
        """The chosen smoother on x. Returns (smoothed, delay in samples if run live)."""
        w = int(max(3, round(self.win.value() * 1e-3 * hz))) | 1
        name = self.method.currentText()
        if name == "moving average":
            return np.convolve(x, np.ones(w) / w, mode="same"), (w - 1) / 2.0
        if name == "median":
            return analysis.moving_median(x, w), (w - 1) / 2.0
        if name == "exponential":
            alpha = 2.0 / (w + 1)
            f = dsp.Sos([[alpha, 0.0, 0.0, 1.0, -(1.0 - alpha), 0.0]])
            f.prime(x[0])
            return f.process(x), (1.0 / alpha) - 1.0
        order = min(int(self.order.value()), w - 1)
        return np.convolve(x, analysis.savgol_kernel(w, order), mode="same"), (w - 1) / 2.0

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        d = session.last(8.0)
        if len(d) < 32 or not self.slow():
            return
        t = d[:, core.T] - d[-1, core.T]
        x = self.pick.get(d)
        hz = session.hz
        y, delay = self.smooth(x, hz)
        r = x - y
        self.ty.fit(x, y)
        self.ry.fit(r)
        draw(self.raw, t, x, hz)
        draw(self.out, t, y, hz)
        draw(self.res, t, r, hz)
        rms_x, rms_r = float(np.sqrt(np.mean(x * x))), float(np.sqrt(np.mean(r * r)))
        w = int(max(3, round(self.win.value() * 1e-3 * hz))) | 1
        rows = [_cell("window", "%d" % w, "samples"),
                _cell("removed, RMS", "%.3g" % rms_r, self.pick.unit()),
                _cell("kept, RMS", "%.3g" % float(np.sqrt(np.mean(y * y))), self.pick.unit()),
                _cell("delay if live", "%.0f" % (1000.0 * delay / hz), "ms")]
        self.readout.setText("<table cellspacing=5>%s</table>" % "".join(rows))
        if self.method.currentText() == "median":
            self.state.setText("The median drops a spike shorter than half the window "
                               "without rounding the edges around it. The average rounds "
                               "everything.")
        elif self.method.currentText() == "exponential":
            self.state.setText("One multiply per sample and no memory of the past beyond "
                               "the last output, which is why a microcontroller uses it. "
                               "It lags by about the window length.")
        elif self.method.currentText() == "Savitzky–Golay":
            self.state.setText("A polynomial through each window keeps peaks the moving "
                               "average would flatten. Order 0 is the moving average.")
        else:
            self.state.setText("Every sample replaced by the mean of its neighbours. "
                               "Drawn centred here; run live it comes out half a window "
                               "late.")

    def explain(self):
        name = self.method.currentText()
        eq = {"moving average": "y[n] = (1/W) &#8721;<sub>k&lt;W</sub> x[n&minus;k]",
              "median": "y[n] = median( x[n&minus;h .. n+h] )",
              "exponential": "y[n] = &#945; x[n] + (1 &minus; &#945;) y[n&minus;1]",
              "Savitzky–Golay": "y[n] = &#8721;<sub>k</sub> h[k] x[n&minus;k], &nbsp;h from a "
                                "least-squares polynomial"}[name]
        return dict(
            head=name,
            equation="<span style='font-size:16pt'>%s</span>" % eq,
            terms=(("W", "the window, %.0f ms" % self.win.value()),
                   ("<span style='color:%s'>delay</span>" % theme.GUILTY,
                    "half a window for the symmetric ones; run live, that is what it costs")),
            why="Noise changes faster than the signal, so averaging neighbours removes "
                "more noise than signal. Each smoother makes a different bet about what "
                "the signal looks like inside a window.",
            tip="Put a tap into the simulated board and compare: the average smears the "
                "tap, the median keeps its edges, Savitzky–Golay keeps its height.",
            source=("Moving average, Wikipedia", "https://en.wikipedia.org/wiki/Moving_average"),
            sources=(("Savitzky–Golay filter, Wikipedia",
                      "https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter"),
                     ("Median filter, Wikipedia",
                      "https://en.wikipedia.org/wiki/Median_filter")))


class SeparationPage(Page):
    title = "Separation"
    subtitle = "Principal and independent components of the channels"

    def __init__(self):
        super().__init__()
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["PCA", "ICA"])
        self.method.currentIndexChanged.connect(lambda _: self.explain_changed.emit())
        self.which = QtWidgets.QComboBox()             # only for the IMU: which triple
        self.which.addItems(["accelerometer", "gyroscope"])
        self.span = Slider("window", 2.0, 20.0, 8.0, 0.5, "s", 1, width=180)

        self.inplot = pg.PlotWidget()
        ip = theme.plot(self.inplot, "seconds before now", "channels")
        theme.title(ip, "The channels, as measured")
        theme.legend(ip)
        self.ip = ip
        self.outplot = pg.PlotWidget()
        op = theme.plot(self.outplot, "seconds before now", "components")
        theme.title(op, "The components")
        theme.legend(op)
        self.op = op
        self.in_curves, self.out_curves = [], []
        self.iy = YRange(ip)
        self.oy = YRange(op)

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("COMPONENTS", "Section"), self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(340)

        top, _ = card(self.inplot, row(label("method", "Caption"), self.method, None),
                      row(label("triple", "Caption"), self.which, self.span, None))
        bottom, _ = card(self.outplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 1)
        col.addWidget(bottom, 1)
        self.body(row(col, side))
        self.slow = Every(6)
        self._session = None
        self._n = 0

    def on_show(self, session, source):
        self._session = session
        self.which.setVisible(session.modality == "imu")
        self._make_curves(self._names(session))

    def _names(self, session):
        if session.modality == "imu":
            return ["ax", "ay", "az"] if self.which.currentIndex() == 0 else ["gx", "gy", "gz"]
        return [c.name for c in session.channels]

    def _matrix(self, session, d):
        if session.modality == "imu":
            cols = d[:, core.A] * core.G if self.which.currentIndex() == 0 else d[:, core.W]
            return np.asarray(cols)
        return session.raw(d)

    def _make_curves(self, names):
        names = list(names)
        if names == self._n:
            return
        self._n = names
        self.ip.clear()
        self.op.clear()
        theme.legend(self.ip)
        theme.legend(self.op)
        cols = (theme.X_AXIS, theme.Y_AXIS, theme.Z_AXIS) + theme.SERIES
        self.in_curves = [self.ip.plot(pen=theme.pen(cols[i % len(cols)], 1.3), name=nm)
                          for i, nm in enumerate(names)]
        self.out_curves = [self.op.plot(pen=theme.pen(theme.SERIES[i % 5], 1.5),
                                        name="component %d" % (i + 1))
                           for i in range(len(names))]

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        span = self.span.value()
        d = session.last(span)
        names = self._names(session)
        self._make_curves(names)
        if len(d) < 64 or len(names) < 2 or not self.slow():
            if len(names) < 2:
                self.state.setText("Separation needs at least two channels; this source "
                                   "has one.")
            return
        t = d[:, core.T] - d[-1, core.T]
        X = self._matrix(session, d)
        hz = session.hz
        ica = self.method.currentText() == "ICA"
        if ica:
            comps, W = analysis.fastica(X)
            shares = [analysis.kurtosis(comps[:, j]) for j in range(comps.shape[1])]
        else:
            comps, V, share = analysis.pca(X)
            shares = list(100.0 * share)
        Xc = X - X.mean(axis=0)
        for j, c in enumerate(self.in_curves):
            draw(c, t, Xc[:, j], hz)
        for j, c in enumerate(self.out_curves):
            draw(c, t, comps[:, j], hz)
        self.iy.fit(Xc)
        self.oy.fit(comps)

        rows = []
        for j, s in enumerate(shares):
            rows.append(_cell("component %d" % (j + 1),
                              ("%.2f" % s) if ica else ("%.0f" % s), "kurtosis" if ica else "%"))
        truth = None
        if source is not None and hasattr(source, "true_sources"):
            truth = source.true_sources(d[:, core.T])
        if truth is not None and truth.shape[1] >= 2:
            for j, (i, r) in enumerate(analysis.match(comps, truth)):
                rows.append(_cell("&nbsp;&nbsp;matches source %d" % (i + 1), "%.2f" % r, "|r|"))
        self.readout.setText("<table cellspacing=5>%s</table>" % "".join(rows))
        if ica:
            self.state.setText("ICA turns the whitened axes until each component is as "
                               "far from Gaussian as it can be. Two sources added together "
                               "come apart; two Gaussian sources cannot.")
        else:
            self.state.setText("PCA rotates the axes so that the first component carries "
                               "the most variance. On a shaken board it is the shake "
                               "direction, whatever the board's own axes.")

    def explain(self):
        ica = self.method.currentText() == "ICA"
        if ica:
            return dict(
                head="Independent component analysis",
                equation="<span style='font-size:16pt'>x = A s &nbsp;&nbsp;&#8658;&nbsp;&nbsp; "
                         "&#349; = W x, &nbsp;W chosen so the &#349;<sub>i</sub> are "
                         "<span style='color:%s'>least Gaussian</span></span>" % theme.ACCENT,
                terms=(("A", "the unknown mixing"),
                       ("W", "the unmixing found by FastICA"),
                       ("<span style='color:%s'>Gaussian sources</span>" % theme.GUILTY,
                        "cannot be separated: every rotation of them looks the same")),
                why="A sum of independent signals is closer to Gaussian than either of "
                    "them. Undo the sum by finding the rotation whose outputs are least "
                    "Gaussian.",
                tip="Simulate two muscles with crosstalk: the inputs both show both bursts, "
                    "the components show one each. Then try PCA on the same data.",
                source=("Independent component analysis, Wikipedia",
                        "https://en.wikipedia.org/wiki/Independent_component_analysis"),
                sources=(("FastICA, Wikipedia", "https://en.wikipedia.org/wiki/FastICA"),))
        return dict(
            head="Principal component analysis",
            equation="<span style='font-size:16pt'>C = X<sup>T</sup>X / N, &nbsp;&nbsp;"
                     "C v<sub>i</sub> = &#955;<sub>i</sub> v<sub>i</sub>, &nbsp;&nbsp;"
                     "p<sub>i</sub> = X v<sub>i</sub></span>",
            terms=(("C", "the covariance of the channels"),
                   ("v<sub>i</sub>, &#955;<sub>i</sub>",
                    "its eigenvectors and eigenvalues, largest first"),
                   ("<span style='color:%s'>&#955;<sub>i</sub> / &#8721;&#955;</span>"
                    % theme.ACCENT, "the share of variance in component i")),
            why="The eigenvectors of the covariance are the directions the data actually "
                "spread along. The first is the dominant motion, in whatever frame the "
                "sensor happens to be in.",
            tip="Shake the simulated board along x, then tilt it: the axes swap shares "
                "but the first component stays the shake.",
            source=("Principal component analysis, Wikipedia",
                    "https://en.wikipedia.org/wiki/Principal_component_analysis"))
