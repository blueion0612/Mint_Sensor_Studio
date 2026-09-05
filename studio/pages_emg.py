# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_emg.py
The two pages for a muscle.

    Envelope   the amplitude: rectify, average, and decide whether it is on
    Fatigue    the spectrum: where its median sits, and how it slides down

Both read the columns EmgSession has already worked out, so the envelope on
this page is the envelope everywhere else, and both draw the simulator's truth
beside the estimate when there is one.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import core, dsp, theme
from .shell import Every, Page, Slider, YRange, card, decimate, draw, label, row

LOOK = 8.0             # seconds on the fast trace
TREND = 30.0           # seconds on the slow one


def _channel_box(session, box):
    names = [c.name for c in session.channels]
    if [box.itemText(i) for i in range(box.count())] != names:
        box.blockSignals(True)
        box.clear()
        box.addItems(names)
        box.blockSignals(False)


class EnvelopePage(Page):
    title = "Envelope"
    subtitle = "Rectified, averaged, and thresholded"
    modalities = ("emg",)

    def __init__(self):
        super().__init__()
        self.channel = QtWidgets.QComboBox()
        self.window = Slider("RMS window", 20.0, 500.0, 100.0, 10.0, "ms", 0, width=200)
        self.window.changed.connect(self._window_changed)
        self.k = Slider("onset at", 1.0, 10.0, 4.0, 0.5, "x rest", 1, width=160)
        self.k.changed.connect(lambda _: self.explain_changed.emit())
        self.rest_btn = QtWidgets.QPushButton("Set rest")
        self.rest_btn.setObjectName("Primary")
        self.rest_btn.setToolTip("Take the last two seconds as the resting level")
        self.rest_btn.clicked.connect(self._set_rest)
        self.mvc_btn = QtWidgets.QPushButton("Set MVC")
        self.mvc_btn.setToolTip("Take the strongest of the last three seconds as 100 %")
        self.mvc_btn.clicked.connect(self._set_mvc)

        self.fast = pg.PlotWidget()
        fp = theme.plot(self.fast, "seconds before now", "voltage", "", "mV")
        theme.title(fp, "Filtered EMG, and its RMS envelope")
        fp.setXRange(-LOOK, 0, padding=0)
        theme.legend(fp)
        self.raw = fp.plot(pen=theme.pen(theme.LINE, 1.0), name="filtered")
        self.env = fp.plot(pen=theme.pen(theme.MODALITY["emg"], 2.2), name="RMS envelope")
        self.thresh = pg.InfiniteLine(angle=0, pen=theme.pen(theme.GUILTY, 1.2, dash=True),
                                      label="onset {value:.3f} mV",
                                      labelOpts={"position": 0.06, "color": theme.GUILTY})
        fp.addItem(self.thresh)
        self.fy = YRange(fp, floor=0.05)

        self.slowp = pg.PlotWidget()
        sp = theme.plot(self.slowp, "seconds before now", "RMS", "", "mV")
        theme.title(sp, "The envelope over the last half minute")
        sp.setXRange(-TREND, 0, padding=0)
        theme.legend(sp)
        self.trend = sp.plot(pen=theme.pen(theme.MODALITY["emg"], 1.8), name="estimate")
        self.truth = sp.plot(pen=theme.pen(theme.TRUTH, 1.6, dash=True), name="truth")
        self.sy = YRange(sp, symmetric=False, floor=0.05)

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.state = label("", "Caption", wrap=True)

        side, sl = card(label("AMPLITUDE", "Section"), self.readout,
                        label("of maximum voluntary contraction", "Caption"), self.bar,
                        self.state)
        sl.addStretch(1)
        sl.addLayout(row(self.rest_btn, self.mvc_btn, None))
        side.setFixedWidth(360)

        # Two rows, not one: at the smallest window the layout is designed for
        # a channel box and two sliders do not fit beside each other.
        top, _ = card(self.fast, row(label("channel", "Caption"), self.channel, None),
                      row(self.window, self.k, None))
        bottom, _ = card(self.slowp)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, side))
        self.slow = Every(6)
        self._session = None
        self.rest = None            # (mean, std) of the RMS at rest, in mV
        self.mvc = None             # the RMS of a full contraction
        self._on_since = None
        self._note = ""

    # ---- controls ----
    def _window_changed(self, v):
        if self._session is not None and hasattr(self._session, "set_rms_window"):
            self._session.set_rms_window(v)
        self.explain_changed.emit()

    def _rms(self, session, seconds):
        d = session.last(seconds)
        if len(d) < 4:
            return None
        return d[:, session.rms_cols][:, self.channel.currentIndex()]

    def _set_rest(self):
        r = self._rms(self._session, 2.0) if self._session is not None else None
        if r is None:
            return
        r30 = self._rms(self._session, TREND)
        # Pressed during a contraction, this would put the onset above the
        # contraction itself and the page would never call anything active.
        if r30 is not None and float(r.mean()) > 0.25 * float(r30.max()):
            self._note = ("That was not rest: the last two seconds were %.0f %% of the "
                          "strongest of the last half minute. Relax and press it again."
                          % (100.0 * float(r.mean()) / float(r30.max())))
            return
        self._note = ""
        self.rest = (float(r.mean()), float(r.std()))

    def _set_mvc(self):
        r = self._rms(self._session, 3.0) if self._session is not None else None
        if r is not None:
            self.mvc = float(r.max())

    def on_show(self, session, source):
        if session is not self._session:
            self.rest = None
            self.mvc = None
        self._session = session
        _channel_box(session, self.channel)
        if hasattr(session, "set_rms_window"):
            session.set_rms_window(self.window.value())
        self.explain_changed.emit()

    def threshold(self, r30):
        """The onset level: rest mean plus k standard deviations, rest measured or guessed."""
        if self.rest is not None:
            mean, std = self.rest
        else:
            # guessed from the quietest tenth of the last half minute
            q = np.percentile(r30, 10)
            quiet = r30[r30 <= np.percentile(r30, 25)]
            mean, std = float(q), float(quiet.std()) if len(quiet) > 2 else 0.0
        return mean + self.k.value() * max(std, 0.002)

    # ---- the frame ----
    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        d = session.last(LOOK)
        if len(d) < 8:
            return
        ch = self.channel.currentIndex()
        t = d[:, core.T] - d[-1, core.T]
        filt = d[:, session.filt_cols][:, ch]
        rms = d[:, session.rms_cols][:, ch]
        draw(self.raw, t, filt, session.hz)
        draw(self.env, t, rms, session.hz, 400)
        self.fy.fit(filt, rms)

        d30 = session.last(TREND)
        t30 = d30[:, core.T]
        r30 = d30[:, session.rms_cols][:, ch]
        level = self.threshold(r30)
        self.thresh.setValue(level)
        self.trend.setData(*decimate(t30 - t30[-1], r30, limit=600))
        self.sy.fit(r30)
        if source is not None and hasattr(source, "true_envelope"):
            tt, _ = decimate(t30, r30, limit=600)
            self.truth.setData(tt - t30[-1], source.true_envelope(tt))
        else:
            self.truth.setData([], [])

        now = float(rms[-1])
        active = now > level
        if active and self._on_since is None:
            self._on_since = d[-1, core.T]
        if not active:
            self._on_since = None
        if not self.slow():
            return

        w = int(self.window.value() * 1e-3 * session.hz)
        mav = float(np.abs(filt[-w:]).mean()) if w else 0.0
        pct = 100.0 * now / self.mvc if self.mvc else 0.0
        self.bar.setValue(int(min(100.0, pct)))
        faint = theme.INK_FAINT
        self.readout.setText(
            "<table cellspacing=5>"
            "<tr><td style='color:%s'>RMS</td><td align=right><b>%.3f</b></td>"
            "<td style='color:%s'>mV</td></tr>"
            "<tr><td style='color:%s'>MAV</td><td align=right>%.3f</td>"
            "<td style='color:%s'>mV</td></tr>"
            "<tr><td style='color:%s'>of MVC</td><td align=right>%s</td>"
            "<td style='color:%s'>%%</td></tr>"
            "<tr><td colspan=3><hr></td></tr>"
            "<tr><td style='color:%s'>rest</td><td align=right>%s</td>"
            "<td style='color:%s'>mV</td></tr>"
            "<tr><td style='color:%s'>onset</td><td align=right>%.3f</td>"
            "<td style='color:%s'>mV</td></tr></table>"
            % (faint, now, faint, faint, mav, faint, faint,
               ("%.0f" % pct) if self.mvc else "&#8212;", faint, faint,
               ("%.3f" % self.rest[0]) if self.rest else "guessed", faint,
               faint, level, faint))
        bits = []
        if self._note:
            bits.append("<span style='color:%s'>%s</span>" % (theme.WARN, self._note))
        if active:
            held = d[-1, core.T] - (self._on_since or d[-1, core.T])
            bits.append("<span style='color:%s'><b>Active</b> for %.1f s.</span>"
                        % (theme.MODALITY["emg"], held))
        else:
            bits.append("Resting.")
        if self.rest is None:
            bits.append("Rest level guessed from the quietest tenth of the last half "
                        "minute. Relax and press <b>Set rest</b> to measure it.")
        if self.mvc is None:
            bits.append("Contract hard for three seconds and press <b>Set MVC</b> to read "
                        "amplitude as a percentage.")
        if source is not None and hasattr(source, "true_active"):
            lag = self._onset_lag(t30, r30 > level, source.true_active(t30))
            if lag is not None:
                bits.append("<span style='color:%s'>&#8212;</span> the true envelope. "
                            "Detected onsets lag the true ones by <b>%.0f ms</b>, which is "
                            "the RMS window catching up." % (theme.TRUTH, lag * 1000))
        self.state.setText("<br>".join(bits))

    @staticmethod
    def _onset_lag(t, detected, truth):
        """Mean delay from a true onset to the detected one that follows it."""
        rise_true = np.flatnonzero(np.diff(truth.astype(int)) > 0)
        rise_det = np.flatnonzero(np.diff(detected.astype(int)) > 0)
        if not len(rise_true) or not len(rise_det):
            return None
        lags = []
        for i in rise_true:
            later = rise_det[(t[rise_det] >= t[i] - 0.05) & (t[rise_det] <= t[i] + 0.6)]
            if len(later):
                lags.append(t[later[0]] - t[i])
        return float(np.mean(lags)) if lags else None

    def explain(self):
        w = self.window.value()
        return dict(
            head="RMS envelope and onset",
            equation="<span style='font-size:16pt'>RMS[n] &nbsp;=&nbsp; "
                     "&#8730;( (1/W) &#8721;<sub>k&lt;W</sub> x[n&minus;k]<sup>2</sup> )"
                     "&nbsp;&nbsp;&nbsp;&nbsp; active &nbsp;&#8660;&nbsp; RMS &gt; "
                     "&#956;<sub>rest</sub> + <span style='color:%s'>k</span> "
                     "&#963;<sub>rest</sub></span>" % theme.ACCENT,
            terms=(("W", "the window, %.0f ms here, so %d samples at this rate"
                    % (w, int(w * 1e-3 * (self._session.hz if self._session else 1000)))),
                   ("&#956;<sub>rest</sub>, &#963;<sub>rest</sub>",
                    "the resting level and its spread"),
                   ("<span style='color:%s'>k</span>" % theme.ACCENT,
                    "%.1f: how far above rest counts as on" % self.k.value())),
            why="Force is read from amplitude, and the amplitude of a noise-like signal "
                "is its RMS. A short window follows closely and wobbles; a long one is "
                "steady and late.",
            tip="Set the window to 20 ms and watch the envelope shiver; set it to 500 and "
                "watch each burst arrive late.",
            source=("Electromyography, Wikipedia",
                    "https://en.wikipedia.org/wiki/Electromyography"),
            sources=(("Root mean square, Wikipedia",
                      "https://en.wikipedia.org/wiki/Root_mean_square"),
                     ("Hodges & Bui (1996), EMG onset detection",
                      "https://doi.org/10.1016/S0921-884X(96)95190-5")))


class FatiguePage(Page):
    title = "Fatigue"
    subtitle = "Median frequency of the power spectrum, and how it falls"
    modalities = ("emg",)

    def __init__(self):
        super().__init__()
        self.channel = QtWidgets.QComboBox()
        self.span = Slider("analysis window", 0.5, 2.0, 1.0, 0.25, "s", 2, width=180)
        self.span.changed.connect(lambda _: self.explain_changed.emit())

        self.fplot = pg.PlotWidget()
        self.fp = theme.plot(self.fplot, "frequency", "power", "Hz", "")
        theme.title(self.fp, "Power spectrum of the last window")
        self.psd = self.fp.plot(pen=theme.pen(theme.MODALITY["emg"], 1.6), fillLevel=0,
                                brush=theme.fill(theme.MODALITY["emg"], 40))
        self.mdf_line = pg.InfiniteLine(angle=90, pen=theme.pen(theme.GUILTY, 1.4, dash=True),
                                        label="MDF {value:.0f} Hz",
                                        labelOpts={"position": 0.9, "color": theme.GUILTY})
        self.mnf_line = pg.InfiniteLine(angle=90, pen=theme.pen(theme.Z_AXIS, 1.2, dash=True),
                                        label="MNF {value:.0f} Hz",
                                        labelOpts={"position": 0.8, "color": theme.Z_AXIS})
        self.fp.addItem(self.mdf_line)
        self.fp.addItem(self.mnf_line)
        self.py = YRange(self.fp, symmetric=False, floor=1e-6)
        self._fx = None

        self.tplot = pg.PlotWidget()
        tp = theme.plot(self.tplot, "seconds before now", "frequency", "", "Hz")
        theme.title(tp, "Where the spectrum sits, over time")
        tp.setXRange(-60.0, 0, padding=0)
        theme.legend(tp, offset=(10, -10))          # bottom left: the data lives up top
        self.mdf_curve = tp.plot(pen=theme.pen(theme.GUILTY, 1.8), symbol="o", symbolSize=4,
                                 symbolBrush=theme.GUILTY, symbolPen=None, name="MDF")
        self.mnf_curve = tp.plot(pen=theme.pen(theme.Z_AXIS, 1.4), name="MNF")
        self.truth = tp.plot(pen=theme.pen(theme.TRUTH, 1.6, dash=True), name="true centre")
        self.fit = tp.plot(pen=theme.pen(theme.INK_SOFT, 1.2, dash=True), name="trend")
        tp.setYRange(0, 250, padding=0)
        tp.disableAutoRange(axis="y")

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("SPECTRUM", "Section"), self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(360)

        top, _ = card(self.fplot, row(label("channel", "Caption"), self.channel, None),
                      row(self.span, None))
        bottom, _ = card(self.tplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, side))
        self._session = None
        self._points = []           # (t, mdf, mnf)
        self._last = None
        self.slow = Every(4)

    def on_show(self, session, source):
        if session is not self._session:
            self._points = []
            self._last = None
        self._session = session
        _channel_box(session, self.channel)
        self.explain_changed.emit()

    def on_hide(self):
        self._points = []
        self._last = None

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        span = self.span.value()
        d = session.last(span)
        if len(d) < 32:
            return
        ch = self.channel.currentIndex()
        now = float(d[-1, core.T])
        if self._last is not None and now - self._last < 0.5:
            return
        self._last = now
        filt = d[:, session.filt_cols][:, ch]
        rms = d[:, session.rms_cols][:, ch]
        d30 = session.last(TREND)
        r30 = d30[:, session.rms_cols][:, ch]
        # Only a contracting muscle has a spectrum worth reading; at rest the
        # window is electrode noise and its median is meaningless.
        gate = max(3.0 * float(np.percentile(r30, 10)), 0.02)
        active = float(rms.mean()) > gate
        mdf, mnf, freq, psd = dsp.median_frequency(filt, session.hz)
        top_f = round(freq[-1])
        if top_f != self._fx:
            self._fx = top_f
            self.fp.setXRange(0, min(top_f, 500), padding=0)
        self.psd.setData(*decimate(freq, psd, limit=1200))
        self.py.fit(psd)
        self.mdf_line.setValue(mdf if active else 0.0)
        self.mnf_line.setValue(mnf if active else 0.0)
        if active:
            self._points.append((now, mdf, mnf))
            self._points = [p for p in self._points if p[0] > now - 60.0]

        pts = np.asarray(self._points, float) if self._points else np.zeros((0, 3))
        if len(pts):
            self.mdf_curve.setData(pts[:, 0] - now, pts[:, 1])
            self.mnf_curve.setData(pts[:, 0] - now, pts[:, 2])
        slope = None
        if len(pts) >= 6:
            recent = pts[pts[:, 0] > now - 30.0]
            if len(recent) >= 6:
                a, b = np.polyfit(recent[:, 0] - now, recent[:, 1], 1)
                slope = a
                xs = np.array([recent[0, 0] - now, 0.0])
                self.fit.setData(xs, a * xs + b)
        if source is not None and hasattr(source, "true_centre") and len(pts):
            self.truth.setData(pts[:, 0] - now, [source.true_centre(p) for p in pts[:, 0]])
        else:
            self.truth.setData([], [])

        faint = theme.INK_FAINT
        self.readout.setText(
            "<table cellspacing=5>"
            "<tr><td style='color:%s'>median</td><td align=right><b>%s</b></td>"
            "<td style='color:%s'>Hz</td></tr>"
            "<tr><td style='color:%s'>mean</td><td align=right>%s</td>"
            "<td style='color:%s'>Hz</td></tr>"
            "<tr><td style='color:%s'>trend</td><td align=right>%s</td>"
            "<td style='color:%s'>Hz/s</td></tr>"
            "<tr><td style='color:%s'>RMS</td><td align=right>%.3f</td>"
            "<td style='color:%s'>mV</td></tr></table>"
            % (faint, ("%.0f" % mdf) if active else "&#8212;", faint,
               faint, ("%.0f" % mnf) if active else "&#8212;", faint,
               faint, ("%+.2f" % slope) if slope is not None else "&#8212;", faint,
               faint, float(rms.mean()), faint))
        if not active:
            self.state.setText("The muscle is at rest, so the spectrum is the electrode's "
                               "noise and the median says nothing. Contract and hold.")
        elif slope is not None and slope < -0.3:
            self.state.setText("<span style='color:%s'>The median is falling at %.1f Hz "
                               "a second. The muscle is tiring: its fibres conduct more "
                               "slowly, and the whole spectrum shifts down.</span>"
                               % (theme.WARN, -slope))
        else:
            self.state.setText("Holding. The median frequency of a fresh muscle sits "
                               "between 80 and 150 Hz and stays there.")

    def explain(self):
        return dict(
            head="Median frequency",
            equation="<span style='font-size:16pt'>&#8747;<sub>0</sub><sup>MDF</sup> P(f) df "
                     "&nbsp;=&nbsp; &#189; &#8747;<sub>0</sub><sup>&#8734;</sup> P(f) df"
                     "&nbsp;&nbsp;&nbsp;&nbsp; MNF &nbsp;=&nbsp; &#8747; f P(f) df / "
                     "&#8747; P(f) df</span>",
            terms=(("P(f)", "the power spectrum of the window, %.2f s of it" % self.span.value()),
                   ("MDF", "the frequency that splits the power in half"),
                   ("<span style='color:%s'>slope</span>" % theme.GUILTY,
                    "how fast it falls, in Hz per second, which is the fatigue index")),
            why="A tiring muscle's fibres conduct more slowly, so its spectrum slides "
                "down. The median frequency measures that, and unlike amplitude it does "
                "not depend on how hard the electrodes are pressed.",
            tip="Choose the simulated muscle's fatigue pattern and watch the red dots walk "
                "down over forty seconds.",
            source=("Muscle fatigue, Wikipedia",
                    "https://en.wikipedia.org/wiki/Muscle_fatigue"),
            sources=(("Spectral density, Wikipedia",
                      "https://en.wikipedia.org/wiki/Spectral_density"),
                     ("De Luca (1997), The use of surface EMG in biomechanics",
                      "https://doi.org/10.1123/jab.13.2.135")))
