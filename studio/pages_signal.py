# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_signal.py
The two pages about the measurements themselves, before anything is done to them.

    Signals    every channel the source has, live, whatever the sensor is
    Sampling   how often they really arrive, and what is lost by asking less often

The Signals page is the one page that does not care what is plugged in. It
asks the session what its channels are, draws one plot per group of them, and
explains whichever sensor it turns out to be.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import core, theme
from .shell import (Band, Every, Page, Picker, Slider, YRange, card, clear_layout,
                    decimate, draw, label, row)

VIEW = 10.0            # seconds of history on the live traces

# what a plot of each group of channels is called
GROUP_TITLES = {
    "accelerometer": "Accelerometer  —  specific force",
    "gyroscope": "Gyroscope  —  angular rate",
    "magnetometer": "Magnetometer  —  magnetic field",
    "muscle": "Surface EMG  —  electrode voltage",
    "light": "Light received  —  photodiode",
    "signal": "Signal",
}
GROUP_YLABEL = {
    "accelerometer": "acceleration", "gyroscope": "rate", "magnetometer": "field",
    "muscle": "voltage", "light": "level", "signal": "value",
}


def trace_plot(title, y_label, y_unit, colours, names, centred=False):
    w = pg.PlotWidget()
    pi = theme.plot(w, "", y_label, "", y_unit)
    theme.title(pi, title)
    pi.setXRange(-VIEW, 0, padding=0)
    pi.setMouseEnabled(x=False, y=True)
    theme.axis_label(pi, "bottom", "seconds before now")
    curves = [pi.plot(pen=theme.pen(c, 1.4), name=n) for c, n in zip(colours, names)]
    rng = Band(pi) if centred else YRange(pi)
    return w, pi, curves, rng


class SignalsPage(Page):
    title = "Signals"
    subtitle = "Accelerometer, gyroscope, magnetometer"

    def __init__(self):
        super().__init__()
        self.left, self.left_lay = card()
        self.units = QtWidgets.QComboBox()
        self.units.addItems(["m/s²", "g"])
        self.units.setStyleSheet("QComboBox { min-width: 78px; }")
        self.units.currentIndexChanged.connect(self._units_changed)
        self.units_label = label("shown in", "Caption")

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.note = label("", "Caption", wrap=True)
        self.key = label("")
        self.key.setTextFormat(QtCore.Qt.RichText)

        panel, side = card(
            label("READING NOW", "Section"),
            self.readout,
            row(self.key, None, self.units_label, self.units),
            self.note,
        )
        side.addStretch(1)
        panel.setFixedWidth(348)

        self.body(row(self.left, panel))
        self.slow = Every(8)
        # The magnetometer is a 10 Hz sensor. Redrawing it twenty-two times a
        # second draws the same three curves again and again, and three curves
        # cost about eight milliseconds of a forty-five millisecond frame.
        self.mag_slow = Every(4)
        self._sig = None
        self._span = None
        self.plots = []                # one entry per group of channels
        self._kind = "imu"
        self._session = None
        self.configure(core.ImuSession(hz=100.0))

    # ---- following the source ----
    def configure(self, session):
        """Build the plots for whatever channels this session has."""
        sig = (session.modality, tuple(c.key for c in session.channels))
        if sig == self._sig:
            return
        self._sig = sig
        self._kind = session.modality
        self._span = None
        clear_layout(self.left_lay)
        self.plots = []
        groups = []
        for i, c in enumerate(session.channels):
            if c.group not in groups:
                groups.append(c.group)
        for g in groups:
            idx = [i for i, c in enumerate(session.channels) if c.group == g]
            chans = [session.channels[i] for i in idx]
            w, pi, curves, rng = trace_plot(
                GROUP_TITLES.get(g, g), GROUP_YLABEL.get(g, "value"), chans[0].unit,
                [c.colour for c in chans], [c.name for c in chans], centred=(g == "light"))
            self.left_lay.addWidget(w)
            self.plots.append(dict(group=g, pi=pi, curves=curves, rng=rng, idx=idx,
                                   unit=chans[0].unit))
        imu = session.modality == "imu"
        self.units.setVisible(imu)
        self.units_label.setVisible(imu)
        if imu:
            self._units_changed()
        if imu:
            # one key for the three plots: the axes share their colors
            shown = [(colour, name) for colour, name in
                     zip((theme.X_AXIS, theme.Y_AXIS, theme.Z_AXIS), "xyz")]
        else:
            shown = [(c.colour, c.name) for c in session.channels[:6]]
        self.key.setText("&nbsp;&nbsp;".join(
            "<span style='color:%s'>&#9632;</span>&nbsp;%s" % pair for pair in shown))
        subtitle = {"imu": "Accelerometer, gyroscope, magnetometer",
                    "emg": "Electrode voltage, as it arrives",
                    "ppg": "Light received at each wavelength"}.get(
            session.modality, "Every channel the source sends")
        self.set_subtitle(subtitle)
        self.readout.setText("")
        self.note.setText("")

    def _units_changed(self):
        for p in self.plots:
            if p["group"] == "accelerometer":
                theme.axis_label(p["pi"], "left", "acceleration", self.units.currentText())

    def on_show(self, session, source):
        self._session = session
        self.configure(session)
        self.explain_changed.emit()

    # ---- the frame ----
    def tick(self, session, source):
        if session is not self._session:
            self._session = session
            self.configure(session)
        d = session.last(VIEW)
        if len(d) < 2:
            return
        t_all = d[:, core.T] - d[-1, core.T]
        span = max(2.0, min(VIEW, round(float(t_all[-1] - t_all[0]) * 2) / 2))
        if span != self._span:
            self._span = span
            for p in self.plots:
                p["pi"].setXRange(-span, 0, padding=0)
        imu = session.modality == "imu"
        k = (1.0 if self.units.currentText() == "g" else core.G) if imu else 1.0
        fast = session.hz > 400.0
        for p in self.plots:
            if p["group"] == "magnetometer" and not self.mag_slow():
                continue
            scale = k if p["group"] == "accelerometer" else 1.0
            cols = []
            for curve, i in zip(p["curves"], p["idx"]):
                y = d[:, 1 + i] * scale
                draw(curve, t_all, y, session.hz, 300 if fast else 250)
                cols.append(y)
            if cols:
                if isinstance(p["rng"], Band):
                    p["rng"].fit(np.concatenate(cols))
                else:
                    p["rng"].fit(*cols)
        if not self.slow():
            return
        self._readout(session, d, k)

    def _readout(self, session, d, k):
        last = d[-1]
        faint = theme.INK_FAINT
        rows = []
        for p in self.plots:
            vals = [last[1 + i] * (k if p["group"] == "accelerometer" else 1.0)
                    for i in p["idx"]]
            unit = self.units.currentText() if p["group"] == "accelerometer" else p["unit"]
            name = {"accelerometer": "accel", "gyroscope": "gyro", "magnetometer": "mag",
                    "muscle": "emg", "light": "light"}.get(p["group"], p["group"])
            fmt = "%7.3f" if abs(max(vals, key=abs)) < 100 else "%9.1f"
            cells = "".join("<td align=right>%s</td>" % (fmt % v) for v in vals)
            rows.append("<tr><td style='color:%s'>%s</td>%s<td style='color:%s'>&nbsp;%s</td></tr>"
                        % (faint, name, cells, faint, unit))
        extra = ""
        if session.modality == "imu":
            a = last[core.A]
            na = float(np.linalg.norm(a))
            extra = ("<tr><td colspan=5><hr></td></tr>"
                     "<tr><td style='color:%s'>|a|</td><td align=right>%7.3f</td>"
                     "<td colspan=3 style='color:%s'>&nbsp;%s</td></tr>"
                     % (faint, na * k, faint, self.units.currentText()))
            if abs(na - 1.0) < 0.03:
                self.note.setText("|a| is one g, so the board is not accelerating. It is at "
                                  "rest, or moving at a constant velocity: the accelerometer "
                                  "cannot tell those apart.")
            else:
                self.note.setText("|a| is %.2f g. Something other than gravity is acting on "
                                  "the board." % na)
        elif session.modality == "emg":
            recent = d[-int(session.hz):, 1:1 + session.n_raw]
            pp = float(np.ptp(recent[:, 0])) if len(recent) else 0.0
            self.note.setText("Peak to peak over the last second: %.2f %s. A resting muscle "
                              "is tens of microvolts of noise; a contraction is millivolts, "
                              "and looks like noise too, because it is thousands of motor "
                              "units firing out of step." % (pp, session.channels[0].unit))
        elif session.modality == "ppg":
            recent = d[-int(4 * session.hz):, 1:1 + session.n_raw]
            if len(recent) > 10:
                dc = float(recent[:, 0].mean())
                ac = float(np.ptp(recent[:, 0]))
                self.note.setText("On the first channel the pulse is %.1f %% of the light "
                                  "received. Almost all of it is absorbed by tissue that does "
                                  "not pulse; the heartbeat is the small part that does."
                                  % (100.0 * ac / dc if dc else 0.0))
        else:
            self.note.setText("")
        self.readout.setText("<table cellspacing=4>%s%s</table>" % ("".join(rows), extra))

    # ---- the panel ----
    def explain(self):
        kind = self._kind
        if kind == "emg":
            return dict(
                head="Surface electromyography",
                equation="<span style='font-size:16pt'>v(t) &nbsp;=&nbsp; &#8721;<sub>units</sub> "
                         "&#8721;<sub>firings</sub> MUAP(t &minus; t<sub>k</sub>) "
                         "&nbsp;+&nbsp; <span style='color:%s'>hum</span> "
                         "&nbsp;+&nbsp; <span style='color:%s'>motion</span></span>"
                         % (theme.GUILTY, theme.GUILTY),
                terms=(("MUAP", "one motor unit's action potential, about 10 ms long"),
                       ("<span style='color:%s'>hum</span>" % theme.GUILTY,
                        "mains pick-up at 50 or 60 Hz, inside the muscle's own band"),
                       ("<span style='color:%s'>motion</span>" % theme.GUILTY,
                        "electrode movement, below 20 Hz")),
                why="Electrodes on the skin sum every motor unit firing underneath: tens "
                    "of microvolts to a few millivolts, mostly between 20 and 450 Hz. "
                    "Amplitude gives force; the spectrum gives fatigue.",
                tip="Watch a burst: noise that gets louder, which is what a contraction is.",
                source=("Electromyography, Wikipedia",
                        "https://en.wikipedia.org/wiki/Electromyography"),
                sources=(("Motor unit, Wikipedia", "https://en.wikipedia.org/wiki/Motor_unit"),))
        if kind == "ppg":
            return dict(
                head="Photoplethysmography",
                equation="<span style='font-size:16pt'>I &nbsp;=&nbsp; I<sub>0</sub> "
                         "e<sup>&minus;&#949; c l</sup> &nbsp;&nbsp;&#8658;&nbsp;&nbsp; "
                         "I(t) &nbsp;=&nbsp; DC &nbsp;+&nbsp; "
                         "<span style='color:%s'>AC(t)</span></span>" % theme.ACCENT,
                terms=(("I<sub>0</sub>, I", "light sent into the finger, and light received"),
                       ("&#949; c l", "absorption: how much blood, of what kind, over what path"),
                       ("<span style='color:%s'>AC</span>" % theme.ACCENT,
                        "the pulsatile part, one or two per cent of DC, which is the pulse")),
                why="Light through a fingertip falls off exponentially with the blood in "
                    "its path. Each beat adds blood, so a little less gets through: a "
                    "ripple of one or two per cent on a large constant.",
                tip="Raise the simulated fingertip's movement: the baseline wander swamps "
                    "the ripple, which is why the pulse is band-passed first.",
                source=("Photoplethysmogram, Wikipedia",
                        "https://en.wikipedia.org/wiki/Photoplethysmogram"),
                sources=(("Beer–Lambert law, Wikipedia",
                          "https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law"),))
        if kind == "generic":
            return dict(
                head="A signal",
                equation="<span style='font-size:16pt'>x[n] &nbsp;=&nbsp; x(n T<sub>s</sub>)"
                         "</span>",
                terms=(("T<sub>s</sub>", "the sampling interval"),
                       ("x[n]", "one number per channel per sample")),
                why="Channels this program has no page of its own for, drawn as they "
                    "arrive. The signal-processing pages work on them all the same.",
                source=("Sampling (signal processing), Wikipedia",
                        "https://en.wikipedia.org/wiki/Sampling_(signal_processing)"))
        return dict(
            head="Specific force",
            equation="<span style='font-size:16pt'>f &nbsp;=&nbsp; a &nbsp;&minus;&nbsp; g"
                     "</span>",
            terms=(("f", "what the accelerometer reports, per unit mass"),
                   ("a", "acceleration of the case"),
                   ("g", "gravity, 9.81 m/s² down")),
            why="An accelerometer measures specific force, not acceleration. At rest it "
                "reads the desk pushing up, 9.81 m/s²; in free fall it reads zero. Gravity "
                "has to be taken off before the rest is motion.",
            tip="Tip the board: the three axes trade the 9.81 between them while |a| "
                "stays put.",
            source=("Specific force — Wikipedia",
                    "https://en.wikipedia.org/wiki/Specific_force"),
            sources=(("Accelerometer, Wikipedia",
                      "https://en.wikipedia.org/wiki/Accelerometer"),))


class SamplingPage(Page):
    title = "Sampling"
    subtitle = "Sample rate, jitter, aliasing"

    def __init__(self):
        super().__init__()
        self.hist = pg.PlotWidget()
        hp = theme.plot(self.hist, "interval between samples", "count", "ms", "")
        theme.title(hp, "Where the samples actually landed in time")
        self.bars = pg.BarGraphItem(x=[0], height=[0], width=0.1,
                                    brush=pg.mkBrush(theme.ACCENT_LINE), pen=None)
        hp.addItem(self.bars)
        self.nominal = pg.InfiniteLine(angle=90, pen=theme.pen(theme.TRUTH, 1.4, dash=True),
                                       label="1/fₛ = {value:.2f} ms",
                                       labelOpts={"position": 0.92, "color": theme.INK_FAINT})
        hp.addItem(self.nominal)

        self.alias = pg.PlotWidget()
        self.ap = theme.plot(self.alias, "seconds before now", "value", "", "")
        theme.title(self.ap, "The same signal, sampled less often")
        self.ap.setXRange(-4, 0, padding=0)
        theme.legend(self.ap)
        self.alias_y = YRange(self.ap)
        self.full = self.ap.plot(pen=theme.pen(theme.LINE, 1.6), name="every sample")
        self.kept = self.ap.plot(pen=theme.pen(theme.X_AXIS, 1.8), symbol="o", symbolSize=5,
                                 symbolBrush=theme.X_AXIS, symbolPen=None, name="what is kept")

        self.keep = Slider("keep one sample in", 1, 40, 1, 1, "", 0, width=220)
        self.keep.changed.connect(lambda _: self.explain_changed.emit())
        self.bits = Slider("bits", 2, 16, 16, 1, "", 0, width=150)
        self.bits.changed.connect(lambda _: self.explain_changed.emit())
        self.pick = Picker(self._picked)

        self.stats = label("", "Readout")
        self.stats.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.verdict = label("", wrap=True)

        top, _ = card(self.hist)
        bottom, _ = card(self.alias,
                         row(self.keep, self.bits, label("channel", "Caption"),
                             self.pick.box, None),
                         self.verdict)
        side, sl = card(label("THE LINK", "Section"), self.stats)
        sl.addStretch(1)
        side.setFixedWidth(330)

        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 2)
        col.addWidget(bottom, 3)
        self.body(row(col, side))
        self.slow = Every(10)
        self._window = None
        self._session = None

    def _picked(self):
        theme.axis_label(self.ap, "left", self.pick.name() or "value", self.pick.unit())

    def on_show(self, session, source):
        self._session = session
        self.pick.follow(session)
        self._picked()

    def tick(self, session, source):
        if session is not self._session:
            self.on_show(session, source)
        d = session.last(20.0)
        if len(d) < 20:
            return
        t = d[:, core.T]
        dt = np.diff(t) * 1e3                                   # ms
        dt = dt[(dt > 0) & (dt < 200)]
        if len(dt) < 5:
            return
        med = float(np.median(dt))
        hz = 1000.0 / med if med > 0 else 0.0
        # Rounded to a quarter of a millisecond, so that the axis is redrawn
        # when the link's rate really changes and not because the median
        # wobbled in the last decimal.
        window = (round(max(0.0, med - 8) * 4) / 4, round((med + 12) * 4) / 4)
        if window != self._window:
            self._window = window
            self.hist.getPlotItem().setXRange(*window, padding=0)
        if self.slow():
            lo, hi = window
            counts, edges = np.histogram(dt, bins=48, range=(lo, hi))
            mids = 0.5 * (edges[1:] + edges[:-1])
            self.bars.setOpts(x=mids, height=counts, width=(hi - lo) / 52.0)
            self.nominal.setValue(med)

        if not self.slow.i % 10 == 0:
            self._alias(session, hz)
            return
        jitter = float(np.percentile(dt, 99) - np.percentile(dt, 1))
        late = int(np.sum(dt > 1.5 * med))
        self.stats.setText(
            "<table cellspacing=5>"
            "<tr><td style='color:%s'>arriving</td><td align=right><b>%.1f</b></td>"
            "<td style='color:%s'>Hz</td></tr>"
            "<tr><td style='color:%s'>interval</td><td align=right>%.2f</td>"
            "<td style='color:%s'>ms</td></tr>"
            "<tr><td style='color:%s'>spread, 1–99&#37;</td><td align=right>%.2f</td>"
            "<td style='color:%s'>ms</td></tr>"
            "<tr><td style='color:%s'>late by half a tick</td><td align=right>%d</td>"
            "<td style='color:%s'>of %d</td></tr>"
            "<tr><td style='color:%s'>never arrived</td><td align=right>%d</td>"
            "<td></td></tr></table>"
            % (theme.INK_FAINT, hz, theme.INK_FAINT,
               theme.INK_FAINT, med, theme.INK_FAINT,
               theme.INK_FAINT, jitter, theme.INK_FAINT,
               theme.INK_FAINT, late, theme.INK_FAINT, len(dt),
               theme.INK_FAINT, source.dropped if source else 0))

        self._alias(session, hz)

    def _alias(self, session, hz):
        d4 = session.last(4.0)
        if len(d4) < 8:
            return
        tt = d4[:, core.T] - d4[-1, core.T]
        yy = self.pick.get(d4)
        step = int(self.keep.value())
        self.alias_y.fit(yy)
        draw(self.full, tt, yy, session.hz, 500)
        kt, ky = tt[::step], yy[::step]
        # the ADC's other loss: a value can only be one of 2^bits levels
        bits = int(self.bits.value())
        span = 2.0 * (self.alias_y.now or 1.0)
        self.q_step = span / (2 ** bits)
        if bits < 16:
            ky = np.round(ky / self.q_step) * self.q_step
        # Symbols are drawn one at a time. Four hundred of them cost more than
        # everything else on this page put together, and at that density they
        # are a solid line anyway, so they are only worth drawing once the
        # decimation has made them countable.
        if len(kt) <= 150:
            self.kept.setData(kt, ky, symbol="o", symbolSize=5,
                              symbolBrush=theme.X_AXIS, symbolPen=None)
        else:
            self.kept.setData(*decimate(kt, ky, limit=500), symbol=None)

        eff = hz / step
        self.verdict.setText(
            "<span style='color:%s'>effective rate <b>%.1f Hz</b> &nbsp;·&nbsp; "
            "anything above <b>%.1f Hz</b> now comes back as a lower frequency"
            "&nbsp;·&nbsp; %d bits: a step of <b>%.3g</b> %s</span>"
            % (theme.WARN if step > 1 else theme.INK_SOFT, eff, eff / 2,
               bits, self.q_step, self.pick.unit()))

    q_step = 0.0

    def explain(self):
        return dict(
            head="Nyquist frequency",
            equation="<span style='font-size:16pt'>f<sub>N</sub> &nbsp;=&nbsp; "
                     "f<sub>s</sub> / 2 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; f &gt; f<sub>N</sub> "
                     "&nbsp;&#8594;&nbsp; <span style='color:%s'>| f &minus; "
                     "k f<sub>s</sub> |</span></span>" % theme.GUILTY,
            terms=(("f<sub>s</sub>", "sample rate"),
                   ("f<sub>N</sub>", "Nyquist frequency"),
                   ("<span style='color:%s'>alias</span>" % theme.GUILTY,
                    "a component above f<sub>N</sub>, returned at a lower frequency"),
                   ("q", "the quantisation step, range / 2<sup>bits</sup>: %.3g now"
                         % self.q_step)),
            why="Sampling loses anything above f<sub>s</sub>/2: it is folded down and "
                "cannot be told apart afterwards. Quantising loses anything smaller than "
                "one step.",
            tip="Shake the board at 8 Hz and keep one sample in twelve: the 8 Hz comes "
                "back slower. Then drop to 4 bits and watch the staircase.",
            source=("Nyquist–Shannon sampling theorem, Wikipedia",
                    "https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem"),
            sources=(("Aliasing, Wikipedia", "https://en.wikipedia.org/wiki/Aliasing"),
                     ("Quantization (signal processing), Wikipedia",
                      "https://en.wikipedia.org/wiki/Quantization_(signal_processing)")))
