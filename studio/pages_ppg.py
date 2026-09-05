# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_ppg.py
The two pages for a fingertip.

    Heart rate   the beats PpgSession found, the rate between them, and how
                 much that rate wanders
    SpO2         two wavelengths, the ratio of their pulses, and the textbook
                 line that turns it into a saturation

Both draw the simulator's truth beside the estimate when there is one.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import core, dsp, theme
from .shell import Band, Every, Page, YRange, card, decimate, draw, label, row

LOOK = 8.0
TREND = 60.0
RED, IR = theme.MODALITY["ppg"], "#8b2e2e"


class HeartRatePage(Page):
    title = "Heart rate"
    subtitle = "Beat detection, heart rate, and its variability"
    modalities = ("ppg",)

    def __init__(self):
        super().__init__()
        self.pulse = pg.PlotWidget()
        pp = theme.plot(self.pulse, "seconds before now", "pulse", "", "")
        theme.title(pp, "The band-passed pulse, and the beats found in it")
        pp.setXRange(-LOOK, 0, padding=0)
        self.wave = pp.plot(pen=theme.pen(RED, 1.6))
        self.marks = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(theme.INK),
                                        pen=pg.mkPen(theme.SURFACE, width=1.5))
        pp.addItem(self.marks)
        self.py = YRange(pp, floor=1.0)

        self.hrplot = pg.PlotWidget()
        hp = theme.plot(self.hrplot, "seconds before now", "heart rate", "", "BPM")
        theme.title(hp, "Beat to beat")
        hp.setXRange(-TREND, 0, padding=0)
        hp.setYRange(30, 200, padding=0)
        hp.disableAutoRange(axis="y")
        theme.legend(hp, offset=(10, -10))          # bottom left, away from the newest beats
        self.hr_curve = hp.plot(pen=theme.pen(RED, 1.8), symbol="o", symbolSize=4,
                                symbolBrush=RED, symbolPen=None, name="60 / IBI")
        self.truth = hp.plot(pen=theme.pen(theme.TRUTH, 1.6, dash=True), name="truth")
        self.hp = hp

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.big = label("", "Readout")
        self.big.setStyleSheet("font-size: %dpt; font-weight: 600; color: %s;"
                               % (theme.SIZE_READOUT + 10, RED))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("HEART", "Section"), self.big, self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(360)

        top, _ = card(self.pulse)
        bottom, _ = card(self.hrplot)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, side))
        self.slow = Every(6)
        self._session = None

    def on_show(self, session, source):
        self._session = session
        self.explain_changed.emit()

    def tick(self, session, source):
        d = session.last(LOOK)
        if len(d) < 8 or not hasattr(session, "BEAT"):
            return
        t = d[:, core.T] - d[-1, core.T]
        ch = session.beat_channel
        y = d[:, session.filt_cols][:, ch]
        draw(self.wave, t, y, session.hz)
        self.py.fit(y)
        hit = d[:, session.BEAT] > 0.5
        self.marks.setData(t[hit], y[hit])

        bt, ibi = session.tachogram(TREND)
        now = d[-1, core.T]
        if len(bt):
            self.hr_curve.setData(bt - now, 60000.0 / ibi)
        if source is not None and hasattr(source, "true_bpm") and len(bt):
            self.truth.setData(bt - now, [source.true_bpm(b) for b in bt])
        else:
            self.truth.setData([], [])
        if not self.slow():
            return

        h = session.heart(10.0)
        faint = theme.INK_FAINT
        self.big.setText("%.0f <span style='font-size:%dpt; color:%s'>BPM</span>"
                         % (h["hr"], theme.SIZE_BODY, faint) if h["hr"] else "&#8212;")
        self.readout.setText(
            "<table cellspacing=5>"
            "<tr><td style='color:%s'>mean, 10 s</td><td align=right><b>%.1f</b></td>"
            "<td style='color:%s'>BPM</td></tr>"
            "<tr><td style='color:%s'>interval</td><td align=right>%.0f</td>"
            "<td style='color:%s'>ms</td></tr>"
            "<tr><td colspan=3><hr></td></tr>"
            "<tr><td style='color:%s'>RMSSD</td><td align=right>%.0f</td>"
            "<td style='color:%s'>ms</td></tr>"
            "<tr><td style='color:%s'>SDNN</td><td align=right>%.0f</td>"
            "<td style='color:%s'>ms</td></tr>"
            "<tr><td style='color:%s'>beats used</td><td align=right>%d</td><td></td></tr>"
            "</table>"
            % (faint, h["mean"], faint,
               faint, 60000.0 / h["hr"] if h["hr"] else 0.0, faint,
               faint, h["rmssd"], faint, faint, h["sdnn"], faint, faint, h["n"]))
        if source is not None and hasattr(source, "true_bpm"):
            true = source.true_bpm(now)
            self.state.setText(
                "<span style='color:%s'>&#8212;</span> what the simulated heart was really "
                "doing. It beat at <b>%.1f</b>; the detector read <b>%.1f</b> over the last "
                "ten seconds, <b>%.1f BPM</b> off." % (theme.TRUTH, true, h["mean"],
                                                         abs(h["mean"] - true)))
        elif h["n"] < 3:
            self.state.setText("Waiting for beats. Keep the finger still on the sensor; "
                               "the first few seconds are the filter settling.")
        else:
            self.state.setText("RMSSD is the beat-to-beat variability: high at rest and "
                               "in the young, low under stress. It needs a still finger; "
                               "a missed or doubled beat inflates it at once.")

    def explain(self):
        return dict(
            head="Heart rate and variability",
            equation="<span style='font-size:16pt'>HR &nbsp;=&nbsp; 60 / IBI"
                     "&nbsp;&nbsp;&nbsp;&nbsp; RMSSD &nbsp;=&nbsp; &#8730;( mean( "
                     "(IBI<sub>i+1</sub> &minus; IBI<sub>i</sub>)<sup>2</sup> ) )</span>",
            terms=(("IBI", "the interval between two beats, in seconds"),
                   ("RMSSD", "how much that interval changes from one beat to the next"),
                   ("<span style='color:%s'>a missed beat</span>" % theme.GUILTY,
                    "doubles one interval and ruins the variability")),
            why="A beat is the peak after a steep upstroke of the band-passed pulse, "
                "at least a refractory interval after the last. Variability is read from "
                "the gaps, so it is only as good as the detection under it.",
            tip="Raise the simulated variability and watch the beat-to-beat trace breathe; "
                "add movement and see which beats are lost.",
            source=("Heart rate, Wikipedia", "https://en.wikipedia.org/wiki/Heart_rate"),
            sources=(("Heart rate variability, Wikipedia",
                      "https://en.wikipedia.org/wiki/Heart_rate_variability"),
                     ("Photoplethysmogram, Wikipedia",
                      "https://en.wikipedia.org/wiki/Photoplethysmogram")))


class Spo2Page(Page):
    title = "SpO2"
    subtitle = "Oxygen saturation from two wavelengths"
    modalities = ("ppg",)

    def __init__(self):
        super().__init__()
        self.two = pg.PlotWidget()
        tp = theme.plot(self.two, "seconds before now", "change from DC", "", "%")
        theme.title(tp, "Red and infrared, each as a percentage of its own level")
        tp.setXRange(-LOOK, 0, padding=0)
        theme.legend(tp)
        self.ir_curve = tp.plot(pen=theme.pen(IR, 1.6), name="infrared")
        self.red_curve = tp.plot(pen=theme.pen(RED, 1.6), name="red")
        self.ty = YRange(tp, floor=0.5)

        self.trend = pg.PlotWidget()
        sp = theme.plot(self.trend, "seconds before now", "SpO₂", "", "%")
        theme.title(sp, "Saturation, one reading every second")
        sp.setXRange(-TREND, 0, padding=0)
        sp.setYRange(80, 101, padding=0)
        sp.disableAutoRange(axis="y")
        theme.legend(sp, offset=(10, -10))
        self.spo2_curve = sp.plot(pen=theme.pen(RED, 1.8), symbol="o", symbolSize=4,
                                  symbolBrush=RED, symbolPen=None, name="estimate")
        self.truth = sp.plot(pen=theme.pen(theme.TRUTH, 1.6, dash=True), name="truth")

        self.big = label("", "Readout")
        self.big.setStyleSheet("font-size: %dpt; font-weight: 600; color: %s;"
                               % (theme.SIZE_READOUT + 10, RED))
        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("SATURATION", "Section"), self.big, self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(360)

        top, _ = card(self.two)
        bottom, _ = card(self.trend)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, side))
        self._points = []
        self._last = None
        self._session = None
        self.slow = Every(6)

    def on_show(self, session, source):
        if session is not self._session:
            self._points = []
            self._last = None
        self._session = session
        self.explain_changed.emit()

    @staticmethod
    def _find(session):
        names = [c.name.lower() for c in session.channels]
        ir = names.index("ir") if "ir" in names else None
        red = names.index("red") if "red" in names else None
        return ir, red

    def tick(self, session, source):
        d = session.last(LOOK)
        if len(d) < 8 or not hasattr(session, "filt_cols"):
            return
        ir, red = self._find(session)
        if ir is None or red is None:
            self.state.setText("Saturation needs an infrared and a red channel, named "
                               "<b>ir</b> and <b>red</b> in the board's header. This source "
                               "has: %s." % ", ".join(c.name for c in session.channels))
            self.big.setText("&#8212;")
            return
        t = d[:, core.T] - d[-1, core.T]
        raw = session.raw(d)
        for curve, i in ((self.ir_curve, ir), (self.red_curve, red)):
            x = raw[:, i]
            dc = float(x.mean())
            pct = 100.0 * (x - dc) / dc if dc else x * 0
            draw(curve, t, pct, session.hz)
        self.ty.fit(100.0 * (raw[:, ir] - raw[:, ir].mean()) / max(raw[:, ir].mean(), 1e-9),
                    100.0 * (raw[:, red] - raw[:, red].mean()) / max(raw[:, red].mean(), 1e-9))

        now = float(d[-1, core.T])
        if self._last is None or now - self._last >= 1.0:
            self._last = now
            w = session.last(4.0)
            wr = session.raw(w)
            R, s = dsp.spo2(wr[:, red], wr[:, ir])
            if np.isfinite(s):
                self._points.append((now, R, s))
                self._points = [p for p in self._points if p[0] > now - TREND]
        pts = np.asarray(self._points, float) if self._points else np.zeros((0, 3))
        if len(pts):
            self.spo2_curve.setData(pts[:, 0] - now, np.clip(pts[:, 2], 70, 102))
        if source is not None and hasattr(source, "spo2") and len(pts):
            self.truth.setData(pts[:, 0] - now, np.full(len(pts), float(source.spo2)))
        else:
            self.truth.setData([], [])
        if not self.slow() or not len(pts):
            return

        R, s = pts[-1, 1], pts[-1, 2]
        w = session.raw(session.last(4.0))
        faint = theme.INK_FAINT
        self.big.setText("%.0f <span style='font-size:%dpt; color:%s'>%%</span>"
                         % (s, theme.SIZE_BODY, faint))
        self.readout.setText(
            "<table cellspacing=5>"
            "<tr><td style='color:%s'>R</td><td align=right><b>%.3f</b></td><td></td></tr>"
            "<tr><td style='color:%s'>AC/DC infrared</td><td align=right>%.2f</td>"
            "<td style='color:%s'>%%</td></tr>"
            "<tr><td style='color:%s'>AC/DC red</td><td align=right>%.2f</td>"
            "<td style='color:%s'>%%</td></tr>"
            "<tr><td style='color:%s'>DC infrared</td><td align=right>%.0f</td>"
            "<td style='color:%s'>counts</td></tr>"
            "<tr><td style='color:%s'>DC red</td><td align=right>%.0f</td>"
            "<td style='color:%s'>counts</td></tr></table>"
            % (faint, R,
               faint, 100.0 * w[:, ir].std() / max(w[:, ir].mean(), 1e-9), faint,
               faint, 100.0 * w[:, red].std() / max(w[:, red].mean(), 1e-9), faint,
               faint, w[:, ir].mean(), faint, faint, w[:, red].mean(), faint))
        if source is not None and hasattr(source, "spo2"):
            self.state.setText(
                "<span style='color:%s'>&#8212;</span> the saturation the simulated "
                "fingertip was set to, <b>%.1f %%</b>. The line reads back <b>%.1f %%</b>. "
                "On a real finger the line is only approximate: an oximeter carries a "
                "calibration curve measured on volunteers, and this program does not."
                % (theme.TRUTH, source.spo2, s))
        else:
            self.state.setText("The textbook line, not a medical reading: a real "
                               "oximeter carries a calibration measured on volunteers. "
                               "Expect a few per cent of disagreement.")

    def explain(self):
        return dict(
            head="Ratio of ratios",
            equation="<span style='font-size:16pt'>R &nbsp;=&nbsp; "
                     "(AC<sub>red</sub> / DC<sub>red</sub>) / (AC<sub>ir</sub> / DC<sub>ir</sub>)"
                     "&nbsp;&nbsp;&nbsp;&nbsp; SpO<sub>2</sub> &nbsp;&#8776;&nbsp; "
                     "110 &minus; 25 <span style='color:%s'>R</span></span>" % theme.ACCENT,
            terms=(("AC, DC", "the pulsatile and the steady part of each colour, over 4 s"),
                   ("<span style='color:%s'>R</span>" % theme.ACCENT,
                    "how much redder the pulse is than the infrared pulse"),
                   ("<span style='color:%s'>110 &minus; 25R</span>" % theme.GUILTY,
                    "an approximation; every real device has its own calibration")),
            why="Oxygenated and deoxygenated blood absorb red and infrared differently. "
                "Dividing each colour's pulse by its own level cancels the finger and the "
                "lamp; what is left depends on the blood alone.",
            tip="Set the simulated saturation to 88 % and watch the red pulse grow against "
                "the infrared one.",
            source=("Pulse oximetry, Wikipedia",
                    "https://en.wikipedia.org/wiki/Pulse_oximetry"),
            sources=(("Beer–Lambert law, Wikipedia",
                      "https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law"),))
