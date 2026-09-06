# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_theory.py
The Signal theory pages: demonstrations on synthetic signals whose truth is
known exactly. None of them reads a sensor, so all of them work with nothing
connected.

    Sampling theorem    samples put back into a signal, and where that fails
    Aliasing            a frequency above f_s / 2 coming back as a lower one
    Quantisation        bits, step size, and the noise they add
    Fourier synthesis   a square, triangle or sawtooth built from sinusoids
    Spectral leakage    a tone between bins, and what a window does about it

Each page redraws when a control moves and does nothing between, so they cost
the window no time while a live signal is on another page.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme, theory
from .shell import Page, Slider, caption, card, flexible, label, row

BIG = "font-size: %dpt;" % (theme.SIZE_BODY + 2)


def cell(name, value, unit=""):
    return ("<tr><td style='color:%s'>%s</td><td align=right><b>%s</b></td>"
            "<td style='color:%s'>%s</td></tr>" % (theme.INK_FAINT, name, value,
                                                   theme.INK_FAINT, unit))


def table(*cells):
    return "<table cellspacing=5>" + "".join(cells) + "</table>"


def verdict(colour, text):
    return "<span style='color:%s'>%s</span>" % (colour, text)


class Demo(Page):
    """A page on a synthetic signal: redraws on a control, ignores the session."""

    modalities = ("*",)

    def _side(self, section):
        self.stats = label("", "Readout")
        self.stats.setStyleSheet(BIG)
        self.verdict = label("", wrap=True)
        side, lay = card(label(section, "Section"), self.stats, self.verdict)
        lay.addStretch(1)
        side.setFixedWidth(theme.SIDE_W)
        return side

    def _watch(self, *controls):
        for c in controls:
            if isinstance(c, Slider):
                c.changed.connect(lambda _v: self._redraw())
            else:
                c.currentIndexChanged.connect(lambda _i: self._redraw())

    def _redraw(self):
        raise NotImplementedError


def combo(items, width):
    """A combo box that asks for `width` px and gives the row's spare width back."""
    w = QtWidgets.QComboBox()
    w.addItems(list(items))
    return flexible(w, floor=width, cap=max(width, 220))


def stems(xs, heights):
    """x, y arrays that draw one vertical line per (x, height) with connect='pairs'."""
    x = np.repeat(np.asarray(xs, float), 2)
    y = np.zeros(2 * len(xs))
    y[1::2] = heights
    return x, y


# ---------------------------------------------------------------
# sampling theorem
# ---------------------------------------------------------------
SPAN = 2.0          # seconds shown
EXTRA = 3.0         # seconds sampled either side of that, so the ends are not edge effects


class TheoremPage(Demo):
    title = "Sampling theorem"
    subtitle = "Samples put back into a signal, and where that stops working"

    def __init__(self):
        super().__init__()
        # The column beside the readout card is about 400 px wide in the
        # smallest window, so the controls sit in two rows.
        self.f = Slider("signal frequency", 0.5, 12.0, 3.0, 0.1, "Hz", 1, width=170)
        self.fs = Slider("sample rate", 2.0, 40.0, 10.0, 0.5, "Hz", 1, width=170)
        self.kind = combo(["sine", "sine + third harmonic"], 160)

        self.time = pg.PlotWidget()
        tp = theme.plot(self.time, "time", "value", "s", "")
        theme.title(tp, "The signal, its samples, and the signal rebuilt from the samples alone")
        theme.legend(tp)
        self.truth = tp.plot(pen=theme.pen(theme.TRUTH, 1.4, dash=True), name="the signal")
        self.rebuilt = tp.plot(pen=theme.pen(theme.ACCENT, 2.0), name="rebuilt from the samples")
        self.dots = tp.plot(pen=None, symbol="o", symbolSize=6, symbolBrush=theme.X_AXIS,
                            symbolPen=None, name="samples")
        tp.setXRange(0, SPAN, padding=0)
        tp.setYRange(-1.6, 1.6, padding=0)
        tp.setMouseEnabled(x=False, y=False)

        self.freq = pg.PlotWidget()
        self.fp = theme.plot(self.freq, "frequency", "amplitude", "Hz", "")
        theme.title(self.fp, "Where each component is, and where the samples say it is")
        theme.legend(self.fp)
        self.in_signal = self.fp.plot(pen=theme.pen(theme.TRUTH, 4.0), connect="pairs",
                                      name="in the signal")
        self.as_sampled = self.fp.plot(pen=theme.pen(theme.WARN, 2.0), connect="pairs",
                                       name="as the samples have it")
        self.nyq = pg.InfiniteLine(angle=90, pen=theme.pen(theme.INK_FAINT, 1.4, dash=True),
                                   label="f_s / 2", labelOpts={"position": 0.88,
                                                                "color": theme.INK_FAINT})
        self.fp.addItem(self.nyq)
        self.fp.setYRange(0, 1.25, padding=0)
        self.fp.setMouseEnabled(x=False, y=False)

        top, _ = card(self.time, row(self.f, self.fs, None),
                      row(caption("signal"), self.kind, None))
        bottom, _ = card(self.freq)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, self._side("RECONSTRUCTION")))
        self._watch(self.f, self.fs, self.kind)
        self.error_pct = 0.0
        self._redraw()

    def _components(self):
        f = self.f.value()
        comps = [(f, 1.0)]
        if self.kind.currentIndex() == 1:
            comps.append((3.0 * f, 1.0 / 3.0))
        return comps

    def _redraw(self):
        fs = self.fs.value()
        comps = self._components()
        t = np.linspace(0.0, SPAN, 1200)
        x = sum(a * np.cos(2.0 * np.pi * fc * t) for fc, a in comps)
        tn = np.arange(-EXTRA, SPAN + EXTRA, 1.0 / fs)
        xn = sum(a * np.cos(2.0 * np.pi * fc * tn) for fc, a in comps)
        y = theory.reconstruct(tn, xn, t, fs)
        self.truth.setData(t, x)
        self.rebuilt.setData(t, y)
        shown = (tn >= 0.0) & (tn <= SPAN)
        self.dots.setData(tn[shown], xn[shown])

        fmax = max(fc for fc, _a in comps)
        self.error_pct = float(np.sqrt(np.mean((y - x) ** 2)) / max(np.abs(x).max(), 1e-9) * 100)
        self.in_signal.setData(*stems([fc for fc, _a in comps], [a for _fc, a in comps]))
        self.as_sampled.setData(*stems([float(theory.alias(fc, fs)) for fc, _a in comps],
                                       [a for _fc, a in comps]))
        self.nyq.setValue(fs / 2.0)
        self.fp.setXRange(0, max(fs, 1.15 * fmax) + 1.0, padding=0)

        ok = fs > 2.0 * fmax
        self.stats.setText(table(
            cell("highest frequency in it", "%.1f" % fmax, "Hz"),
            cell("sample rate f<sub>s</sub>", "%.1f" % fs, "Hz"),
            cell("f<sub>s</sub> / 2", "%.2f" % (fs / 2.0), "Hz"),
            cell("f<sub>s</sub> / f<sub>max</sub>", "%.2f" % (fs / fmax), "x"),
            cell("rebuilt, error", "%.2f" % self.error_pct, "% of amplitude")))
        if ok:
            self.verdict.setText(verdict(
                theme.GOOD, "f<sub>s</sub> is above 2 f<sub>max</sub>: the samples hold the "
                "whole signal, and the sinc sum gives it back. Nothing between the samples "
                "was lost; it was implied. The fraction of a per cent left is the finite "
                "record: sinc tails that were cut off."))
        else:
            self.verdict.setText(verdict(
                theme.WARN, "f<sub>s</sub> is below 2 f<sub>max</sub>: a lower frequency "
                "passes through the very same samples, and the rebuilt signal is that one. "
                "The original is not in the samples any more, and no method gets it back."))
        self.explain_changed.emit()

    def explain(self):
        return dict(
            head="Nyquist–Shannon sampling theorem",
            equation="<span style='font-size:16pt'>x(t) &nbsp;=&nbsp; &sum;<sub>n</sub> "
                     "x[n] &middot; sinc( f<sub>s</sub> (t &minus; n T<sub>s</sub>) )"
                     "&nbsp;&nbsp;&nbsp;&nbsp; if &nbsp;<span style='color:%s'>f<sub>s</sub> "
                     "&gt; 2 f<sub>max</sub></span></span>" % theme.GUILTY,
            terms=(("x[n]", "the samples, T<sub>s</sub> = 1 / f<sub>s</sub> apart"),
                   ("sinc", "sin(&pi;u) / &pi;u: one for every sample, 1 at its own instant "
                            "and 0 at every other"),
                   ("<span style='color:%s'>f<sub>s</sub> &gt; 2 f<sub>max</sub></span>"
                    % theme.GUILTY, "the condition. Error now: %.2f %% of the amplitude"
                    % self.error_pct)),
            why="A band-limited signal is fixed completely by samples taken faster than "
                "twice its highest frequency: there is exactly one such signal through "
                "them, and the sinc sum is it. Sample slower and a second, lower-frequency "
                "signal fits the same points, so the samples no longer say which was meant.",
            tip="Set the signal to 3 Hz and slide the rate from 10 Hz down through 6 Hz: "
                "the rebuilt curve is right until 6 and wrong below it. Add the third "
                "harmonic and the limit is 18 Hz, not 6.",
            source=("Nyquist–Shannon sampling theorem, Wikipedia",
                    "https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem"),
            sources=(("Whittaker–Shannon interpolation formula, Wikipedia",
                      "https://en.wikipedia.org/wiki/Whittaker%E2%80%93Shannon_interpolation_formula"),
                     ("Sinc function, Wikipedia", "https://en.wikipedia.org/wiki/Sinc_function")))


# ---------------------------------------------------------------
# aliasing
# ---------------------------------------------------------------
class AliasPage(Demo):
    title = "Aliasing"
    subtitle = "A frequency above f_s / 2 comes back as a lower one"

    def __init__(self):
        super().__init__()
        self.f = Slider("true frequency", 0.0, 60.0, 45.0, 0.5, "Hz", 1, width=170)
        self.fs = Slider("sample rate", 10.0, 100.0, 50.0, 1.0, "Hz", 0, width=170)

        self.time = pg.PlotWidget()
        tp = theme.plot(self.time, "time", "value", "s", "")
        theme.title(tp, "One second: the true cosine, its samples, and the cosine the samples "
                        "also fit")
        theme.legend(tp)
        self.truth = tp.plot(pen=theme.pen(theme.TRUTH, 1.2, dash=True), name="true")
        self.seen = tp.plot(pen=theme.pen(theme.WARN, 2.2), name="what the samples fit")
        self.dots = tp.plot(pen=None, symbol="o", symbolSize=6, symbolBrush=theme.X_AXIS,
                            symbolPen=None, name="samples")
        tp.setXRange(0, 1.0, padding=0)
        tp.setYRange(-1.4, 1.4, padding=0)
        tp.setMouseEnabled(x=False, y=False)

        self.fold = pg.PlotWidget()
        self.fdp = theme.plot(self.fold, "true frequency", "apparent frequency", "Hz", "Hz")
        theme.title(self.fdp, "Folding: what every true frequency comes back as")
        self.curve = self.fdp.plot(pen=theme.pen(theme.LINE, 1.8))
        self.mark = self.fdp.plot(pen=None, symbol="o", symbolSize=11, symbolBrush=theme.WARN,
                                  symbolPen=None)
        self.nyq = pg.InfiniteLine(angle=90, pen=theme.pen(theme.INK_FAINT, 1.4, dash=True),
                                   label="f_s / 2", labelOpts={"position": 0.88,
                                                                "color": theme.INK_FAINT})
        self.fdp.addItem(self.nyq)
        self.fdp.setMouseEnabled(x=False, y=False)

        top, _ = card(self.time, row(self.f, self.fs, None))
        bottom, _ = card(self.fold)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, self._side("WHAT THE SAMPLES SAY")))
        self._watch(self.f, self.fs)
        self.fa = 0.0
        self._redraw()

    def _redraw(self):
        f, fs = self.f.value(), self.fs.value()
        t = np.linspace(0.0, 1.0, 2000)
        tn = np.arange(0.0, 1.0 + 1e-9, 1.0 / fs)
        fa = theory.signed_alias(f, fs)
        self.fa = abs(fa)
        self.truth.setData(t, np.cos(2.0 * np.pi * f * t))
        self.seen.setData(t, np.cos(2.0 * np.pi * fa * t))
        self.dots.setData(tn, np.cos(2.0 * np.pi * f * tn))

        ff = np.linspace(0.0, 3.0 * fs, 900)
        self.curve.setData(ff, theory.alias(ff, fs))
        self.mark.setData([f], [self.fa])
        self.nyq.setValue(fs / 2.0)
        self.fdp.setXRange(0, 3.0 * fs, padding=0)
        self.fdp.setYRange(0, 0.6 * fs, padding=0)

        k = int(round(f / fs))
        self.stats.setText(table(
            cell("true frequency f", "%.1f" % f, "Hz"),
            cell("sample rate f<sub>s</sub>", "%.0f" % fs, "Hz"),
            cell("f<sub>s</sub> / 2", "%.1f" % (fs / 2.0), "Hz"),
            cell("nearest multiple k", "%d" % k, ""),
            cell("appears as | f &minus; k f<sub>s</sub> |", "%.1f" % self.fa, "Hz")))
        if f <= fs / 2.0:
            self.verdict.setText(verdict(
                theme.GOOD, "Below f<sub>s</sub> / 2 the samples mean what they say: the "
                "only cosine through them at this rate is the true one."))
        else:
            self.verdict.setText(verdict(
                theme.WARN, "The samples cannot tell %.1f Hz from %.1f Hz: both cosines "
                "pass through every one of them. Anything that reads the samples sees "
                "the lower one." % (f, self.fa)))
        self.explain_changed.emit()

    def explain(self):
        return dict(
            head="Aliasing",
            equation="<span style='font-size:16pt'>f<sub>apparent</sub> &nbsp;=&nbsp; "
                     "<span style='color:%s'>| f &minus; k f<sub>s</sub> |</span>"
                     "&nbsp;&nbsp;&nbsp;&nbsp; k = round( f / f<sub>s</sub> )</span>"
                     % theme.GUILTY,
            terms=(("f", "the frequency that was really there"),
                   ("f<sub>s</sub>", "the sample rate"),
                   ("<span style='color:%s'>f<sub>apparent</sub></span>" % theme.GUILTY,
                    "what any reading of the samples sees: %.1f Hz now" % self.fa)),
            why="cos(2&pi;f t) and cos(2&pi;(f &minus; k f<sub>s</sub>) t) take the same "
                "value at every sample instant t = n / f<sub>s</sub>, so once sampled they "
                "are the same sequence. Every frequency folds down into 0 to "
                "f<sub>s</sub> / 2, and nothing afterwards can unfold it, which is why a "
                "filter has to go before the sampler.",
            tip="At 50 Hz, slide f from 20 up through 25 and on to 45: the apparent "
                "frequency climbs to 25 and comes back down to 5. At exactly 50 Hz the "
                "samples do not move at all.",
            source=("Aliasing, Wikipedia", "https://en.wikipedia.org/wiki/Aliasing"),
            sources=(("Nyquist frequency, Wikipedia",
                      "https://en.wikipedia.org/wiki/Nyquist_frequency"),
                     ("Anti-aliasing filter, Wikipedia",
                      "https://en.wikipedia.org/wiki/Anti-aliasing_filter")))


# ---------------------------------------------------------------
# quantisation
# ---------------------------------------------------------------
class QuantisePage(Demo):
    title = "Quantisation"
    subtitle = "Bits, step size, and the noise they add"

    def __init__(self):
        super().__init__()
        self.bits = Slider("bits", 1, 16, 4, 1, "", 0, width=170)
        self.amp = Slider("amplitude, of full scale", 5, 120, 90, 5, "%", 0, width=170)

        self.time = pg.PlotWidget()
        tp = theme.plot(self.time, "time", "value", "s", "")
        theme.title(tp, "Two cycles, and the same two cycles held to 2^bits levels")
        theme.legend(tp)
        self.truth = tp.plot(pen=theme.pen(theme.TRUTH, 1.2, dash=True), name="the signal")
        self.held = tp.plot(pen=theme.pen(theme.ACCENT, 2.0), name="quantised")
        tp.setXRange(0, 2.0, padding=0)
        tp.setYRange(-1.3, 1.3, padding=0)
        tp.setMouseEnabled(x=False, y=False)
        for y in (1.0, -1.0):
            tp.addItem(pg.InfiniteLine(angle=0, pos=y,
                                       pen=theme.pen(theme.INK_FAINT, 1.0, dash=True)))

        self.err = pg.PlotWidget()
        self.ep = theme.plot(self.err, "time", "error", "s", "")
        theme.title(self.ep, "What was lost: quantised minus true")
        self.error = self.ep.plot(pen=theme.pen(theme.WARN, 1.4))
        self.half_up = pg.InfiniteLine(angle=0, pen=theme.pen(theme.INK_FAINT, 1.0, dash=True))
        self.half_dn = pg.InfiniteLine(angle=0, pen=theme.pen(theme.INK_FAINT, 1.0, dash=True))
        self.ep.addItem(self.half_up)
        self.ep.addItem(self.half_dn)
        self.ep.setXRange(0, 2.0, padding=0)
        self.ep.setMouseEnabled(x=False, y=False)

        top, _ = card(self.time, row(self.bits, self.amp, None))
        bottom, _ = card(self.err)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, self._side("THE COST OF A LEVEL")))
        self._watch(self.bits, self.amp)
        self.step = 0.0
        self.snr = 0.0
        self._redraw()

    def _redraw(self):
        bits = int(self.bits.value())
        amp = self.amp.value() / 100.0
        t = np.linspace(0.0, 2.0, 2000)
        x = amp * np.sin(2.0 * np.pi * t)
        q, self.step = theory.quantise(x, bits, 1.0)
        e = q - x
        self.truth.setData(t, x)
        self.held.setData(t, q)
        self.error.setData(t, e)
        self.half_up.setValue(self.step / 2.0)
        self.half_dn.setValue(-self.step / 2.0)
        lim = max(self.step, float(np.abs(e).max()) * 1.1, 1e-3)
        self.ep.setYRange(-lim, lim, padding=0)

        rms = float(np.sqrt(np.mean(e ** 2)))
        self.snr = theory.snr_db(x, e)
        self.stats.setText(table(
            cell("levels", "%d" % (2 ** bits), "= 2<sup>%d</sup>" % bits),
            cell("step &Delta;", "%.4g" % self.step, "of full scale 2"),
            cell("error, RMS", "%.4g" % rms, ""),
            cell("&Delta; / &radic;12", "%.4g" % (self.step / np.sqrt(12.0)), ""),
            cell("SNR, measured", "%.1f" % self.snr, "dB"),
            cell("6.02 N + 1.76", "%.1f" % theory.quantisation_snr(bits), "dB, full scale")))
        if amp > 1.0:
            self.verdict.setText(verdict(
                theme.WARN, "The signal is past full scale, so its tops are clipped. That "
                "error is not the small, even one of rounding: it is the peaks, cut off."))
        elif bits <= 3:
            self.verdict.setText(verdict(
                theme.WARN, "With %d levels the error is a staircase of the signal itself, "
                "not noise: it follows the waveform." % (2 ** bits)))
        else:
            self.verdict.setText(verdict(
                theme.GOOD, "The rounding error is spread evenly between &minus;&Delta;/2 and "
                "+&Delta;/2, so its RMS is &Delta;/&radic;12, and each extra bit halves it: "
                "6 dB a bit."))
        self.explain_changed.emit()

    def explain(self):
        return dict(
            head="Quantisation",
            equation="<span style='font-size:16pt'>&Delta; &nbsp;=&nbsp; FS / 2<sup>N</sup>"
                     "&nbsp;&nbsp;&nbsp;&nbsp; e<sub>RMS</sub> &nbsp;=&nbsp; &Delta; / "
                     "&radic;12 &nbsp;&nbsp;&nbsp;&nbsp; SNR &nbsp;&asymp;&nbsp; "
                     "<span style='color:%s'>6.02 N + 1.76</span> dB</span>" % theme.GUILTY,
            terms=(("N", "bits: the ADC's, or the ones kept afterwards"),
                   ("FS", "full scale, the whole range the levels are spread over"),
                   ("&Delta;", "one step: %.4g of a range of 2 now" % self.step),
                   ("<span style='color:%s'>SNR</span>" % theme.GUILTY,
                    "for a full-scale sine; measured now %.1f dB" % self.snr)),
            why="A number held to N bits can only be one of 2<sup>N</sup> values, so every "
                "sample is off by up to half a step. For a signal that uses the range, that "
                "error is as good as random and evenly spread, which gives &Delta;/&radic;12 "
                "and the 6 dB a bit. A signal that uses a small part of the range gets far "
                "fewer levels than the ADC has.",
            tip="Drop to 3 bits and watch the error trace turn into the signal's own shape. "
                "Then push the amplitude past 100 %: clipping is a different loss.",
            source=("Quantization (signal processing), Wikipedia",
                    "https://en.wikipedia.org/wiki/Quantization_(signal_processing)"),
            sources=(("Signal-to-quantization-noise ratio, Wikipedia",
                      "https://en.wikipedia.org/wiki/Signal-to-quantization-noise_ratio"),
                     ("Analog-to-digital converter, Wikipedia",
                      "https://en.wikipedia.org/wiki/Analog-to-digital_converter")))


# ---------------------------------------------------------------
# Fourier synthesis
# ---------------------------------------------------------------
class SynthesisPage(Demo):
    title = "Fourier synthesis"
    subtitle = "A square, a triangle or a sawtooth, built from sinusoids"

    def __init__(self):
        super().__init__()
        self.kind = combo(theory.WAVES, 96)
        self.n = Slider("harmonics", 1, 60, 5, 1, "", 0, width=190)

        self.time = pg.PlotWidget()
        tp = theme.plot(self.time, "time", "value", "s", "")
        theme.title(tp, "The wave, and the first harmonics added up")
        theme.legend(tp)
        self.truth = tp.plot(pen=theme.pen(theme.TRUTH, 1.2, dash=True), name="the wave")
        self.partial = tp.plot(pen=theme.pen(theme.ACCENT, 2.0), name="harmonics added up")
        tp.setXRange(0, 2.0, padding=0)
        tp.setYRange(-1.4, 1.4, padding=0)
        tp.setMouseEnabled(x=False, y=False)

        self.spec = pg.PlotWidget()
        self.sp = theme.plot(self.spec, "harmonic k", "|b_k|", "", "")
        theme.title(self.sp, "The harmonics: the ones added in colour, the rest in grey")
        self.rest = self.sp.plot(pen=theme.pen(theme.INK_FAINT, 2.0), connect="pairs")
        self.used = self.sp.plot(pen=theme.pen(theme.ACCENT, 4.0), connect="pairs")
        self.sp.setXRange(0, 61, padding=0)
        self.sp.setYRange(0, 1.4, padding=0)
        self.sp.setMouseEnabled(x=False, y=False)

        top, _ = card(self.time, row(caption("wave"), self.kind, self.n, None))
        bottom, _ = card(self.spec)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 3)
        col.addWidget(bottom, 2)
        self.body(row(col, self._side("ADDING UP")))
        self._watch(self.kind, self.n)
        self.overshoot = 0.0
        self._redraw()

    def _redraw(self):
        kind = theory.WAVES[self.kind.currentIndex()]
        n = int(self.n.value())
        t = np.linspace(0.0, 2.0, 2000)
        x = theory.waveform(kind, 1.0, t)
        y = theory.partial_sum(kind, 1.0, t, n)
        self.truth.setData(t, x)
        self.partial.setData(t, y)
        k, b = theory.harmonics(kind, 60)
        self.rest.setData(*stems(k[n:], np.abs(b[n:])))
        self.used.setData(*stems(k[:n], np.abs(b[:n])))

        err = float(np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x ** 2)) * 100)
        # Gibbs: the sum overshoots a jump by about 9 % of the jump's size, and
        # the jump here is 2, from -1 to +1
        self.overshoot = float((np.abs(y).max() - 1.0) / 2.0 * 100)
        present = int(np.count_nonzero(b[:n]))
        self.stats.setText(table(
            cell("harmonics added", "%d" % n, "(%d of them non-zero)" % present),
            cell("falling off as", {"square": "1 / k", "triangle": "1 / k&sup2;",
                                    "sawtooth": "1 / k"}[kind], ""),
            cell("error, RMS", "%.1f" % err, "% of the wave"),
            cell("overshoot at a jump", "%.1f" % max(self.overshoot, 0.0), "% of the jump")))
        if kind == "triangle":
            self.verdict.setText(verdict(
                theme.GOOD, "No jumps, so the harmonics fall off as 1/k&sup2; and a handful "
                "of them is nearly the whole wave."))
        else:
            self.verdict.setText(verdict(
                theme.WARN, "A jump needs every harmonic: they fall off only as 1/k, and "
                "however many are added the sum overshoots the edge by about 9 %. That is "
                "the Gibbs phenomenon, and it does not go away."))
        self.explain_changed.emit()

    def explain(self):
        return dict(
            head="Fourier series",
            equation="<span style='font-size:16pt'>x(t) &nbsp;=&nbsp; a<sub>0</sub> + "
                     "&sum;<sub>k</sub> [ a<sub>k</sub> cos(2&pi;k f t) + "
                     "<span style='color:%s'>b<sub>k</sub></span> sin(2&pi;k f t) ]"
                     "&nbsp;&nbsp;&nbsp;&nbsp; square: b<sub>k</sub> = 4 / &pi;k, "
                     "k odd</span>" % theme.GUILTY,
            terms=(("k f", "the harmonics: whole multiples of the fundamental"),
                   ("<span style='color:%s'>b<sub>k</sub></span>" % theme.GUILTY,
                    "how much of each; a jump makes them fall off as 1/k, a kink as 1/k&sup2;"),
                   ("overshoot", "%.1f %% now, and about 9 %% at a jump however many are added"
                    % max(self.overshoot, 0.0))),
            why="Any periodic signal is a sum of sinusoids at multiples of its own frequency. "
                "How fast the sum converges is set by how smooth the signal is: a corner "
                "costs a few harmonics, a jump costs all of them.",
            tip="Square wave, 1 harmonic, then 3, 9, 30, 60: the edges sharpen and the "
                "overshoot stays. Switch to the triangle and 5 harmonics is already hard to "
                "tell from the wave.",
            source=("Fourier series, Wikipedia", "https://en.wikipedia.org/wiki/Fourier_series"),
            sources=(("Gibbs phenomenon, Wikipedia",
                      "https://en.wikipedia.org/wiki/Gibbs_phenomenon"),
                     ("Square wave, Wikipedia", "https://en.wikipedia.org/wiki/Square_wave")))


# ---------------------------------------------------------------
# spectral leakage
# ---------------------------------------------------------------
FS_LEAK = 100.0
SIZES = ("64", "100", "128", "256", "512")


class LeakagePage(Demo):
    title = "Spectral leakage"
    subtitle = "A tone between bins, and what a window does about it"

    def __init__(self):
        super().__init__()
        self.f = Slider("tone", 5.0, 20.0, 10.0, 0.05, "Hz", 2, width=200)
        self.win = combo(theory.WINDOWS, 104)
        self.n = combo(SIZES, 64)
        self.n.setCurrentIndex(1)

        self.time = pg.PlotWidget()
        self.tp = theme.plot(self.time, "sample", "value", "", "")
        theme.title(self.tp, "The record that goes into the DFT: the tone through the window")
        theme.legend(self.tp)
        self.env_up = self.tp.plot(pen=theme.pen(theme.INK_FAINT, 1.0, dash=True), name="window")
        self.env_dn = self.tp.plot(pen=theme.pen(theme.INK_FAINT, 1.0, dash=True))
        self.seg = self.tp.plot(pen=theme.pen(theme.LINE, 1.4), name="windowed tone")
        self.tp.setYRange(-1.1, 1.1, padding=0)
        self.tp.setMouseEnabled(x=False, y=False)

        self.spec = pg.PlotWidget()
        self.sp = theme.plot(self.spec, "frequency", "magnitude", "Hz", "dB")
        theme.title(self.sp, "Its spectrum, bin by bin")
        theme.legend(self.sp)
        self.bins = self.sp.plot(pen=theme.pen(theme.ACCENT, 1.6), symbol="o", symbolSize=4,
                                 symbolBrush=theme.ACCENT, symbolPen=None, name="DFT bins")
        self.tone = pg.InfiniteLine(angle=90, pen=theme.pen(theme.TRUTH, 1.4, dash=True),
                                    label="the tone", labelOpts={"position": 0.9,
                                                                 "color": theme.INK_FAINT})
        self.sp.addItem(self.tone)
        self.sp.setXRange(0, FS_LEAK / 2.0, padding=0)
        self.sp.setYRange(-100, 5, padding=0)
        self.sp.setMouseEnabled(x=False, y=False)

        top, _ = card(self.time, row(self.f, None),
                      row(caption("window"), self.win, caption("N"), self.n, None))
        bottom, _ = card(self.spec)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addWidget(top, 2)
        col.addWidget(bottom, 3)
        self.body(row(col, self._side("BINS AND LOBES")))
        self._watch(self.f, self.win, self.n)
        self.leak_db = 0.0
        self._redraw()

    def _redraw(self):
        f = self.f.value()
        kind = theory.WINDOWS[self.win.currentIndex()]
        n = int(SIZES[self.n.currentIndex()])
        i = np.arange(n)
        x = np.cos(2.0 * np.pi * f * i / FS_LEAK)
        w = theory.window(kind, n)
        self.seg.setData(i, x * w)
        self.env_up.setData(i, w)
        self.env_dn.setData(i, -w)
        self.tp.setXRange(0, n - 1, padding=0)
        freq, db = theory.spectrum_db(x, FS_LEAK, w)
        self.bins.setData(freq, db)
        self.tone.setValue(f)

        spacing = FS_LEAK / n
        where = f / spacing
        on_bin = abs(where - round(where)) < 1e-6
        peak = int(np.argmax(db))
        width, side = theory.LOBES[kind]
        away = np.abs(np.arange(len(db)) - peak) > width
        self.leak_db = float(db[away].max()) if np.any(away) else -120.0
        self.stats.setText(table(
            cell("bin spacing f<sub>s</sub> / N", "%.3g" % spacing, "Hz"),
            cell("the tone sits at bin", "%.2f" % where, "on a bin" if on_bin else "between"),
            cell("main lobe", "%d" % width, "bins wide"),
            cell("highest sidelobe", "%.0f" % side, "dB, this window"),
            cell("outside the main lobe", "%.0f" % self.leak_db, "dB, measured")))
        if on_bin and kind == "rectangular":
            self.verdict.setText(verdict(
                theme.GOOD, "A whole number of cycles fits the record, so the tone lands in "
                "one bin and every other bin is empty. This is the only case where "
                "the rectangular window is the best one."))
        elif kind == "rectangular":
            self.verdict.setText(verdict(
                theme.WARN, "Not a whole number of cycles: the record ends mid-cycle, the "
                "DFT sees that step, and the tone's energy spreads across every bin, "
                "falling off only 6 dB per octave."))
        else:
            self.verdict.setText(verdict(
                theme.GOOD, "The window tapers the ends to zero, so there is no step to "
                "leak. The price is a main lobe %d bins wide instead of 2: two close "
                "tones blur together sooner." % width))
        self.explain_changed.emit()

    def explain(self):
        return dict(
            head="Spectral leakage",
            equation="<span style='font-size:16pt'>X[k] &nbsp;=&nbsp; &sum;<sub>n</sub> "
                     "<span style='color:%s'>w[n]</span> x[n] e<sup>&minus;j2&pi;kn/N</sup>"
                     "&nbsp;&nbsp;&nbsp;&nbsp; bins at k f<sub>s</sub> / N</span>"
                     % theme.GUILTY,
            terms=(("N", "samples in the record, %d now" % int(SIZES[self.n.currentIndex()])),
                   ("f<sub>s</sub> / N", "the bin spacing: the finest frequency the DFT "
                                          "can name"),
                   ("<span style='color:%s'>w[n]</span>" % theme.GUILTY,
                    "the window; rectangular means none. Outside the main lobe now: "
                    "%.0f dB" % self.leak_db)),
            why="The DFT treats the record as one period of something that repeats. A tone "
                "that does not complete a whole number of cycles then has a jump at the "
                "wrap, and a jump has energy at every frequency: that is the leakage. A "
                "window removes the jump by fading the ends, and pays in resolution.",
            tip="Rectangular, N = 100: the tone at 10.00 Hz is one bin; at 10.50 it is "
                "everywhere. Switch to Hann and the skirt drops 30 dB; to Blackman, 60.",
            source=("Spectral leakage, Wikipedia",
                    "https://en.wikipedia.org/wiki/Spectral_leakage"),
            sources=(("Window function, Wikipedia",
                      "https://en.wikipedia.org/wiki/Window_function"),
                     ("Hann function, Wikipedia", "https://en.wikipedia.org/wiki/Hann_function")))
