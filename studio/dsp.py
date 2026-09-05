# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
dsp.py
The signal processing the pages share, written so that it runs on a stream.

Everything here works on blocks of samples as they arrive and keeps its own
state between blocks, because that is what a live window needs: a filter that
has to see the whole record before it can produce the first sample cannot be
drawn while the record is still being made. The same objects are used by the
simulators, the sessions and the pages, so that a page never shows a result
computed one way and explains it another.

    Sos           a cascade of second-order sections, streaming
    butterworth   design one, with or without scipy
    notch         one section that removes a mains hum
    MovingRms     the envelope an EMG page draws
    PeakDetector  the beats a PPG page counts
    spectrum      a magnitude spectrum with a window
    median_frequency   what falls when a muscle tires
    spo2          the ratio of ratios, and the textbook line through it

scipy is used when it is installed, because it is faster. Everything is also
written out in numpy, so that the program behaves the same on a machine that
does not have it, and so that the thing being taught can be read.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy import signal as _sp
except ImportError:                                   # noqa: BLE001
    _sp = None


# ---------------------------------------------------------------
# second-order sections
# ---------------------------------------------------------------
def _rbj(kind, f0, hz, q):
    """
    One biquad from the Audio EQ Cookbook (R. Bristow-Johnson), normalized so
    that a0 = 1. Returns the six coefficients b0 b1 b2 a0 a1 a2 as one row.
    """
    w0 = 2.0 * math.pi * f0 / hz
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    elif kind == "notch":
        b = [1.0, -2 * cw, 1.0]
    elif kind == "bandpass":                          # constant 0 dB peak gain
        b = [alpha, 0.0, -alpha]
    else:
        raise ValueError(kind)
    a = [1 + alpha, -2 * cw, 1 - alpha]
    row = np.array(b + a, float) / a[0]
    return row


def _first_order(kind, f0, hz):
    """A single real pole, as a section with b2 = a2 = 0, for odd orders."""
    k = math.tan(math.pi * f0 / hz)
    if kind == "lowpass":
        b0 = b1 = k / (1 + k)
    else:
        b0, b1 = 1 / (1 + k), -1 / (1 + k)
    a1 = (k - 1) / (k + 1)
    return np.array([b0, b1, 0.0, 1.0, a1, 0.0])


def butterworth(order, kind, f, hz, f_hi=None):
    """
    A Butterworth filter as second-order sections, shape (n, 6).

        kind   "lowpass", "highpass" or "bandpass"
        f      the cut-off, or the lower edge of a band-pass
        f_hi   the upper edge of a band-pass

    Uses scipy when it is there. Without it, the same response is built from
    cookbook sections: a Butterworth of order N is N/2 second-order sections
    whose Q values are 1 / (2 sin((2k-1) pi / 2N)), plus one first-order
    section when N is odd. A band-pass is a high-pass followed by a low-pass.
    """
    order = max(1, int(order))
    ny = hz / 2.0
    if kind == "bandpass":
        lo, hi = max(float(f), 1e-3), min(float(f_hi), ny * 0.995)
        if hi <= lo:
            raise ValueError("band edges cross")
        if _sp is not None:
            return _sp.butter(order, [lo / ny, hi / ny], btype="bandpass", output="sos")
        return np.vstack([butterworth(order, "highpass", lo, hz),
                          butterworth(order, "lowpass", hi, hz)])
    fc = min(max(float(f), 1e-3), ny * 0.995)
    if _sp is not None:
        return _sp.butter(order, fc / ny, btype="lowpass" if kind == "lowpass" else "highpass",
                          output="sos")
    rows = []
    n_pairs = order // 2
    for k in range(1, n_pairs + 1):
        q = 1.0 / (2.0 * math.sin((2 * k - 1) * math.pi / (2 * order)))
        rows.append(_rbj(kind, fc, hz, q))
    if order % 2:
        rows.append(_first_order(kind, fc, hz))
    return np.array(rows)


def notch(f0, hz, q=30.0):
    """One section that takes out a hum at f0 and leaves the rest alone."""
    return _rbj("notch", f0, hz, q)[None, :]


class Sos:
    """
    A cascade of second-order sections that remembers where it got to.

    Feed it blocks of any length, in order, and the output is the same as if
    the whole signal had been filtered at once. That is the property a live
    page needs, and it is what a filter on a microcontroller does anyway.

    Each section is direct form II transposed:

        y[n]  = b0 x[n] + s1
        s1    = b1 x[n] - a1 y[n] + s2
        s2    = b2 x[n] - a2 y[n]
    """

    def __init__(self, sos, width=1):
        self.sos = np.atleast_2d(np.asarray(sos, float))
        self.width = int(width)
        self.reset()

    def reset(self):
        # state per section, per channel: two delays each
        self.z = np.zeros((len(self.sos), 2, self.width))

    def prime(self, x0):
        """
        Start as if the input had been x0 for ever.

        A filter that starts from empty delays sees the first sample as a step
        from zero. For a pulse oximeter that step is a hundred thousand
        counts, and the band-pass rings for seconds, twenty times taller than
        the pulse it is meant to show. Setting the state to the steady-state
        response for a constant x0 removes the step, which is what scipy's
        sosfilt_zi is for; this does the same arithmetic section by section.
        """
        x = np.broadcast_to(np.asarray(x0, float), (self.width,)).astype(float)
        for k, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            y = x * (b0 + b1 + b2) / (1.0 + a1 + a2)
            s2 = b2 * x - a2 * y
            s1 = b1 * x - a1 * y + s2
            self.z[k, 0], self.z[k, 1] = s1, s2
            x = y

    def process(self, x) -> np.ndarray:
        """Filter a block of shape (n,) or (n, width)."""
        x = np.asarray(x, float)
        flat = x.ndim == 1
        if flat:
            x = x[:, None]
        if x.shape[1] != self.width:
            raise ValueError("expected %d channels, got %d" % (self.width, x.shape[1]))
        if len(x) == 0:
            return x[:, 0] if flat else x
        if _sp is not None:
            y, self.z = _sp.sosfilt(self.sos, x, axis=0, zi=self.z)
        else:
            y = self._loop(x)
        return y[:, 0] if flat else y

    def _loop(self, x):
        y = x.copy()
        for k, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            s1, s2 = self.z[k, 0].copy(), self.z[k, 1].copy()
            for i in range(len(y)):
                v = y[i]
                out = b0 * v + s1
                s1 = b1 * v - a1 * out + s2
                s2 = b2 * v - a2 * out
                y[i] = out
            self.z[k, 0], self.z[k, 1] = s1, s2
        return y


def filtfilt(sos, x):
    """
    Forwards and then backwards, so nothing comes out late.

    Only possible on a record that is already complete, which is why the
    Filters page calls it "offline". Uses scipy when it can; otherwise the
    streaming filter run twice, with the ends padded by reflection.
    """
    x = np.asarray(x, float)
    if _sp is not None:
        return _sp.sosfiltfilt(sos, x)
    pad = min(len(x) - 1, 3 * len(sos) * 10)
    ext = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
    f = Sos(sos)
    y = f.process(ext)
    f.reset()
    y = f.process(y[::-1])[::-1]
    return y[pad:pad + len(x)]


def response(sos, hz, n=512):
    """|H(f)| and the phase, at n frequencies from 0 to the Nyquist rate."""
    if _sp is not None:
        w, h = _sp.sosfreqz(sos, worN=n, fs=hz)
        return w, np.abs(h), np.unwrap(np.angle(h))
    w = np.linspace(0, hz / 2.0, n)
    z = np.exp(-1j * 2 * np.pi * w / hz)
    h = np.ones_like(z)
    for b0, b1, b2, _a0, a1, a2 in np.atleast_2d(sos):
        h = h * (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
    return w, np.abs(h), np.unwrap(np.angle(h))


# ---------------------------------------------------------------
# envelopes and beats
# ---------------------------------------------------------------
class MovingRms:
    """
    The root mean square over the last W samples, one value per sample.

        rms[n] = sqrt( (1/W) sum_{k<W} x[n-k]^2 )

    A running sum of squares, so each sample costs the same however long the
    window is. The window can be changed while it runs.
    """

    def __init__(self, window: int, width: int = 1):
        self.width = int(width)
        self.set_window(window)

    def set_window(self, window: int):
        self.window = max(1, int(window))
        self._buf = np.zeros((self.window, self.width))
        self._sum = np.zeros(self.width)
        self._i = 0
        self._filled = 0

    def process(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        flat = x.ndim == 1
        if flat:
            x = x[:, None]
        out = np.empty_like(x)
        sq = x * x
        w = self.window
        for i in range(len(x)):
            self._sum += sq[i] - self._buf[self._i]
            self._buf[self._i] = sq[i]
            self._i = (self._i + 1) % w
            self._filled = min(w, self._filled + 1)
            # the sum can go a hair negative through rounding; never sqrt that
            out[i] = np.sqrt(np.maximum(self._sum, 0.0) / self._filled)
        return out[:, 0] if flat else out


class PeakDetector:
    """
    Find the beats in a pulse wave as it arrives.

    The systolic upstroke is the steepest thing in a pulse: four times steeper
    than the climb to the dicrotic hump behind it, whatever the rate. So a beat
    is armed when the slope rises past a fraction of the steepest recent
    upstroke, and confirmed at the next local maximum, which is the systolic
    peak. The slope scale follows the signal, so a finger pressed harder or
    softer does not stop the counting; the refractory interval follows the
    rhythm, so a slow heart is not counted twice a beat and a fast one is not
    halved.

    Peaks are confirmed one sample late, because a maximum cannot be known
    until the next sample is lower. `process` returns the beats it confirmed
    as (index into this block, or -1 for the previous block's last sample;
    height).
    """

    def __init__(self, hz, refractory=0.30, fraction=0.5):
        self.hz = float(hz)
        self.floor = int(refractory * hz)         # the least ever allowed
        self.fraction = fraction
        self.reset()

    def reset(self):
        self._prev = [0.0, 0.0]        # the two samples before this block
        self._since = 10 ** 9          # samples since the last beat
        self.level = 0.0               # the steepest recent upstroke, decaying
        self.refractory = self.floor
        self._ibi = []                 # the last few intervals, in samples
        self._armed = False
        self._age = 0                  # samples since arming
        self._steep = 0.0              # the steepest slope since arming
        self.n = 0                     # beats seen

    def process(self, x) -> list:
        x = np.asarray(x, float)
        beats = []
        p2, p1 = self._prev
        decay = math.exp(-1.0 / (4.0 * self.hz))       # the scale forgets over 4 s
        wait = int(0.4 * self.hz)                       # a peak must follow an upstroke soon
        for i in range(len(x)):
            v = x[i]
            self._since += 1
            slope = v - p1
            # The scale is the steepest recent upstroke. For the first few
            # beats it follows every slope, so that it has something to start
            # from; after that it is only raised at a confirmed beat, and by
            # at most a factor of two, so that one movement artifact does not
            # set a bar the real pulse then fails for seconds.
            self.level *= decay
            if self.n < 3:
                self.level = max(self.level, slope)
            if not self._armed:
                if (self._since >= self.refractory and slope > 0
                        and slope >= self.fraction * self.level):
                    self._armed = True
                    self._age = 0
                    self._steep = slope
            else:
                self._age += 1
                self._steep = max(self._steep, slope)
                if p1 > p2 and p1 >= v:                 # the peak the upstroke led to
                    beats.append((i - 1, p1))
                    if self.n:
                        median = sorted(self._ibi)[len(self._ibi) // 2] if self._ibi else None
                        # An interval twice the usual one is a missed beat, not
                        # a slower heart. Letting it into the list would lengthen
                        # the refractory and make the next miss more likely.
                        if median is None or self._since < 1.6 * median:
                            self._ibi.append(self._since)
                            self._ibi = self._ibi[-5:]
                        median = sorted(self._ibi)[len(self._ibi) // 2]
                        # under half the usual interval, never below the floor
                        # and never so long that a fast heart would be halved
                        self.refractory = int(min(max(0.4 * median, self.floor),
                                                  0.75 * self.hz))
                    self._since = 0
                    self.n += 1
                    self.level = max(self.level, min(self._steep, 2.0 * self.level))
                    self._armed = False
                elif self._age > wait:
                    self._armed = False
            p2, p1 = p1, v
        self._prev = [p2, p1]
        return beats


# ---------------------------------------------------------------
# spectra
# ---------------------------------------------------------------
def spectrum(x, hz, window="hann"):
    """
    Magnitude spectrum, scaled so that a sinusoid of amplitude A shows as A.

    Returns (frequency, magnitude). The mean is removed first, because a
    signal's offset is not a frequency anyone is looking for.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if window == "rect":
        w = np.ones(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "blackman":
        w = np.blackman(n)
    else:
        w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w)) * 2.0 / max(w.sum(), 1.0)
    return np.fft.rfftfreq(n, 1.0 / hz), mag


def median_frequency(x, hz):
    """
    The frequency that splits the power in half, and the power-weighted mean.

        MDF:   integral_0^MDF P(f) df  =  1/2 integral_0^inf P(f) df
        MNF:   integral f P(f) df / integral P(f) df

    Both fall as a muscle tires, because the conduction velocity of its
    fibers falls. Returns (mdf, mnf, freq, psd).
    """
    freq, mag = spectrum(x, hz)
    psd = mag * mag
    total = psd.sum()
    if total <= 0:
        return 0.0, 0.0, freq, psd
    cum = np.cumsum(psd)
    i = int(np.searchsorted(cum, total / 2.0))
    mdf = float(freq[min(i, len(freq) - 1)])
    mnf = float((freq * psd).sum() / total)
    return mdf, mnf, freq, psd


# ---------------------------------------------------------------
# pulse oximetry
# ---------------------------------------------------------------
def spo2(red, ir):
    """
    The ratio of ratios, and oxygen saturation from the textbook line.

        R    = (AC_red / DC_red) / (AC_ir / DC_ir)
        SpO2 = 110 - 25 R

    AC is the pulsatile part, taken here as the standard deviation of the
    window, and DC its mean. The line is the classical approximation; a real
    oximeter carries a calibration curve measured on volunteers, and this
    program does not, so the number is an estimate and says so.
    Returns (R, SpO2 in per cent), or (nan, nan) if there is no pulse to read.
    """
    red = np.asarray(red, float)
    ir = np.asarray(ir, float)
    if len(red) < 8 or len(ir) < 8:
        return float("nan"), float("nan")
    dc_r, dc_i = float(red.mean()), float(ir.mean())
    ac_r, ac_i = float(red.std()), float(ir.std())
    if dc_r <= 0 or dc_i <= 0 or ac_i <= 0 or ac_r <= 0:
        return float("nan"), float("nan")
    r = (ac_r / dc_r) / (ac_i / dc_i)
    return r, 110.0 - 25.0 * r
