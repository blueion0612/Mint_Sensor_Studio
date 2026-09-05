# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
bio.py
Sessions for the two physiological signals: what is derived from each as it
arrives, so that every page reads the same numbers.

    EmgSession    band-passed and de-hummed EMG, and its RMS envelope
    PpgSession    band-passed PPG, the beats found in it, and the heart rate

Both are built the way ImuSession is built: the raw channels are stored as they
came, and the derived columns sit beside them. A page that wants the envelope
reads a column; it does not filter anything itself, so two pages cannot
disagree about what the envelope is.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import core, dsp


class EmgSession(core.Session):
    """
    Surface EMG, cleaned up the standard way.

    The raw signal is band-passed to 20-450 Hz, which is where the muscle's own
    activity lies and below where the electrodes' movement lives, and a notch
    takes out the mains hum, because 50 Hz sits right in that band and no
    electrode arrangement keeps all of it out. The envelope is a moving RMS,
    the amplitude a clinician reads.

        columns    raw x n  |  filtered x n  |  rms x n
    """

    modality = "emg"
    BAND = (20.0, 450.0)
    ORDER = 4

    def __init__(self, channels, hz=1000.0, seconds=None, notch_hz=50.0, rms_ms=100.0):
        self.notch_hz = notch_hz
        self.rms_ms = float(rms_ms)
        super().__init__(channels, hz, seconds)

    def extra_columns(self) -> int:
        return 2 * self.n_raw

    def prepare(self):
        n = max(self.n_raw, 1)
        hi = min(self.BAND[1], 0.45 * self.hz)
        lo = self.BAND[0]
        if hi > lo * 1.5:
            sos = dsp.butterworth(self.ORDER, "bandpass", lo, self.hz, hi)
        else:
            # a slow board: keep what it can carry, above the movement artifact
            sos = dsp.butterworth(self.ORDER, "highpass", min(lo, 0.2 * self.hz), self.hz)
        self.band = dsp.Sos(sos, n)
        self.hum = self._make_notch(n)
        self.rms = dsp.MovingRms(int(self.rms_ms * 1e-3 * self.hz), n)
        self.filt_cols = slice(1 + self.n_raw, 1 + 2 * self.n_raw)
        self.rms_cols = slice(1 + 2 * self.n_raw, 1 + 3 * self.n_raw)

    def _make_notch(self, n):
        if self.notch_hz and self.notch_hz < 0.45 * self.hz:
            return dsp.Sos(dsp.notch(self.notch_hz, self.hz), n)
        return None

    def derive_block(self, raw, dts):
        x = raw[:, 1:1 + self.n_raw]
        if self.n_raw == 0:
            return None
        if self.n == 0:
            self.band.prime(x[0])        # no step from zero on the first sample
        y = self.band.process(x)
        if self.hum is not None:
            y = self.hum.process(y)
        return np.hstack([y, self.rms.process(y)])

    # ---- controls the pages offer ----
    def set_rms_window(self, ms: float):
        self.rms_ms = float(ms)
        self.rms.set_window(int(self.rms_ms * 1e-3 * self.hz))

    def set_notch(self, hz):
        """50, 60, or None for no notch at all."""
        self.notch_hz = hz
        self.hum = self._make_notch(max(self.n_raw, 1))

    def menu(self) -> list:
        out = []
        for i, c in enumerate(self.channels):
            out.append(("%s raw" % c.name, c.unit, core._column(1 + i)))
        for i, c in enumerate(self.channels):
            out.append(("%s filtered" % c.name, c.unit, core._column(1 + self.n_raw + i)))
        for i, c in enumerate(self.channels):
            out.append(("%s envelope" % c.name, c.unit, core._column(1 + 2 * self.n_raw + i)))
        return out


class PpgSession(core.Session):
    """
    A pulse wave, with its beats counted.

    The raw light level is mostly a constant, the DC part: skin, bone, and the
    blood that is always there. The heartbeat is the small AC part on top. A
    band-pass keeps 0.5-8 Hz, which is where a pulse and its harmonics live and
    above the slow wander of breathing and pressure, and the beats are found
    on that.

        columns    raw x n  |  filtered x n  |  beat  |  heart rate
    """

    modality = "ppg"
    BAND = (0.5, 8.0)

    def __init__(self, channels, hz=100.0, seconds=None):
        super().__init__(channels, hz, seconds)

    def extra_columns(self) -> int:
        return self.n_raw + 2

    def prepare(self):
        n = max(self.n_raw, 1)
        hi = min(self.BAND[1], 0.45 * self.hz)
        self.band = dsp.Sos(dsp.butterworth(2, "bandpass", self.BAND[0], self.hz, hi), n)
        self.detector = dsp.PeakDetector(self.hz)
        # Beats are counted on the infrared channel when there is one, because
        # it has the largest pulse; otherwise on the first channel.
        self.beat_channel = 0
        for i, c in enumerate(self.channels):
            if c.name.lower() == "ir":
                self.beat_channel = i
        self.filt_cols = slice(1 + self.n_raw, 1 + 2 * self.n_raw)
        self.BEAT = 1 + 2 * self.n_raw
        self.HR = self.BEAT + 1
        self.last_beat_t = None
        self.hr_now = 0.0
        self.beats = deque(maxlen=600)         # (t, ibi in seconds)
        self._pending = []                     # beats confirmed in the previous block

    def derive_block(self, raw, dts):
        if self.n_raw == 0:
            return None
        x = raw[:, 1:1 + self.n_raw]
        if self.n == 0:
            # The first sample is a hundred thousand counts of DC. Started from
            # empty, the band-pass would ring for seconds at twenty times the
            # pulse, and the beat detector would learn that as its scale.
            self.band.prime(x[0])
        y = self.band.process(x)
        k = len(raw)
        beat = np.zeros((k, 1))
        hr = np.zeros((k, 1))
        found = self.detector.process(y[:, self.beat_channel])
        self._pending = []
        for i, height in found:
            if i >= 0:
                t = float(raw[i, 0])
                beat[i, 0] = 1.0
            else:
                # the peak was the last sample of the previous block
                t = self._prev_t
                self._pending.append(-i)
            if self.last_beat_t is not None:
                ibi = t - self.last_beat_t
                if 0.3 <= ibi <= 2.0:                # 30 to 200 beats a minute
                    self.hr_now = 60.0 / ibi
                    self.beats.append((t, ibi))
            self.last_beat_t = t
            if i >= 0:
                hr[i:, 0] = self.hr_now
        hr[hr[:, 0] == 0.0, 0] = self.hr_now
        self._prev_t = float(raw[-1, 0])
        return np.hstack([y, beat, hr])

    def after_block(self, rows):
        # A peak confirmed one sample late belongs to the sample before this
        # block, which is already in the ring. Mark it there.
        for ago in self._pending:
            self.ring.poke(len(rows) - 1 + ago, self.BEAT, 1.0)
        self._pending = []

    # ---- what the heart-rate page reads ----
    def heart(self, seconds: float = 10.0) -> dict:
        """Heart rate and its variability over the last `seconds` of beats."""
        if not self.beats:
            return dict(hr=0.0, mean=0.0, rmssd=0.0, sdnn=0.0, n=0)
        t_end = self.beats[-1][0]
        recent = [ibi for t, ibi in self.beats if t >= t_end - seconds]
        ibi = np.asarray(recent, float)
        if len(ibi) < 2:
            return dict(hr=self.hr_now, mean=self.hr_now, rmssd=0.0, sdnn=0.0, n=len(ibi))
        # The rate is taken from the median interval, not the mean. One beat
        # missed or one artifact counted puts a doubled or halved interval in
        # the list, and the mean follows it; the median does not.
        return dict(hr=self.hr_now,
                    mean=float(60.0 / np.median(ibi)),
                    rmssd=float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000.0),
                    sdnn=float(ibi.std() * 1000.0),
                    n=len(ibi))

    def tachogram(self, seconds: float = 60.0):
        """(t, ibi in ms) of the recent beats, for drawing."""
        if not self.beats:
            return np.zeros(0), np.zeros(0)
        t_end = self.beats[-1][0]
        pts = [(t, ibi) for t, ibi in self.beats if t >= t_end - seconds]
        a = np.asarray(pts, float)
        return a[:, 0], a[:, 1] * 1000.0

    def menu(self) -> list:
        out = []
        for i, c in enumerate(self.channels):
            out.append(("%s raw" % c.name, c.unit, core._column(1 + i)))
        for i, c in enumerate(self.channels):
            out.append(("%s pulse" % c.name, c.unit, core._column(1 + self.n_raw + i)))
        return out
