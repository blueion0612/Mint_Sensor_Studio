# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
check_studio_math.py
Do the estimators say what they should, against a truth that is known?

The simulated sources write the signal first and derive the measurements from
it, so this can ask a question no bench test with real hardware can: not "does
the estimate drift" but "by how much is it wrong". Every number below is an
error against a truth, not against another estimate.

    IMU     the four position estimators, on both attitude backends, which
            must agree
    EMG     the RMS envelope against the envelope the muscle was made with,
            the hum the notch is meant to remove, the onset delay, and the
            median frequency falling when the muscle tires
    PPG     the beats found against the beats generated, at six heart rates,
            still and moving, and the saturation read back from the ratio
    DSP     the numpy filters against scipy's, section for section
    files   a recording written and played back, sample for sample

A teaching tool that behaves differently depending on what happens to be
installed is worse than one that is merely slow, which is why the two attitude
backends and the two filter engines are both required to agree.
"""

import os
import sys
import tempfile
import time

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

from studio import analysis, bio, core, dsp, sources                # noqa: E402

HZ = 100.0
RUN = 30.0
GYRO_BIAS = 0.8          # deg/s put into the simulated gyroscope
ACCEL_BIAS = 6.0         # mg put into the simulated accelerometer

fails = []


def want(label, got, lo, hi, unit="m"):
    ok = lo <= got <= hi
    print("  %-58s %9.4f %-4s %s" % (label, got, unit, "ok" if ok else "FAIL"))
    if not ok:
        fails.append("%s: %.4f %s, wanted %g to %g" % (label, got, unit, lo, hi))


def head(text):
    print()
    print("=" * 78)
    print("  " + text)
    print("=" * 78)
    print()


# ---------------------------------------------------------------
# IMU
# ---------------------------------------------------------------
def align_and_run(motion, seconds=RUN, bias=GYRO_BIAS, accel=ACCEL_BIAS):
    """
    The same two-stage alignment the Position page performs, then a run.

    Kept in step with pages_motion.PositionPage on purpose: if the page's
    alignment changes and this does not, the numbers here stop describing what
    a student sees.
    """
    src = sources.SimulatedSource(motion=motion, bias_dps=bias, accel_bias_mg=accel,
                                  realtime=False)
    ses = core.ImuSession(hz=HZ)

    # The board is still for the whole warm-up. The first few seconds are the
    # attitude filter starting up, and nothing is called stationary until it
    # has, so the window is taken from the end of that.
    src.generate(7.0)
    ses.add(src.read())
    block = ses.last(1.5)
    block = block[block[:, core.STILL] > 0.5]
    assert len(block) > 20, "no stationary samples to align on"

    # Exactly what the Position page does when Align and start is pressed: one
    # window of stillness, and everything read out of it at once.
    a0 = block[:, core.A].mean(axis=0)
    rot0 = core.attitude_from_gravity(a0)
    ses.set_gyro_bias(block[:, core.W].mean(axis=0))
    ses.set_attitude(rot0)
    e0 = rot0 @ a0 - np.array([0.0, 0.0, 1.0])

    ests = core.make_estimators()
    for e in ests:
        e.start(block[-1, core.T], a0, rot0, e0)
    src.set_origin()

    prev = block[-1, core.T]
    src.generate(seconds)
    ses.add(src.read())
    for r in ses.last(seconds + 2):
        if r[core.T] <= prev:
            continue
        dt = r[core.T] - prev
        prev = r[core.T]
        if 0 < dt < 1.0:
            for e in ests:
                e.step(r, dt)
    truth = float(np.linalg.norm(src.displacement))
    return truth, {e.name: e.distance for e in ests}, ses


def imu():
    for use in (True, False):
        if use and core.imufusion is None:
            print("  imufusion is not installed, only the numpy path is checked")
            continue
        core.USE_LIBRARY = use
        head("IMU: attitude by %s" % ("imufusion" if use else "numpy"))

        print("  board left alone for %.0f s, %.1f deg/s gyro bias, %.0f mg accel bias"
              % (RUN, GYRO_BIAS, ACCEL_BIAS))
        truth, got, ses = align_and_run("still")
        want("truth: it did not move", truth, 0.0, 0.001)
        # b t^2 / 2 with b the residual after alignment. The bounds are loose on
        # purpose: what is being checked is that each method fails the way it
        # is described as failing, not that it lands on a particular number.
        want("double integration is under a few metres", got["Double integration"], 0.0, 2.5)
        want("attitude tracking is too", got["Attitude tracking"], 0.0, 2.5)
        want("leaky integration is bounded", got["Leaky integration"], 0.0, 0.10)
        want("a zero-velocity update holds it at zero",
             got["Zero-velocity update"], 0.0, 0.02)
        order = (got["Zero-velocity update"] <= got["Leaky integration"]
                 <= min(got["Double integration"], got["Attitude tracking"]))
        want("and they come out in the order the page claims", 1.0 if order else 0.0,
             1.0, 1.0, "")
        want("the gyroscope offset was found, x", ses.gyro_bias()[0], 0.70, 0.90, "deg/s")

        print()
        print("  board tilted slowly, never translated")
        truth, got, _ = align_and_run("slow tilt")
        want("truth: it did not move", truth, 0.0, 0.001)
        want("double integration is destroyed by the tilt",
             got["Double integration"], 100.0, 1e9)
        want("attitude tracking is far better",
             got["Double integration"] / max(got["Attitude tracking"], 1e-9), 10.0, 1e9, "x")
        want("but is still far outside any tolerance", got["Attitude tracking"], 1.0, 1e9)

        print()
        print("  board pushed and stopped, three times")
        truth, got, _ = align_and_run("slide and stop")
        want("truth: it moved", truth, 1.5, 3.0)
        want("a zero-velocity update keeps the real distance, near enough",
             got["Zero-velocity update"] / truth, 0.4, 3.5, "x")
        want("leaky integration throws the real distance away",
             got["Leaky integration"] / truth, 0.0, 0.3, "x")
        want("and it keeps far less of it than the update does",
             got["Zero-velocity update"] / max(got["Leaky integration"], 1e-9), 2.0, 1e9, "x")

    head("IMU: the two attitude backends agree")
    for motion in ("still", "slow tilt", "shake 2 Hz"):
        angles = {}
        for use in (True, False):
            if use and core.imufusion is None:
                continue
            core.USE_LIBRARY = use
            src = sources.SimulatedSource(motion=motion, bias_dps=GYRO_BIAS,
                                          accel_bias_mg=ACCEL_BIAS, realtime=False)
            ses = core.ImuSession(hz=HZ)
            src.generate(40.0)
            ses.add(src.read())
            d = ses.last(2.0)
            angles[use] = np.array([core.euler_deg(q) for q in d[:, core.QUAT]]).mean(axis=0)
        if len(angles) == 2:
            gap = np.abs(angles[True][:2] - angles[False][:2]).max()
            want("%s: roll and pitch agree to" % motion, float(gap), 0.0, 2.5, "deg")
    core.USE_LIBRARY = True


# ---------------------------------------------------------------
# DSP
# ---------------------------------------------------------------
def filters():
    head("DSP: the numpy filters against scipy")
    if dsp._sp is None:
        print("  scipy is not installed, so the numpy engine is the only one here and "
              "there is nothing to compare it with")
        return
    from scipy import signal as sp
    for order in (1, 2, 4, 6):
        for kind in ("lowpass", "highpass"):
            real = dsp._sp
            dsp._sp = None
            mine = dsp.butterworth(order, kind, 30.0, 1000.0)
            dsp._sp = real
            ref = sp.butter(order, 30.0 / 500.0, btype=kind, output="sos")
            _, m1, _ = dsp.response(mine, 1000.0)
            _, h2 = sp.sosfreqz(ref, worN=512, fs=1000.0)
            want("%s order %d: response gap" % (kind, order),
                 float(np.abs(m1 - np.abs(h2)).max()), 0.0, 1e-6, "")
    x = np.random.default_rng(1).normal(size=3000)
    sos = sp.butter(4, [20 / 500, 450 / 500], btype="bandpass", output="sos")
    ref = sp.sosfilt(sos, x)
    real = dsp._sp
    dsp._sp = None
    f = dsp.Sos(sos, 1)
    loop = np.concatenate([f.process(x[i:i + 97]) for i in range(0, 3000, 97)])
    dsp._sp = real
    want("numpy Sos in 97-sample chunks equals scipy sosfilt",
         float(np.abs(loop - ref).max()), 0.0, 1e-9, "")
    g = dsp.Sos(sos, 1)
    g.prime(120000.0)
    zi = sp.sosfilt_zi(sos) * 120000.0
    want("prime() equals scipy sosfilt_zi", float(np.abs(g.z[:, :, 0] - zi).max()),
         0.0, 1e-6, "")
    y = g.process(np.full(50, 120000.0))
    want("a primed band-pass of a constant is zero from the first sample",
         float(np.abs(y).max()), 0.0, 1e-6, "")
    r = dsp.MovingRms(200, 1).process(np.sin(2 * np.pi * 50 * np.arange(2000) / 1000.0) * 2.0)
    want("moving RMS of a 2 V sine is 2/sqrt(2)", float(r[-1]), 1.40, 1.43, "V")


# ---------------------------------------------------------------
# EMG
# ---------------------------------------------------------------
def emg():
    head("EMG: the envelope against the muscle it was made with")
    src = sources.SimulatedEmg(pattern="bursts", amplitude_mv=1.0, hum_mv=0.2, realtime=False)
    ses = bio.EmgSession(src.channels, 1000.0)
    src.generate(24.0)
    ses.add(src.read())
    d = ses.last(20.0)
    t = d[:, core.T]
    rms = d[:, ses.rms_cols][:, 0]
    filt = d[:, ses.filt_cols][:, 0]
    truth = src.true_envelope(t)
    on, off = truth > 0.5, truth <= 0.5
    want("envelope during a 1 mV contraction", float(np.median(rms[on])), 0.75, 1.25, "mV")
    want("envelope at rest", float(np.median(rms[off])), 0.0, 0.03, "mV")
    f, mag = dsp.spectrum(filt[off][:4000], 1000.0)
    want("0.2 mV of 50 Hz hum, after the notch", float(mag[np.argmin(abs(f - 50))]),
         0.0, 0.02, "mV")
    # the onset rule the Envelope page uses, with its default k = 4
    q = np.percentile(rms, 10)
    quiet = rms[rms <= np.percentile(rms, 25)]
    level = float(q) + 4.0 * max(float(quiet.std()), 0.002)
    detected = rms > level
    active = src.true_active(t)
    rise_true = np.flatnonzero(np.diff(active.astype(int)) > 0)
    rise_det = np.flatnonzero(np.diff(detected.astype(int)) > 0)
    lags = []
    for i in rise_true:
        later = rise_det[(t[rise_det] >= t[i] - 0.05) & (t[rise_det] <= t[i] + 0.6)]
        if len(later):
            lags.append(t[later[0]] - t[i])
    want("every true onset was detected", float(len(lags)) / max(len(rise_true), 1),
         1.0, 1.0, "")
    want("and detected within the RMS window", float(np.mean(lags)) if lags else 9.0,
         0.0, 0.15, "s")
    want("with no false onsets at rest", float(len(rise_det) - len(lags)), 0.0, 1.0, "")

    src = sources.SimulatedEmg(pattern="fatigue", realtime=False)
    ses = bio.EmgSession(src.channels, 1000.0)
    src.generate(50.0)
    ses.add(src.read())
    d = ses.last(47.0)
    filt = d[:, ses.filt_cols][:, 0]
    t = d[:, core.T]
    early = dsp.median_frequency(filt[(t > 4) & (t < 6)], 1000.0)[0]
    late = dsp.median_frequency(filt[(t > 42) & (t < 44)], 1000.0)[0]
    want("fresh muscle, median frequency", early, 100.0, 200.0, "Hz")
    want("tired muscle, forty seconds on, has fallen by", early - late, 20.0, 120.0, "Hz")


# ---------------------------------------------------------------
# PPG
# ---------------------------------------------------------------
def ppg():
    head("PPG: the beats found against the beats generated")
    for bpm in (45.0, 60.0, 72.0, 100.0, 150.0, 175.0):
        for motion in (0.0, 0.5):
            src = sources.SimulatedPpg(bpm=bpm, hrv=4.0, spo2=95.0, motion=motion, hum=0.05,
                                       realtime=False)
            ses = bio.PpgSession(src.channels, 100.0)
            src.generate(40.0)
            ses.add(src.read())
            d = ses.last(30.0)
            beats = d[d[:, ses.BEAT] > 0.5, core.T]
            truth = np.array([b for b in src.beat_times if b >= d[0, core.T]])
            h = ses.heart(20.0)
            tag = "%3.0f BPM, %s" % (bpm, "still" if motion == 0 else "moving")
            if motion == 0.0:
                want("%s: heart rate read back" % tag, h["mean"] - bpm, -1.5, 1.5, "BPM")
                want("%s: beats found of %d" % (tag, len(truth)), float(len(beats)),
                     len(truth) - 1, len(truth) + 1, "")
                near = ([min(abs(b - truth - 0.15 * 60 / bpm)) for b in beats]
                        if len(beats) and len(truth) else [9.0])
                want("%s: worst distance from a true peak" % tag, max(near), 0.0, 0.12, "s")
            else:
                # Movement puts a 6 Hz artifact into the signal every nine
                # seconds. A simple detector is expected to stumble on it, not
                # to fall over: the count and the rate stay within a few per cent.
                want("%s: heart rate within 15 %%" % tag, abs(h["mean"] - bpm) / bpm,
                     0.0, 0.15, "")
                want("%s: beats found of %d" % (tag, len(truth)), float(len(beats)),
                     len(truth) * 0.88 - 1, len(truth) * 1.12 + 1, "")
    print()
    for target in (90.0, 95.0, 99.0):
        src = sources.SimulatedPpg(bpm=72.0, spo2=target, realtime=False)
        ses = bio.PpgSession(src.channels, 100.0)
        src.generate(12.0)
        ses.add(src.read())
        raw = ses.raw(ses.last(4.0))
        _, s = dsp.spo2(raw[:, 1], raw[:, 0])
        want("SpO2 set to %.0f, read back" % target, s, target - 2.5, target + 2.5, "%")


# ---------------------------------------------------------------
# files
# ---------------------------------------------------------------
def files():
    head("Files: a recording written and played back")
    src = sources.SimulatedPpg(realtime=False)
    src.generate(3.0)
    samples = src.read()
    path = os.path.join(tempfile.gettempdir(), "studio_roundtrip.csv")
    rec = sources.Recorder(path, src.channels)
    rec.feed(samples)
    rec.close()
    fs = sources.FileSource(path, speed=0, loop=False)
    time.sleep(0.5)
    back = fs.read()
    fs.close()
    want("modality read back from the header", 1.0 if fs.modality == "ppg" else 0.0, 1, 1, "")
    want("samples played back of %d" % len(samples), float(len(back)), len(samples),
         len(samples), "")
    gap = max(abs(a[1][0] - b[1][0]) for a, b in zip(samples, back)) if back else 9.0
    want("largest difference in a value", gap, 0.0, 0.01, "")
    want("rate read back", fs.hz, 99.0, 101.0, "Hz")
    os.remove(path)
    real = sources.FileSource(os.path.join(APP, "data", "still_30s.csv"),
                              speed=0, loop=False)
    time.sleep(0.6)
    rows = real.read()
    real.close()
    want("the sample recording plays back as an IMU", 1.0 if real.modality == "imu" else 0.0,
         1, 1, "")
    want("with its samples", float(len(rows)), 2000, 4000, "")


# ---------------------------------------------------------------
# the analysis pages
# ---------------------------------------------------------------
def analyses():
    head("Analysis: periodicity, time-frequency, smoothing, separation")
    src = sources.SimulatedSource(motion="shake 2 Hz", realtime=False)
    ses = core.ImuSession(hz=HZ)
    src.generate(20.0)
    ses.add(src.read())
    d = ses.last(8.0)
    x = d[:, core.AX] * core.G
    f0, r, _, _ = analysis.fundamental(x, HZ, f_lo=0.25, f_hi=25.0)
    want("a 2 Hz shake: fundamental found", f0, 1.95, 2.05, "Hz")
    want("and it repeats (r at the peak)", r, 0.8, 1.0, "")
    k, amp, _ph, recon = analysis.fourier_series(d[:, core.T] - d[0, core.T], x, f0, 5)
    want("first harmonic carries the shake", amp[0] / max(np.abs(x - x.mean()).max(), 1e-9),
         0.85, 1.05, "x")
    want("five harmonics explain the window", 1 - np.var(x - recon) / np.var(x), 0.9, 1.0, "")

    src = sources.SimulatedSource(motion="sweep 1-8 Hz", realtime=False)
    ses = core.ImuSession(hz=HZ)
    # the sweep starts after the 8 s warm-up and starts again at 28 s; stay inside
    src.generate(27.0)
    ses.add(src.read())
    d = ses.last(17.0)
    freq, times, mag = analysis.stft(d[:, core.AX] * core.G, HZ, 100)
    early = float(freq[int(np.argmax(mag[1:, 3])) + 1])
    late = float(freq[int(np.argmax(mag[1:, -4])) + 1])
    want("a sweep: the spectrogram's band rises by", late - early, 3.0, 8.0, "Hz")

    h = analysis.savgol_kernel(21, 3)
    want("Savitzky-Golay kernel sums to one", float(h.sum()), 0.999999, 1.000001, "")
    if dsp._sp is not None:
        from scipy.signal import savgol_coeffs
        want("and equals scipy's", float(np.abs(h - savgol_coeffs(21, 3)).max()), 0.0, 1e-9, "")
    spike = np.zeros(50)
    spike[25] = 9.0
    want("a median of 5 removes a lone spike", float(analysis.moving_median(spike, 5).max()),
         0.0, 0.0, "")

    src = sources.SimulatedEmg(pattern="bursts", channels=2, realtime=False)
    ses = bio.EmgSession(src.channels, 1000.0)
    src.generate(20.0)
    ses.add(src.read())
    d = ses.last(12.0)
    X = ses.raw(d)
    truth = src.true_sources(d[:, core.T])
    comps, _W = analysis.fastica(X)
    m = analysis.match(comps, truth)
    want("two mixed muscles: ICA finds both", float(len({i for i, _ in m})), 2, 2, "")
    want("and each matches its source to", float(min(r for _, r in m)), 0.9, 1.0, "|r|")
    pcs, _V, share = analysis.pca(X)
    want("PCA's first share is the larger", float(share[0]), 0.5, 1.0, "")


def main():
    imu()
    filters()
    emg()
    ppg()
    analyses()
    files()
    print()
    if fails:
        print("  %d check(s) failed:" % len(fails))
        for f in fails:
            print("    - " + f)
        return 1
    print("  every estimator did what it claims to do, on every sensor, on both backends")
    return 0


if __name__ == "__main__":
    sys.exit(main())
