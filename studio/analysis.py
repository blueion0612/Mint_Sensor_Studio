# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
analysis.py
The arithmetic behind the four analysis pages, in plain numpy.

    autocorrelation, fundamental      does the signal repeat, and how often
    fourier_series                    the harmonics of that repetition
    stft                              the spectrum as it changes
    savgol_kernel, moving_median      two smoothers numpy does not ship
    pca, fastica, match               pulling mixed channels apart

Each function is a few lines a student could write themselves.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------
# periodicity
# ---------------------------------------------------------------
def autocorrelation(x):
    """r[k] for k = 0..N-1, normalized so that r[0] = 1. Mean removed first."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 4 or not np.any(x):
        return np.zeros(max(n, 1))
    size = 1 << (2 * n - 1).bit_length()           # by FFT: n log n, not n squared
    X = np.fft.rfft(x, size)
    r = np.fft.irfft(X * np.conj(X), size)[:n]
    return r / r[0] if r[0] > 0 else r


def fundamental(x, hz, f_lo=0.3, f_hi=None):
    """
    The rate at which a signal repeats, from the first peak of its
    autocorrelation between lags 1/f_hi and 1/f_lo, refined by a parabola
    through the three samples around the peak.

    Returns (f0 in Hz, r at the peak, the autocorrelation, its lags in
    seconds). An r under about 0.3 means the signal does not really repeat.
    """
    r = autocorrelation(x)
    n = len(r)
    lags = np.arange(n) / float(hz)
    if f_hi is None:
        f_hi = hz / 4.0
    lo = max(2, int(hz / f_hi))
    hi = min(n - 2, int(hz / f_lo))
    if hi <= lo + 2:
        return 0.0, 0.0, r, lags
    i = lo + int(np.argmax(r[lo:hi]))
    if r[i] <= 0:
        return 0.0, float(r[i]), r, lags
    a, b, c = r[i - 1], r[i], r[i + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    shift = max(-1.0, min(1.0, shift))           # a flat top can send this anywhere
    lag = (i + shift) / float(hz)
    return (1.0 / lag if lag > 0 else 0.0), float(b), r, lags


def fourier_series(t, x, f0, harmonics):
    """
    Amplitude and phase of the first `harmonics` multiples of f0 in x(t), and
    the partial sum built from them.

        c_k = (2/N) sum x[n] exp(-j 2 pi k f0 t[n])
        x(t) ~ a0 + sum_k |c_k| cos(2 pi k f0 t + arg c_k)

    Returns (k, |c_k|, arg c_k, reconstruction).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    a0 = float(x.mean())
    k = np.arange(1, int(harmonics) + 1)
    if f0 <= 0 or len(x) < 4:
        return k, np.zeros(len(k)), np.zeros(len(k)), np.full(len(x), a0)
    basis = np.exp(-2j * np.pi * np.outer(k, f0 * t))          # harmonics x samples
    c = 2.0 * (basis @ (x - a0)) / len(x)
    recon = a0 + np.real(np.conj(basis).T @ c)
    return k, np.abs(c), np.angle(c), recon


# ---------------------------------------------------------------
# time-frequency
# ---------------------------------------------------------------
def stft(x, hz, window=256, hop=None):
    """
    Short-time Fourier transform with a Hann window.

    Returns (frequencies, frame centers in seconds from the start, |X| as
    bins x frames). Frames are `hop` samples apart, a quarter of the window
    unless asked otherwise.
    """
    x = np.asarray(x, float)
    window = int(max(8, min(window, len(x))))
    hop = int(max(1, window // 4 if hop is None else hop))
    n = 1 + (len(x) - window) // hop
    if n < 1:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    idx = np.arange(window)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(window)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1)).T * 2.0 / window
    freq = np.fft.rfftfreq(window, 1.0 / hz)
    times = (hop * np.arange(n) + window / 2.0) / hz
    return freq, times, mag


# ---------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------
def savgol_kernel(window, order=3):
    """
    The Savitzky-Golay kernel. Convolving with it gives, at each sample, the
    value at the center of the least-squares polynomial of `order` fitted
    through the `window` samples around it.
    """
    window = int(window) | 1                     # odd
    half = window // 2
    j = np.arange(-half, half + 1, dtype=float)
    A = np.vander(j, int(order) + 1, increasing=True)
    h = np.linalg.pinv(A)[0]                     # the row that evaluates the fit at j = 0
    return h[::-1]


def moving_median(x, window):
    """The median of the `window` samples around each sample, edges held."""
    x = np.asarray(x, float)
    window = int(window) | 1
    if len(x) < window:
        return x.copy()
    half = window // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(view, axis=1)


# ---------------------------------------------------------------
# separation
# ---------------------------------------------------------------
def pca(X):
    """
    Principal components of X (samples x channels).

    Returns (components, in order of variance; the unit vectors they lie
    along, as columns; the share of the variance each carries).
    """
    X = np.asarray(X, float)
    X = X - X.mean(axis=0)
    C = np.cov(X.T) if X.shape[1] > 1 else np.array([[X.var()]])
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = np.maximum(w[order], 0.0), V[:, order]
    share = w / w.sum() if w.sum() > 0 else w
    return X @ V, V, share


def fastica(X, iterations=200, tol=1e-6, seed=0):
    """
    Independent components of X (samples x channels): FastICA with the tanh
    contrast and symmetric decorrelation.

    Whitens first, so every direction has unit variance, then turns the axes
    until the components are as far from Gaussian as they can be. Sources
    that were added together come apart because a sum is more Gaussian than
    its parts. Returns (components, the unmixing matrix W with S = Xc @ W).
    """
    X = np.asarray(X, float)
    X = X - X.mean(axis=0)
    n, m = X.shape
    if m < 2 or n < 8:
        return X.copy(), np.eye(m)
    w, V = np.linalg.eigh(np.cov(X.T))
    w = np.maximum(w, 1e-12)
    whiten = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    Z = X @ whiten
    W = _decorrelate(np.random.default_rng(seed).normal(size=(m, m)))
    for _ in range(int(iterations)):
        g = np.tanh(Z @ W)
        W_new = _decorrelate((Z.T @ g) / n - W * (1.0 - g * g).mean(axis=0)[None, :])
        done = float(np.abs(np.abs(np.sum(W_new * W, axis=0)) - 1).max()) < tol
        W = W_new
        if done:
            break
    S = Z @ W
    signs = np.sign(S[np.argmax(np.abs(S), axis=0), np.arange(m)])
    signs[signs == 0] = 1.0                       # a source comes back with any sign; pick one
    return S * signs, whiten @ W * signs


def _decorrelate(W):
    """W (W^T W)^-1/2: the columns made orthonormal."""
    u, _s, vt = np.linalg.svd(W, full_matrices=False)
    return u @ vt


def kurtosis(x):
    """Excess kurtosis: 0 for Gaussian, positive for peaky, negative for flat."""
    x = np.asarray(x, float)
    x = x - x.mean()
    v = float(np.mean(x * x))
    return float(np.mean(x ** 4) / (v * v) - 3.0) if v > 0 else 0.0


def match(components, sources):
    """For each component, (the source it correlates with best, |r|)."""
    out = []
    S = np.asarray(sources, float)
    S = S - S.mean(axis=0)
    for j in range(components.shape[1]):
        c = components[:, j] - components[:, j].mean()
        best, best_r = 0, 0.0
        for i in range(S.shape[1]):
            den = np.linalg.norm(c) * np.linalg.norm(S[:, i])
            r = abs(float(c @ S[:, i]) / den) if den > 0 else 0.0
            if r > best_r:
                best, best_r = i, r
        out.append((best, best_r))
    return out
