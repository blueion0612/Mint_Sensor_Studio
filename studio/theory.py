# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
theory.py
The arithmetic behind the Signal theory pages: synthetic signals whose truth
is known exactly, so that what sampling, quantising and windowing do to a
signal can be shown against what the signal really was.

    reconstruct              the samples put back into a continuous signal
    alias, signed_alias      where a frequency lands once it has been sampled
    quantise, snr_db         a value held to 2^bits levels, and what that costs
    waveform, harmonics,     a square, triangle or sawtooth, and the sinusoids
    partial_sum              that add up to it
    window, spectrum_db      a segment through a window, and its spectrum

Nothing here reads a sensor. Each function is a few lines a student could
write themselves.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------
# sampling
# ---------------------------------------------------------------
def reconstruct(tn, xn, t, fs):
    """
    Whittaker–Shannon interpolation: the one band-limited signal that passes
    through the samples,

        x(t) = sum_n x[n] sinc( f_s (t - t_n) ).

    Exact when the signal held nothing at or above f_s / 2 and the samples go
    on for ever. With a finite record it is exact away from the ends, so the
    pages sample well past what they show.
    """
    tn = np.asarray(tn, float)
    xn = np.asarray(xn, float)
    t = np.asarray(t, float)
    if len(tn) == 0:
        return np.zeros_like(t)
    return np.sinc(fs * (t[:, None] - tn[None, :])) @ xn


def alias(f, fs):
    """
    The frequency a sampled cosine of f appears to have: |f - k f_s| for the
    k that brings it closest to zero. Below f_s / 2 that is f itself.
    """
    f = np.asarray(f, float)
    k = np.round(f / fs)
    return np.abs(f - k * fs)


def signed_alias(f, fs):
    """The same, with its sign: the cosine through the samples is cos(2 pi (f - k f_s) t)."""
    k = round(f / fs)
    return f - k * fs


# ---------------------------------------------------------------
# quantisation
# ---------------------------------------------------------------
def quantise(x, bits, full_scale=1.0):
    """
    x held to 2^bits levels spread over -full_scale to +full_scale (a mid-rise
    quantiser), and the size of one step. Anything past full scale is clipped
    to the outermost level, which is a different loss from rounding.
    """
    step = 2.0 * full_scale / (2 ** int(bits))
    q = (np.floor(np.asarray(x, float) / step) + 0.5) * step
    top = full_scale - step / 2.0
    return np.clip(q, -top, top), step


def snr_db(signal, error):
    """Signal power over error power, in dB."""
    s = float(np.mean(np.square(np.asarray(signal, float))))
    e = float(np.mean(np.square(np.asarray(error, float))))
    if e <= 0.0:
        return float("inf")
    return 10.0 * np.log10(s / e) if s > 0.0 else -float("inf")


def quantisation_snr(bits):
    """The textbook figure for a full-scale sine: 6.02 N + 1.76 dB."""
    return 6.02 * bits + 1.76


# ---------------------------------------------------------------
# Fourier synthesis
# ---------------------------------------------------------------
WAVES = ("square", "triangle", "sawtooth")


def waveform(kind, f, t):
    """The exact wave of amplitude 1 that the harmonics below add up to."""
    ph = f * np.asarray(t, float)
    if kind == "square":
        return np.where(np.sin(2.0 * np.pi * ph) >= 0.0, 1.0, -1.0)
    if kind == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * ph))
    if kind == "sawtooth":
        return 2.0 * (ph - np.floor(ph + 0.5))
    raise ValueError(kind)


def harmonics(kind, n):
    """
    (k, b_k) for the first n harmonics of a unit wave, x(t) = sum_k b_k sin(2 pi k f t).
    A zero where the wave has no such harmonic: the square and the triangle
    have odd ones only.
    """
    k = np.arange(1, int(n) + 1)
    odd = k % 2 == 1
    if kind == "square":
        b = np.where(odd, 4.0 / (np.pi * k), 0.0)
    elif kind == "triangle":
        sign = np.where((k // 2) % 2 == 0, 1.0, -1.0)
        b = np.where(odd, 8.0 / (np.pi ** 2 * k ** 2) * sign, 0.0)
    elif kind == "sawtooth":
        b = 2.0 / (np.pi * k) * np.where(odd, 1.0, -1.0)
    else:
        raise ValueError(kind)
    return k, b


def partial_sum(kind, f, t, n):
    """The first n harmonics added up."""
    k, b = harmonics(kind, n)
    t = np.asarray(t, float)
    return np.sin(2.0 * np.pi * f * np.outer(t, k)) @ b


# ---------------------------------------------------------------
# windows and leakage
# ---------------------------------------------------------------
WINDOWS = ("rectangular", "Hann", "Hamming", "Blackman")

# main-lobe width in bins, and the highest sidelobe in dB, from the textbooks
LOBES = {"rectangular": (2, -13.3), "Hann": (4, -31.5),
         "Hamming": (4, -42.7), "Blackman": (6, -58.1)}


def window(kind, n):
    if kind == "rectangular":
        return np.ones(int(n))
    if kind == "Hann":
        return np.hanning(int(n))
    if kind == "Hamming":
        return np.hamming(int(n))
    if kind == "Blackman":
        return np.blackman(int(n))
    raise ValueError(kind)


def spectrum_db(x, fs, w=None, floor_db=-120.0):
    """
    The magnitude spectrum of x through the window w, in dB, with 0 dB being
    a tone of amplitude 1 that sits exactly on a bin. Returns (frequencies,
    dB).
    """
    x = np.asarray(x, float)
    n = len(x)
    w = np.ones(n) if w is None else np.asarray(w, float)
    mag = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    return freq, 20.0 * np.log10(np.maximum(mag, 10.0 ** (floor_db / 20.0)))
