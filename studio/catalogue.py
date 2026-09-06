# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
catalogue.py
Every page, in the groups the rail shows them in.

One group per sensor, holding its live view and the functions that only make
sense for it, and one group of functions that work on whichever signal is
connected; and one of demonstrations on synthetic signals, which need no
sensor at all. This is the file to edit to add a page or move one.
"""

from __future__ import annotations


class Group:
    def __init__(self, key, name, blurb, pages):
        self.key = key
        self.name = name
        self.blurb = blurb
        self.pages = list(pages)        # (page key, modality or None), in rail order


# page key -> module, class
PAGES = {
    "signals":     ("pages_signal",    "SignalsPage"),
    "frames":      ("pages_frames",    "FramesPage"),
    "orientation": ("pages_motion",    "OrientationPage"),
    "position":    ("pages_motion",    "PositionPage"),
    "envelope":    ("pages_emg",       "EnvelopePage"),
    "fatigue":     ("pages_emg",       "FatiguePage"),
    "heartrate":   ("pages_ppg",       "HeartRatePage"),
    "spo2":        ("pages_ppg",       "Spo2Page"),
    "sampling":    ("pages_signal",    "SamplingPage"),
    "convolution": ("pages_transform", "ConvolutionPage"),
    "periodicity": ("pages_analysis",  "PeriodicityPage"),
    "spectrum":    ("pages_transform", "SpectrumPage"),
    "spectrogram": ("pages_analysis",  "SpectrogramPage"),
    "filters":     ("pages_transform", "FilterPage"),
    "denoise":     ("pages_analysis",  "DenoisePage"),
    "separation":  ("pages_analysis",  "SeparationPage"),
    "theorem":     ("pages_theory",    "TheoremPage"),
    "aliasing":    ("pages_theory",    "AliasPage"),
    "quantise":    ("pages_theory",    "QuantisePage"),
    "synthesis":   ("pages_theory",    "SynthesisPage"),
    "leakage":     ("pages_theory",    "LeakagePage"),
}

GROUPS = [
    Group("imu", "IMU", "Accelerometer, gyroscope, magnetometer",
          [("signals", "imu"), ("frames", None), ("orientation", None), ("position", None)]),
    Group("emg", "EMG", "Surface electromyography",
          [("signals", "emg"), ("envelope", None), ("fatigue", None)]),
    Group("ppg", "PPG", "Photoplethysmography",
          [("signals", "ppg"), ("heartrate", None), ("spo2", None)]),
    Group("signal", "Signal processing", "Works on whichever signal is connected",
          [("sampling", None), ("convolution", None), ("periodicity", None),
           ("spectrum", None), ("spectrogram", None), ("filters", None),
           ("denoise", None), ("separation", None)]),
    Group("theory", "Signal theory", "Demonstrations on synthetic signals; no sensor needed",
          [("theorem", None), ("aliasing", None), ("quantise", None),
           ("synthesis", None), ("leakage", None)]),
]


def make(key, modality=None):
    """One page by key. Imported late, so a page that cannot load is one missing
    page rather than a program that will not start."""
    import importlib
    module, cls = PAGES[key]
    page = getattr(importlib.import_module("." + module, __package__), cls)()
    if modality is not None:
        page.modalities = (modality,)
    page.slot = key if modality is None else "%s.%s" % (modality, key)
    return page


def build(groups=None):
    """[(group, key, page)] in rail order."""
    out = []
    for group in (GROUPS if groups is None else groups):
        for key, modality in group.pages:
            out.append((group, key, make(key, modality)))
    return out
