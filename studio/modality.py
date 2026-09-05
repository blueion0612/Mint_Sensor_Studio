# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
modality.py
What kind of signal is on the wire, and what its channels are.

A source produces rows of numbers. This module says what each number is: the
channel's name, its unit, its color on a plot, and which group of channels it
belongs to. It also says which *modality* a set of channels adds up to, because
that is what decides which pages apply.

    IMU     accelerometer, gyroscope, magnetometer     nine channels
    EMG     surface electromyography                   one to four channels
    PPG     photoplethysmography                       one to three wavelengths

Channels are named on the wire as `name_unit`, the convention the Week 2 board
already uses (`ax_g`, `gx_dps`, `mx_ut`). An EMG board sends `emg1_mv`; a pulse
oximeter sends `ir_raw` and `red_raw`. The unit suffix is looked up below, and
the modality is read off the names, so a recording made by any of the boards
identifies itself from its own header row.

Nothing here imports Qt, so it can be used by the checks and the recorders
without a display.
"""

from __future__ import annotations

# unit suffix on the wire  ->  how it is written on screen
UNITS = {
    "g": "g", "ms2": "m/s²", "dps": "°/s", "ut": "µT",
    "mv": "mV", "uv": "µV", "v": "V", "raw": "counts", "au": "a.u.",
    "bpm": "BPM", "deg": "°", "pct": "%", "": "",
}

# the three axis colors, the same three the IMU pages use
AXIS = ("#d4453c", "#2e9e4f", "#2f6fd0")
# and a longer run for channels that are not axes
SERIES = ("#e08a1e", "#7a4fbf", "#2f6fd0", "#2e9e4f", "#d4453c", "#c1548a")


class Channel:
    """One number in a sample: what it is called and what it is measured in."""

    __slots__ = ("key", "name", "unit", "colour", "group")

    def __init__(self, key: str, name: str, unit: str, colour: str, group: str = ""):
        self.key = key            # on the wire, e.g. "ax_g"
        self.name = name          # on screen, e.g. "ax"
        self.unit = unit          # on screen, e.g. "g"
        self.colour = colour
        self.group = group        # channels drawn on the same plot share a group

    def __repr__(self):
        return "Channel(%s %s)" % (self.name, self.unit)


class Modality:
    """A kind of sensor: its usual channels, rate, and how it is described."""

    def __init__(self, key, name, long_name, colour, hz, channels, blurb):
        self.key = key
        self.name = name                    # the badge: IMU, EMG, PPG
        self.long_name = long_name
        self.colour = colour
        self.hz = float(hz)                 # the rate its boards usually run at
        self.channels = list(channels)      # the default set, e.g. for a simulator
        self.blurb = blurb                  # one sentence for the source dialog

    def __repr__(self):
        return "Modality(%s)" % self.key


def _imu_channels():
    out = []
    for grp, unit, names in (("accelerometer", "g", ("ax", "ay", "az")),
                             ("gyroscope", "dps", ("gx", "gy", "gz")),
                             ("magnetometer", "ut", ("mx", "my", "mz"))):
        for i, nm in enumerate(names):
            out.append(Channel("%s_%s" % (nm, unit), nm, UNITS[unit], AXIS[i], grp))
    return out


IMU = Modality(
    "imu", "IMU", "Inertial measurement unit", "#4e8f57", 100.0, _imu_channels(),
    "Accelerometer, gyroscope and magnetometer: nine numbers a hundred times a "
    "second. The board handed out in Week 2.")

EMG = Modality(
    "emg", "EMG", "Surface electromyography", "#e08a1e", 1000.0,
    [Channel("emg1_mv", "emg1", "mV", SERIES[0], "muscle")],
    "The electrical activity of a muscle, read through electrodes on the skin. "
    "Tens of microvolts to a few millivolts, most of it between 20 and 450 Hz.")

PPG = Modality(
    "ppg", "PPG", "Photoplethysmography", "#d4453c", 100.0,
    [Channel("ir_raw", "ir", "counts", "#8b2e2e", "light"),
     Channel("red_raw", "red", "counts", "#d4453c", "light")],
    "Light shone into a fingertip and read back. Each heartbeat changes how "
    "much is absorbed; two wavelengths together give oxygen saturation.")

GENERIC = Modality(
    "generic", "SIGNAL", "Unnamed signal", "#5b6169", 100.0, [],
    "Channels this program has no special pages for. The signal pages still "
    "work on them.")

MODALITIES = {m.key: m for m in (IMU, EMG, PPG, GENERIC)}


# ---------------------------------------------------------------
# reading a header
# ---------------------------------------------------------------
def channel_from_key(key: str, index: int = 0) -> Channel:
    """`ax_g` -> Channel(ax, g). An unknown suffix is kept as it is."""
    key = key.strip()
    name, unit = (key.rsplit("_", 1) + [""])[:2] if "_" in key else (key, "")
    shown = UNITS.get(unit.lower(), unit)
    low = name.lower()
    if low in ("ax", "ay", "az"):
        return Channel(key, name, shown, AXIS["xyz".index(low[1])], "accelerometer")
    if low in ("gx", "gy", "gz"):
        return Channel(key, name, shown, AXIS["xyz".index(low[1])], "gyroscope")
    if low in ("mx", "my", "mz"):
        return Channel(key, name, shown, AXIS["xyz".index(low[1])], "magnetometer")
    if low.startswith("emg"):
        return Channel(key, name, shown, SERIES[index % len(SERIES)], "muscle")
    if low in ("ir", "red", "green", "ppg") or low.startswith("ppg"):
        colour = {"ir": "#8b2e2e", "red": "#d4453c", "green": "#2e9e4f"}.get(low,
                                                                            SERIES[index % 6])
        return Channel(key, name, shown, colour, "light")
    return Channel(key, name, shown, SERIES[index % len(SERIES)], "signal")


def modality_of(channels) -> str:
    """Which modality a set of channels adds up to."""
    groups = {c.group for c in channels}
    if {"accelerometer", "gyroscope"} <= groups:
        return "imu"
    if "muscle" in groups:
        return "emg"
    if "light" in groups:
        return "ppg"
    return "generic"


def parse_header(text: str):
    """
    Column names out of a header, or None if the line is not one.

    Accepts both forms the boards and the recorders use:

        #COLUMNS n,micros,ax_g,...        what a sketch prints
        n,micros,ax_g,...                 the first row of a CSV recording

    Returns the channel keys, without the leading `n` and `micros`.
    """
    text = text.strip()
    if text.upper().startswith("#COLUMNS"):
        text = text[8:].strip(" :\t")
    elif text.startswith("#"):
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 3 or parts[0].lower() != "n" or parts[1].lower() != "micros":
        return None
    return parts[2:]


def channels_from_keys(keys) -> list:
    return [channel_from_key(k, i) for i, k in enumerate(keys)]


def column_names(channels) -> list:
    """The header row a recording of these channels is written with."""
    return ["n", "micros"] + [c.key for c in channels]
