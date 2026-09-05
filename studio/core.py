# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
core.py
The buffer every page reads from, and the estimators the motion pages compare.

One object holds the session: samples come in from whichever source is open,
whatever is derived from them (attitude for an IMU, an envelope for EMG, beats
for PPG) is worked out once as they arrive, and the result is stored beside the
raw numbers so that no page has to redo it. Pages read; they do not accumulate.

    Session       the generic one: a clock column and the raw channels
    ImuSession    adds attitude, earth-frame acceleration, and stillness
    (bio.py)      EmgSession and PpgSession, built the same way

Attitude is estimated by Fusion, x-io Technologies' open-source AHRS (the
algorithm that grew out of the Madgwick filter), through its `imufusion`
bindings, or by the numpy implementation of the same algorithm below when the
library is not installed. Three things in it matter for the course, and each
one is a lesson the pages point at:

    gyroscope bias is learnt while the sensor is still, and subtracted. This is
    what stops attitude walking away at a degree a second.

    the accelerometer corrects the vertical continuously, so roll and pitch do
    not drift at all. Nothing observes heading without a magnetometer, so yaw
    still does.

    when the accelerometer is reading something other than gravity, because the
    board is being moved, it is ignored for as long as that lasts, and the
    gyroscope carries the attitude alone. This is why a shaken board does not
    have its vertical dragged sideways.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import modality

try:
    import imufusion
except ImportError:                                   # noqa: BLE001
    imufusion = None

# Use the compiled reference implementation when it is installed, and the
# numpy one written out below when it is not. Set this to False to read the
# algorithm instead of trusting it; the checks run both and compare them.
USE_LIBRARY = True

G = 9.80665

# What counts as stationary.
#
# The test is on how much the accelerometer vector moves over a short window,
# not on how far it is from gravity. Two things go wrong with the alternatives.
# The length of the reading cannot see a push along the desk: a tenth of a g
# sideways changes the length of a one g vector by half a per cent. And the
# distance from the estimated gravity depends on the attitude estimate being
# right, which it is not until the gyroscope offset has been learnt: a board
# with an unlearnt 0.8 deg/s offset settles two degrees off vertical, which is
# 35 mg of apparent horizontal acceleration on a board that is not moving.
#
# Variation over a window is immune to both. What it cannot see is motion at
# constant velocity, and neither can anything else: an accelerometer measures
# force, and there is none.
STILL_A_G = 0.012                # how far the reading may move over the window
STILL_W_DPS = 2.0                # and how fast it may still be turning
STILL_WINDOW = 0.2               # seconds the variation is measured over
STILL_HOLD = 0.25                # and how long both must hold


# ---------------------------------------------------------------
# attitude, written out
# ---------------------------------------------------------------
class Ahrs:
    """
    The same algorithm as Fusion, in numpy, so that the program runs with or
    without the compiled library and so that the thing being taught can be read.

    Three mechanisms, and each is one line to point at:

        integrate      the gyroscope moves the attitude. On its own it walks
                       away, because a rate error of b degrees a second is an
                       attitude error of b times t.

        correct        gravity says which way is down, absolutely and for ever.
                       Nudging the estimate towards it every sample is what
                       stops roll and pitch drifting at all. Nothing here
                       observes heading, so heading still drifts.

        reject         while the case is being accelerated the accelerometer is
                       not measuring gravity, so correcting towards it would
                       drag the vertical sideways. Past `reject_deg` of
                       disagreement the correction is switched off and the
                       gyroscope carries the attitude alone.

    The gyroscope's own offset is learnt while the sensor is still and taken
    off every sample after, which is the difference between an attitude that
    holds for a minute and one that holds for a second.
    """

    def __init__(self, hz=100.0, gain=0.5, reject_deg=10.0, reject_timeout=5.0,
                 settle=3.0):
        self.hz = float(hz)
        self.gain = float(gain)
        self.reject = math.radians(reject_deg)
        self.reject_timeout = float(reject_timeout)
        self.settle = float(settle)
        self.reset()

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)              # deg/s
        self.t = 0.0
        self.rejecting = 0.0                 # how long the accelerometer has been ignored
        self.ignored = False
        self._quiet = 0.0

    # ---- what the pages read ----
    @property
    def startup(self) -> bool:
        return self.t < self.settle

    def matrix(self) -> np.ndarray:
        return quat_to_matrix(self.q)

    def earth_acceleration(self, a_g) -> np.ndarray:
        """The reading rotated into the world and with gravity taken off, in g."""
        return self.matrix() @ np.asarray(a_g, float) - np.array([0.0, 0.0, 1.0])

    def set_bias(self, deg_s):
        self.bias = np.asarray(deg_s, float).copy()

    def restart(self):
        """Keep the learnt offset, but converge again at the startup gain."""
        self.t = 0.0
        self.rejecting = 0.0
        self.ignored = False

    # ---- one sample ----
    def update(self, w_dps, a_g, dt):
        w = (np.asarray(w_dps, float) - self.bias) * (math.pi / 180.0)
        a = np.asarray(a_g, float)
        self.t += dt

        # 1. integrate the rate
        self.q = _q_normalise(self.q + 0.5 * _q_mul(self.q, np.array([0.0, *w])) * dt)

        # 2. correct towards gravity, unless the accelerometer is busy
        n = float(np.linalg.norm(a))
        if n > 1e-6:
            up_measured = a / n
            # where the estimate says up is, expressed in the sensor's own axes
            up_estimated = self.matrix().T @ np.array([0.0, 0.0, 1.0])
            err = np.cross(up_measured, up_estimated)
            angle = math.asin(min(1.0, float(np.linalg.norm(err))))
            self.ignored = angle > self.reject
            if self.ignored:
                self.rejecting += dt
                if self.rejecting > self.reject_timeout:
                    # The disagreement has lasted longer than any real movement
                    # would. Believe the accelerometer again: the fault is more
                    # likely the gyroscope's.
                    self.ignored = False
                    self.rejecting = 0.0
            else:
                self.rejecting = 0.0
            if not self.ignored:
                # A high gain for the first few seconds, so that the estimate
                # reaches the true attitude instead of creeping towards it from
                # whatever it was initialized to.
                k = 10.0 if self.startup else self.gain
                # Added, not subtracted. The error is the rotation that carries
                # the estimated vertical onto the measured one, so it goes into
                # the integration the same way a rate does.
                self.q = _q_normalise(
                    self.q + 0.5 * _q_mul(self.q, np.array([0.0, *err])) * k * dt)

        # 3. learn the gyroscope's offset, but only after the sensor has been
        #    still for a while. One second is not enough: a slow turn passes
        #    through zero rate twice a cycle, and a filter that learns there
        #    mistakes the turn itself for an offset.
        turning = float(np.linalg.norm(np.asarray(w_dps, float) - self.bias))
        self._quiet = self._quiet + dt if turning < 3.0 else 0.0
        if self._quiet > 3.0:
            self.bias += (np.asarray(w_dps, float) - self.bias) * min(1.0, dt / 10.0)
        return self.q


def _q_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _q_normalise(q) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------
# storage
# ---------------------------------------------------------------
class Ring:
    """
    The last `cap` samples, readable as a contiguous array with no copying.

    Every field is stored twice, `cap` apart. A write goes to both halves, so
    the most recent k samples are always one slice, whatever the wrap. The
    memory is the price of never allocating on a draw.
    """

    def __init__(self, cap: int, width: int):
        self.cap = int(cap)
        self._buf = np.zeros((2 * self.cap, width))
        self.n = 0

    def push(self, row) -> None:
        i = self.n % self.cap
        self._buf[i] = row
        self._buf[i + self.cap] = row
        self.n += 1

    def push_block(self, rows) -> None:
        """Many rows at once, in order. The same as push() for each, but fast."""
        rows = np.asarray(rows)
        k = len(rows)
        if k == 0:
            return
        if k >= self.cap:                      # more than fits: keep the last cap
            rows = rows[-self.cap:]
            k = self.cap
        start = self.n % self.cap
        first = min(k, self.cap - start)
        self._buf[start:start + first] = rows[:first]
        self._buf[self.cap + start:self.cap + start + first] = rows[:first]
        rest = k - first
        if rest:
            self._buf[:rest] = rows[first:]
            self._buf[self.cap:self.cap + rest] = rows[first:]
        self.n += k

    def poke(self, ago: int, col: int, value: float) -> None:
        """Change one field of a sample already stored, `ago` samples back."""
        if ago < 0 or ago >= min(self.n, self.cap):
            return
        i = (self.n - 1 - ago) % self.cap
        self._buf[i, col] = value
        self._buf[i + self.cap, col] = value

    def last(self, k: int) -> np.ndarray:
        """The most recent k samples, oldest first. A view, so do not write to it."""
        k = min(int(k), self.cap, self.n)
        if k <= 0:
            return self._buf[:0]
        end = (self.n - 1) % self.cap + 1 + self.cap
        return self._buf[end - k:end]

    def clear(self) -> None:
        self.n = 0


T = 0                                    # the clock column, every session


class Session:
    """
    Everything that has arrived from one source, with what follows from it.

    A row is the clock, then the raw channels in the order the source lists
    them, then whatever the subclass derives. `last(seconds)` is the whole of
    the interface a page needs; `menu()` tells a channel picker what there is.
    """

    modality = "generic"

    def __init__(self, channels=(), hz: float = 100.0, seconds: float = None):
        self.channels = list(channels)
        self.hz = float(hz) if hz else 100.0
        self.n_raw = len(self.channels)
        self.width = 1 + self.n_raw + self.extra_columns()
        # Three minutes at a hundred hertz is the history the motion pages want.
        # A kilohertz signal keeps a minute: the pages never look back further,
        # and the buffer is stored twice.
        if seconds is None:
            seconds = 180.0 if self.hz <= 250.0 else 60.0
        self.seconds = float(seconds)
        self.ring = Ring(int(self.seconds * max(self.hz, 50.0)), self.width)
        self._last_t = None
        self.prepare()

    # ---- what a subclass fills in ----
    def extra_columns(self) -> int:
        return 0

    def prepare(self) -> None:
        """Set up whatever derives columns. Called at start and on reset."""

    def derive_block(self, raw, dts):
        """
        The extra columns for a block of raw rows, shape (n, extra), or None.

        `raw` is (n, 1 + n_raw): the clock and the channels. `dts` is the
        interval before each sample, already sanitised.
        """
        return None

    def after_block(self, rows) -> None:
        """A look at the rows just stored, after they are in the ring."""

    # ---- taking samples in ----
    def reset(self):
        self.ring.clear()
        self._last_t = None
        self.prepare()

    @property
    def n(self):
        return self.ring.n

    def add(self, samples) -> int:
        """Take a block from a source. Returns how many were stored."""
        if not samples:
            return 0
        k = len(samples)
        raw = np.zeros((k, 1 + self.n_raw))
        for i, (t, values) in enumerate(samples):
            raw[i, 0] = t
            m = min(self.n_raw, len(values))
            raw[i, 1:1 + m] = values[:m]
        t = raw[:, 0]
        prev = np.concatenate([[t[0] if self._last_t is None else self._last_t], t[:-1]])
        dts = t - prev
        bad = (dts <= 0) | (dts > 1.0)
        dts[bad] = 1.0 / self.hz
        self._last_t = float(t[-1])
        extra = self.derive_block(raw, dts)
        rows = raw if extra is None else np.hstack([raw, extra])
        self.ring.push_block(rows)
        self.after_block(rows)
        return k

    # ---- what pages ask for ----
    def last(self, seconds: float) -> np.ndarray:
        return self.ring.last(int(seconds * max(self.hz, 1.0)) + 2)

    def raw(self, d) -> np.ndarray:
        """The raw channels of a block of rows, (n, n_raw)."""
        return d[:, 1:1 + self.n_raw]

    def menu(self) -> list:
        """
        (label, unit, getter) for every signal a channel picker may offer.

        The getter takes a block of rows and returns one column of it. The
        generic session offers each raw channel as it is.
        """
        out = []
        for i, c in enumerate(self.channels):
            out.append((c.name, c.unit, _column(1 + i)))
        return out


def _column(i):
    return lambda d: d[:, i]


# columns of an IMU session
AX, AY, AZ, GX, GY, GZ, MX, MY, MZ = range(1, 10)
QW, QX, QY, QZ = 10, 11, 12, 13
EX, EY, EZ = 14, 15, 16          # earth-frame acceleration, gravity removed, in g
CX, CY, CZ = 17, 18, 19          # gyroscope after its learnt bias is removed
STILL = 20
WIDTH = 21

A = slice(AX, AZ + 1)
W = slice(GX, GZ + 1)
MAG = slice(MX, MZ + 1)
QUAT = slice(QW, QZ + 1)
EARTH = slice(EX, EZ + 1)
CLEAN = slice(CX, CZ + 1)


class ImuSession(Session):
    """An IMU's nine channels, with attitude already worked out."""

    modality = "imu"

    def __init__(self, channels=None, hz: float = 100.0, seconds: float = 180.0):
        # Always the canonical nine, in this order. A source with an IMU header
        # puts its columns into this order before handing them over.
        super().__init__(modality.IMU.channels if channels is None else channels,
                         hz, seconds)
        assert self.n_raw == 9, "an IMU session takes nine channels"

    def extra_columns(self) -> int:
        return WIDTH - 10

    def prepare(self):
        self._quiet = 0.0
        self._recent = deque(maxlen=max(4, int(STILL_WINDOW * self.hz)))
        # The pose the board was put down in, if the student has said so. It is
        # not a correction to the attitude: gravity is still gravity, and the
        # filter still knows where down is. It is what the readouts are shown
        # against, because a board wedged at thirty degrees by its own cable is
        # not a board anybody wants to read thirty degrees off.
        self.level = None
        self._new_fusion()

    def set_level(self, rot=None) -> None:
        """Take a pose as the reference. With no argument, the current one."""
        if rot is None:
            if self.ring.n == 0:
                return
            rot = quat_to_matrix(self.ring.last(1)[0, QUAT])
        self.level = np.asarray(rot, float).copy()

    def clear_level(self) -> None:
        self.level = None

    def relative(self, q) -> np.ndarray:
        """The attitude as it should be shown: against the level, if there is one."""
        R = quat_to_matrix(q)
        return R if self.level is None else self.level.T @ R

    def _new_fusion(self):
        if imufusion is None or not USE_LIBRARY:
            self.ahrs = Ahrs(self.hz)
            self.bias = None
            return
        s = imufusion.AhrsSettings()
        s.convention = imufusion.CONVENTION_NWU
        s.gain = 0.5
        s.gyroscope_range = 2000.0
        # Ignore the accelerometer while it reads more than 10 degrees away
        # from where the gyroscope says down is, and give up on ignoring it
        # after 5 s in case the disagreement is the gyroscope's fault.
        s.acceleration_rejection = 10.0
        s.magnetic_rejection = 10.0
        s.rejection_timeout = int(5 * self.hz)
        s.sample_rate = self.hz
        self.ahrs = imufusion.Ahrs()
        self.ahrs.set_settings(s)
        self.ahrs.set_sample_period(1.0 / self.hz)

        b = imufusion.BiasSettings()
        b.sample_rate = self.hz
        b.stationary_threshold = 3.0        # deg/s below which it counts as still
        b.stationary_period = 3.0           # and for how long before the bias is taken
        self.bias = imufusion.Bias()
        self.bias.set_settings(b)

    def derive_block(self, raw, dts):
        out = np.zeros((len(raw), WIDTH - 10))
        for i in range(len(raw)):
            a = raw[i, A]
            w = raw[i, W]
            dt = float(dts[i])
            if self.bias is not None:                 # the compiled library
                wc = self.bias.update(w.astype(np.float32))
                self.ahrs.set_sample_period(dt)
                self.ahrs.update_no_magnetometer(wc, a.astype(np.float32))
                out[i, 0:4] = self.ahrs.get_quaternion()
                out[i, 4:7] = self.ahrs.get_earth_acceleration()
            else:                                     # the same thing, in numpy
                out[i, 0:4] = self.ahrs.update(w, a, dt)
                out[i, 4:7] = self.ahrs.earth_acceleration(a)
                wc = w - self.ahrs.bias
            out[i, 7:10] = wc

            self._recent.append(a.copy())
            spread = 0.0
            if len(self._recent) >= 4:
                block = np.asarray(self._recent)
                spread = float(np.abs(block - block.mean(axis=0)).max())
            quiet = (len(self._recent) >= 4 and spread < STILL_A_G
                     and float(np.linalg.norm(w)) < STILL_W_DPS)
            self._quiet = self._quiet + dt if quiet else 0.0
            out[i, 10] = 1.0 if self._quiet >= STILL_HOLD else 0.0
        return out

    # ---- what pages ask for ----
    def menu(self) -> list:
        g = G
        return [("accel x", "m/s²", lambda d: d[:, AX] * g),
                ("accel y", "m/s²", lambda d: d[:, AY] * g),
                ("accel z", "m/s²", lambda d: d[:, AZ] * g),
                ("|accel|", "m/s²", lambda d: np.linalg.norm(d[:, A], axis=1) * g),
                ("gyro x", "°/s", _column(GX)),
                ("gyro y", "°/s", _column(GY)),
                ("gyro z", "°/s", _column(GZ))]

    def gyro_bias(self) -> np.ndarray:
        """What the filter has decided the gyroscope's own offset is, in deg/s."""
        if self.bias is None:
            return self.ahrs.bias.copy()
        return np.asarray(self.bias.get_offset(), float)

    def set_gyro_bias(self, deg_s) -> None:
        """
        Hand the filter the offset measured over a stationary interval.

        Left to itself the filter finds this in its own time, and until it has,
        the leftover rate tips the estimated vertical and a share of gravity
        appears as horizontal acceleration. Giving it the answer at the moment
        the student aligns the board is what a real system does at power-on,
        and it is the difference between an alignment that is usable in five
        seconds and one that is usable in forty.
        """
        if self.bias is None:
            self.ahrs.set_bias(deg_s)
        else:
            self.bias.set_offset(np.asarray(deg_s, np.float32))

    def set_attitude(self, rot) -> None:
        """
        Put the attitude where the measurement says it is, at once.

        Waiting for the filter to converge to a vertical we have already
        measured costs three seconds and buys nothing. Gravity gives the
        attitude directly, so it is set directly.
        """
        q = matrix_to_quat(rot)
        if self.bias is None:
            self.ahrs.q = q
            self.ahrs.t = self.ahrs.settle          # done starting up
        else:
            self.ahrs.set_quaternion(q.astype(np.float32))
            self.ahrs.skip_startup()

    def settled(self) -> bool:
        """Has the attitude filter finished starting up?"""
        if self.bias is None:
            return not self.ahrs.startup
        f = self.ahrs.get_flags()
        return not f.startup

    def rejecting(self) -> bool:
        """Is the accelerometer being ignored because the board is moving?"""
        if self.bias is None:
            return self.ahrs.ignored
        return bool(self.ahrs.get_flags().acceleration_recovery)


def session_for(source) -> Session:
    """The right kind of session for whatever has just been opened."""
    hz = max(float(getattr(source, "hz", 0.0) or 0.0), 20.0)
    kind = getattr(source, "modality", "generic")
    channels = list(getattr(source, "channels", []))
    if kind == "imu":
        return ImuSession(hz=hz)
    if kind == "emg":
        from . import bio
        return bio.EmgSession(channels, hz)
    if kind == "ppg":
        from . import bio
        return bio.PpgSession(channels, hz)
    return Session(channels, hz)


# ---------------------------------------------------------------
# geometry
# ---------------------------------------------------------------
def quat_to_matrix(q) -> np.ndarray:
    """Body axes to world axes, as a 3x3."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R) -> np.ndarray:
    """A rotation matrix as w, x, y, z."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return _q_normalise(np.array(q))


def euler_deg(q) -> np.ndarray:
    """Roll, pitch, yaw in degrees."""
    if imufusion is not None:
        return np.asarray(imufusion.quaternion_to_euler(np.asarray(q, np.float32)), float)
    R = quat_to_matrix(q)
    return np.degrees([math.atan2(R[2, 1], R[2, 2]),
                       -math.asin(max(-1.0, min(1.0, R[2, 0]))),
                       math.atan2(R[1, 0], R[0, 0])])


def attitude_from_gravity(a_g) -> np.ndarray:
    """
    Which way the board is lying, from the accelerometer alone.

    At rest it reads one g along whichever of its own axes points up. Rotating
    that onto the world's up gives the attitude, all but the turn about the
    vertical: gravity says nothing about heading, so that is left alone rather
    than invented.
    """
    a = np.asarray(a_g, float)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.eye(3)
    up = a / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s, c = float(np.linalg.norm(v)), float(up @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


_CORNER = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.5
_EDGE = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
# The board is a rectangle, not a cube: 45 x 18 x 5 mm, drawn to scale so that
# which way up it is lying can be read off the picture.
_SHAPE = np.array([1.0, 0.40, 0.12])


def board_edges(rot, size=1.0) -> np.ndarray:
    """The board's twelve edges as one path with gaps, ready for a single line."""
    pts = _CORNER * _SHAPE * size
    out = []
    for i, j in _EDGE:
        out.append(rot @ pts[i])
        out.append(rot @ pts[j])
        out.append([np.nan, np.nan, np.nan])
    return np.array(out)


def project(points, elev=22.0, azim=-58.0) -> np.ndarray:
    """
    Orthographic projection of world points onto the screen.

    Orthographic on purpose: a perspective view foreshortens the far side of
    the volume, and a trajectory drawn in one cannot be measured by eye.
    """
    e, a = math.radians(elev), math.radians(azim)
    ca, sa, ce, se = math.cos(a), math.sin(a), math.cos(e), math.sin(e)
    right = np.array([ca, sa, 0.0])
    up = np.array([-sa * se, ca * se, ce])
    p = np.asarray(points, float).reshape(-1, 3)
    return np.column_stack([p @ right, p @ up])


# ---------------------------------------------------------------
# the estimators the position page compares
# ---------------------------------------------------------------
class Estimator:
    """
    Specific force in, position out.

    Subclasses differ only in what they do between the two integrations. That
    is the whole point of the page they appear on: the measurements are
    identical, so any difference in the answer belongs to the processing.
    """

    key = "?"
    name = "?"
    colour = "#000000"
    subtitle = ""
    equation = ""            # rich text, with the term that causes the trouble in red
    terms = ()               # (symbol, what it is) shown under the equation
    why = ""
    source = None            # (title, url) for the panel to link to
    sources = ()             # more of them

    def __init__(self, leak=None, zupt=False, use_fusion=True):
        self.leak = leak
        self.zupt = zupt
        self.use_fusion = use_fusion
        self.reset()

    def reset(self):
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.t0 = None
        self.held = 0
        self.n = 0
        self.a0 = None                 # the resting reading, when it is being subtracted
        self.e0 = np.zeros(3)          # and the same for the route through Fusion
        self.rot0 = np.eye(3)
        self.track_t, self.track_p = [], []

    def start(self, t, a0, rot0, e0=None):
        """
        Begin from rest, with whatever was measured while the board was still.

        Both routes get a zero, and they have to. Subtracting the resting
        reading takes off gravity and the accelerometer's own offset together,
        because at rest they arrive as one number. Giving that to the naive
        route and not to the one through Fusion would have made attitude
        tracking look worse than no attitude tracking, for a reason that has
        nothing to do with attitude: it would have been the only one still
        carrying a six milligravity bias.
        """
        self.reset()
        self.t0 = t
        self.a0 = np.asarray(a0, float).copy()
        self.e0 = np.zeros(3) if e0 is None else np.asarray(e0, float).copy()
        self.rot0 = np.asarray(rot0, float).copy()

    def _accel(self, row) -> np.ndarray:
        """World-frame acceleration in m/s^2, by whichever route this estimator takes."""
        if self.use_fusion:
            # Fusion has already taken gravity off, using an attitude that is
            # kept up to date, so nothing here assumes the board has not moved.
            # What it leaves behind is the accelerometer's own offset, measured
            # at the alignment the same way the other route measures it.
            return (np.asarray(row[EARTH], float) - self.e0) * G
        # The naive route: one attitude, measured once, assumed to hold for ever.
        return (self.rot0 @ np.asarray(row[A], float) - self.rot0 @ self.a0) * G

    def step(self, row, dt) -> None:
        a = self._accel(row)
        leak = 1.0 if self.leak is None else math.exp(-dt / self.leak)
        if self.zupt and row[STILL] > 0.5:
            self.v = np.zeros(3)
            self.held += 1
        else:
            self.v = leak * self.v + a * dt
        # Position is only leaked when nothing else is holding the velocity
        # down. Once the velocity is being reset at every rest the position is
        # already bounded, and leaking it as well would throw away real distance.
        self.p = (1.0 if self.zupt else leak) * self.p + self.v * dt
        self.n += 1
        if self.n % 5 == 0:
            self.track_t.append(row[T] - self.t0)
            self.track_p.append(self.p.copy())
            if len(self.track_t) > 12000:
                del self.track_t[:6000]
                del self.track_p[:6000]

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.p))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))

    def track(self):
        if not self.track_t:
            return np.zeros(0), np.zeros((0, 3))
        return np.asarray(self.track_t), np.asarray(self.track_p)


R = "#d4453c"      # the guilty term


WIKI_INS = "https://en.wikipedia.org/wiki/Inertial_navigation_system"
WIKI_AHRS = "https://en.wikipedia.org/wiki/Attitude_and_heading_reference_system"
WIKI_DR = "https://en.wikipedia.org/wiki/Dead_reckoning"
WIKI_LEAKY = "https://en.wikipedia.org/wiki/Leaky_integrator"
ZUPT_PAPER = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10857087/"
WOODMAN = "https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf"


class DoubleIntegration(Estimator):
    key = "double"
    name = "Double integration"
    colour = "#d4453c"
    subtitle = "uncorrected"
    equation = (
        "<span style='font-size:16pt'>"
        "p(t) &nbsp;=&nbsp; &#8748; ( a(&tau;) "
        "<span style='color:%s'>&nbsp;+&nbsp;b</span> "
        "<span style='color:%s'>&nbsp;+&nbsp;g&#183;&#948;&#952;</span>"
        " ) d&tau;<sup>2</sup>"
        "&nbsp;&nbsp;&#8658;&nbsp;&nbsp; error = "
        "<span style='color:%s'>b t<sup>2</sup>/2</span></span>" % (R, R, R))
    terms = (("<span style='color:%s'>b</span>" % R, "accelerometer bias"),
             ("<span style='color:%s'>g&#183;&#948;&#952;</span>" % R,
              "gravity leaking in once the attitude has changed"))
    why = ("An accelerometer with a 10 &#181;g bias reaches <b>50 m of error in 17 "
           "minutes</b>. A bias does not average away; its position error grows as t&#178;.")
    source = ("Inertial navigation system, Wikipedia", WIKI_INS)
    sources = (("Woodman (2007), An introduction to inertial navigation, §3", WOODMAN),)

    def __init__(self):
        super().__init__(use_fusion=False)


class FusedIntegration(Estimator):
    key = "fused"
    name = "Attitude tracking"
    colour = "#e08a1e"
    subtitle = "gravity removed using a tracked attitude"
    equation = (
        "<span style='font-size:16pt'>"
        "p(t) &nbsp;=&nbsp; &#8748; ( R(t) a<sub>body</sub>(&tau;) &minus; g "
        "<span style='color:%s'>&nbsp;+&nbsp;b</span> ) d&tau;<sup>2</sup>"
        "</span>" % R)
    terms = (("R(t)", "attitude, corrected against gravity every sample"),
             ("<span style='color:%s'>b</span>" % R, "accelerometer bias, still present"))
    why = ("Tracking the attitude removes the gravity term. The accelerometer's own "
           "bias <b>b</b> stays, so the error still grows as t&#178;.")
    source = ("Attitude and heading reference system, Wikipedia", WIKI_AHRS)
    sources = (("x-io Fusion, the open-source AHRS this program uses",
                "https://github.com/xioTechnologies/Fusion"),)


class LeakyIntegration(Estimator):
    key = "leaky"
    name = "Leaky integration"
    colour = "#7a4fbf"
    subtitle = "both integrators leaked over &tau;"
    equation = (
        "<span style='font-size:16pt'>"
        "v&#775; &nbsp;=&nbsp; a &minus; <span style='color:%s'>v/&#964;</span>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&#8658;&nbsp;&nbsp;&nbsp;&nbsp; "
        "p<sub>&#8734;</sub> &nbsp;=&nbsp; b&#183;&#964;<sup>2</sup>"
        "</span>" % R)
    terms = (("&#964;", "leak time constant"),
             ("<span style='color:%s'>v/&#964;</span>" % R,
              "discarded every second, real motion included"))
    why = ("A high-pass on the estimate: the error settles at <b>b&#964;&#178;</b> "
           "instead of growing. Everything slower than &#964; goes with it, real "
           "displacement included.")
    source = ("Leaky integrator, Wikipedia", WIKI_LEAKY)
    sources = (("Dead reckoning, Wikipedia", WIKI_DR),)

    def __init__(self, tau=1.0):
        super().__init__(leak=tau)


class ZuptIntegration(Estimator):
    key = "zupt"
    name = "Zero-velocity update"
    colour = "#2f6fd0"
    subtitle = "velocity forced to zero at every rest"
    equation = (
        "<span style='font-size:16pt'>"
        "v(t) &nbsp;&#8592;&nbsp; 0 &nbsp;&nbsp;whenever&nbsp;&nbsp; "
        "| |a| &minus; g | &lt; &#949;<sub>a</sub> &nbsp;and&nbsp; |&#969;| &lt; &#949;<sub>&#969;</sub>"
        "</span>")
    terms = (("&#949;<sub>a</sub>, &#949;<sub>&#969;</sub>", "20 mg and 2 deg/s here"),
             ("<span style='color:%s'>gap between rests</span>" % R,
              "the only interval the error can grow over"))
    why = ("Every rest is a moment when the velocity is known to be zero. Resetting it "
           "there turns error growth in time into error growth per step, which is how "
           "foot-mounted navigation works.")
    source = ("Zero-velocity interval detection, Sensors 2024", ZUPT_PAPER)
    sources = (("Pedestrian dead reckoning, Wikipedia",
                "https://en.wikipedia.org/wiki/Dead_reckoning#Pedestrian_dead_reckoning"),)

    def __init__(self):
        super().__init__(zupt=True)


def make_estimators(tau=1.0):
    return [DoubleIntegration(), FusedIntegration(), LeakyIntegration(tau), ZuptIntegration()]
