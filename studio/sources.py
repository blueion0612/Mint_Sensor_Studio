# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
sources.py
Where the measurements come from. Five ways, one interface.

    SerialSource      a board on a USB cable, any modality it announces
    BleSource         the IMU board over Bluetooth Low Energy, no cable
    SimulatedSource   a virtual IMU
    SimulatedEmg      a virtual muscle
    SimulatedPpg      a virtual fingertip
    FileSource        a recording, played back

The simulated ones are not stand-ins for missing hardware. They are the only
sources whose truth is known, so they are the only ones that can answer "how
wrong is the estimate": every page that shows an error against truth is showing
it because the simulation was asked what really happened.

Every source hands samples to a thread-safe queue as (t, values), where values
is one number per channel in the order `channels` lists them, and answers the
same questions the toolbar asks: what am I, what am I talking to, how fast are
samples arriving, and how many never did. `Recorder` at the bottom writes what
any of them produce to a CSV that FileSource can play back.
"""

from __future__ import annotations

import asyncio
import csv
import math
import os
import struct
import threading
import time
from collections import deque

import numpy as np

from . import dsp, modality as mod

G = 9.80665                      # m/s^2 in one g

BAUD = 115200
US_WRAP = 2 ** 32                # micros() on the board wraps here
ARDUINO_VIDS = (0x2341, 0x2A03, 0x1B4F, 0x239A)

# The wire format the IMU board speaks over the radio. These must match the sketch.
BLE_NAME = "IMU-"
BLE_SERVICE = "494d554c-4142-4000-8000-000000000000"
BLE_IMU = "494d554c-4142-4000-8000-000000000001"
BLE_MAG = "494d554c-4142-4000-8000-000000000002"
BLE_INFO = "494d554c-4142-4000-8000-000000000003"
BLE_A_SCALE = 4096.0             # counts per g
BLE_G_SCALE = 16.0               # counts per degree a second
BLE_M_SCALE = 16.0               # counts per microtesla
BLE_RECORD = 18                  # bytes in one sample
BLE_N_WRAP = 2 ** 16             # the sample number is sent in two bytes
BLE_SCAN = 5.0                   # seconds a scan listens for


class Source:
    """What every source has to be able to answer."""

    kind = "?"                   # shown in the toolbar
    truth = False                # can it say what really happened
    modality = "generic"
    channels = ()

    def __init__(self, name: str = "", keep: int = 60000):
        self.name = name
        self.queue: deque = deque(maxlen=keep)
        self.info = ""
        self.error = None        # a failure in the reading thread, for the main one
        self.count = 0           # samples that arrived
        self.dropped = 0         # samples the board sent that never did
        self.bytes = 0
        self.hz = 0.0
        self.bps = 0.0
        self._stop = threading.Event()
        self._t = 0.0
        self._last_us = None
        self._last_n = None
        self._mark = (time.time(), 0, 0)

    # ---- what the toolbar shows ----
    def where(self) -> str:
        return "%s  %s" % (self.kind, self.name) if self.name else self.kind

    def rate_text(self) -> str:
        lost = "no loss" if not self.dropped else "%d lost" % self.dropped
        return "%.0f Hz   %.1f kB/s   %s" % (self.hz, self.bps / 1000.0, lost)

    @property
    def badge(self) -> str:
        return mod.MODALITIES.get(self.modality, mod.GENERIC).name

    # ---- the reading side ----
    def _book(self, n, us, nbytes, us_wrap=US_WRAP, n_wrap=None) -> float:
        """
        Book one arriving sample and return its time in seconds.

        The board's clock is in microseconds and wraps; so does its sample
        number over the radio, where it is sent in two bytes. Both are unwound
        here so that a session running past a wrap does not see time jump
        backwards or the lost count leap by sixty thousand.
        """
        if self._last_us is not None:
            step = us - self._last_us
            if step < 0:
                step += us_wrap
            self._t += step * 1e-6
        self._last_us = us

        if self._last_n is not None:
            gap = n - self._last_n
            if n_wrap is not None and gap < 0:
                gap += n_wrap
            # The sketch restarts its count when a listener attaches, so a step
            # backwards is a fresh start, not sixty thousand losses.
            if 1 < gap < 1000:
                self.dropped += int(gap) - 1
        self._last_n = n

        self.count += 1
        self.bytes += nbytes
        now = time.time()
        t0, n0, b0 = self._mark
        span = now - t0
        if span >= 0.4:
            self.hz = (self.count - n0) / span
            self.bps = (self.bytes - b0) / span
            self._mark = (now, self.count, self.bytes)
        return self._t

    def read(self) -> list:
        """Every sample since the last call. Never blocks."""
        out = []
        q = self.queue
        while q:
            out.append(q.popleft())
        return out

    def wait(self, seconds: float = 5.0) -> None:
        """Block until samples arrive, or raise with something worth reading."""
        end = time.time() + seconds
        while time.time() < end:
            if self.error is not None:
                raise RuntimeError(str(self.error))
            if self.count:
                return
            time.sleep(0.03)
        self.close()
        raise RuntimeError(self._silent_message())

    def _silent_message(self) -> str:
        return "the source opened but sent nothing."

    def close(self):
        self._stop.set()


# ---------------------------------------------------------------
# the cable
# ---------------------------------------------------------------
def list_serial_ports() -> list:
    """(device, description, is_an_arduino) for every port, boards first."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    ports = list(serial.tools.list_ports.comports())
    out = [(p.device, p.description or "", (p.vid or 0) in ARDUINO_VIDS) for p in ports]
    out.sort(key=lambda r: (not r[2], r[0]))
    return out


_IMU_ORDER = [c.key for c in mod.IMU.channels]


class SerialSource(Source):
    """
    Whatever board is on the cable, in whatever it says it sends.

    The sketch prints a header, `#COLUMNS n,micros,ax_g,...`, and that is the
    whole of the configuration: the names say what each number is and the
    modality follows from them. A board that has not printed its header yet is
    asked for it, and until it answers the lines are counted but not believed.
    """

    kind = "USB"

    def __init__(self, port: str = "auto", keep: int = 60000, expect: str = "auto"):
        import serial
        if port == "auto":
            boards = [p for p in list_serial_ports() if p[2]]
            if not boards:
                raise RuntimeError(
                    "no Arduino board is on any serial port.\n"
                    "Check the cable first: a charge-only cable carries power but no data.")
            port = boards[0][0]
        super().__init__(port, keep)
        self.expect = expect                 # what the dialog said, for a headerless board
        self.modality = "generic"
        self.channels = []
        self._order = None                   # column index for each channel
        self._ncol = None                    # how many fields a sample line has
        self._unheaded = 0                   # lines seen with no header yet
        try:
            self.ser = serial.Serial(port, BAUD, timeout=0.2)
        except serial.SerialException as e:
            text = str(e).lower()
            # The last one is Korean ("denied"): a Windows in Korean says the
            # access was denied in Korean, and the English words are then
            # absent. Written as escapes so that the source stays ASCII.
            if any(w in text for w in ("permission", "access", "denied")) \
                    or "\uac70\ubd80" in str(e):
                raise RuntimeError("%s is open in another program.\n"
                                   "Close the Arduino IDE's Serial Monitor and try again." % port)
            raise RuntimeError("%s could not be opened.\n%s" % (port, e))
        self.ser.reset_input_buffer()
        try:
            self.ser.write(b"?")                # ask the sketch to introduce itself
        except Exception:
            pass
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except Exception:
                return
            if not chunk:
                continue
            buf += chunk
            # whole lines only; whatever is after the last newline waits
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                self._line(raw.strip(b"\r"), len(raw) + 1)

    def _line(self, raw: bytes, nbytes: int):
        text = raw.decode("ascii", errors="replace").strip()
        if not text:
            return
        if text[0] == "#":
            keys = mod.parse_header(text)
            if keys:
                self._take_header(keys)
            elif "_stream" in text:
                self.info = text.lstrip("#").strip()
            return
        parts = text.split(",")
        if self._order is None:
            # no header yet: a board that was already streaming when the port
            # opened. Ask again, and after enough lines settle for what the
            # dialog said, or for unnamed channels.
            self._unheaded += 1
            if self._unheaded in (5, 50):
                try:
                    self.ser.write(b"?")
                except Exception:
                    pass
            if self._unheaded < 100:
                return
            self._take_header(self._guess(len(parts) - 2))
        if len(parts) != self._ncol:
            return
        try:
            row = [float(p) for p in parts]
        except ValueError:
            return
        t = self._book(row[0], row[1], nbytes)
        self.queue.append((t, [row[i] for i in self._order]))

    def _guess(self, n):
        want = mod.MODALITIES.get(self.expect)
        if want is not None and len(want.channels) == n:
            return [c.key for c in want.channels]
        if n == 9:
            return _IMU_ORDER
        return ["ch%d_raw" % (i + 1) for i in range(n)]

    def _take_header(self, keys):
        chans = mod.channels_from_keys(keys)
        kind = mod.modality_of(chans)
        order = [2 + i for i in range(len(chans))]
        if kind == "imu":
            # the canonical nine, in the canonical order, whatever the board's
            have = {c.key: i for i, c in enumerate(chans)}
            if all(k in have for k in _IMU_ORDER):
                order = [2 + have[k] for k in _IMU_ORDER]
                chans = list(mod.IMU.channels)
            else:
                kind = "generic"
        self.channels = chans
        self.modality = kind
        self._ncol = 2 + len(keys)
        self._order = order

    def _silent_message(self):
        return ("%s opened, but the board sent nothing.\n"
                "1. Is a stream sketch uploaded? Its LED blinks when it is running.\n"
                "2. Is the right board chosen under Tools > Board?\n"
                "3. Press the white RESET button once, then try again." % self.name)

    def close(self):
        super().close()
        try:
            self.ser.close()
        except Exception:
            pass


# ---------------------------------------------------------------
# the radio
# ---------------------------------------------------------------
class BleSource(Source):
    """
    The IMU's nine measurements with no cable.

    A radio packet holds twenty bytes unless both ends agree on more, and one
    line of the text the cable sends is seventy-eight, so the sketch sends the
    numbers rather than their digits: eighteen bytes for a sample, four samples
    to a packet. Multiplying them back out is the whole of the difference.
    """

    kind = "BLE"
    modality = "imu"
    channels = tuple(mod.IMU.channels)

    def __init__(self, want: str = "", keep: int = 60000, scan: float = BLE_SCAN):
        super().__init__(want or "scanning", keep)
        self.want = want
        self.scan = scan
        self._mag = [0.0, 0.0, 0.0]
        threading.Thread(target=self._run, daemon=True).start()

    def _on_imu(self, _handle, data: bytearray):
        raw = bytes(data)
        for i in range(0, len(raw) - BLE_RECORD + 1, BLE_RECORD):
            us, ax, ay, az, gx, gy, gz, n = struct.unpack("<IhhhhhhH", raw[i:i + BLE_RECORD])
            t = self._book(n, us, BLE_RECORD, n_wrap=BLE_N_WRAP)
            self.queue.append((t, [ax / BLE_A_SCALE, ay / BLE_A_SCALE, az / BLE_A_SCALE,
                                   gx / BLE_G_SCALE, gy / BLE_G_SCALE, gz / BLE_G_SCALE]
                               + list(self._mag)))

    def _on_mag(self, _handle, data: bytearray):
        if len(data) >= 6:
            mx, my, mz = struct.unpack("<hhh", bytes(data[:6]))
            self._mag = [mx / BLE_M_SCALE, my / BLE_M_SCALE, mz / BLE_M_SCALE]

    def _run(self):
        try:
            asyncio.run(self._talk())
        except Exception as e:                       # shown to the student as it is
            self.error = e

    async def _talk(self):
        from bleak import BleakClient, BleakScanner

        dev = None
        # A board that has just been disconnected from goes quiet for a couple
        # of seconds before it advertises again, so one empty scan is not
        # evidence that it is absent. Connecting twice in a row is the ordinary
        # case, not the unusual one: it is what happens every time a student
        # closes the window and opens it again.
        for attempt in range(3):
            found = await BleakScanner.discover(timeout=self.scan)
            named = [d for d in found if (d.name or "").upper().startswith(BLE_NAME)]
            if self.want:
                named = [d for d in named if (d.name or "").upper() == self.want.upper()]
            if named:
                dev = named[0]
                break
            if not self._stop.is_set():
                await asyncio.sleep(2.0)
        if dev is None:
            near = sorted({d.name for d in found if d.name})
            raise RuntimeError(
                "no board advertising as %sXXXX was found.\n"
                "1. Is USE_BLE set to 1 in imu_stream.ino, and uploaded?\n"
                "2. Is the board powered? The radio still needs the cable or a battery.\n"
                "3. Is something else already connected? A board talks to one at a time.\n"
                "4. Is Bluetooth switched on?\n"
                "Seen nearby: %s" % (BLE_NAME, ", ".join(near[:8]) or "nothing"))

        self.name = dev.name or "IMU"
        async with BleakClient(dev) as client:
            try:
                raw = await client.read_gatt_char(BLE_INFO)
                self.info = bytes(raw).decode("ascii", errors="replace").lstrip("#").strip()
            except Exception:
                self.info = "imu_stream over BLE"
            await client.start_notify(BLE_IMU, self._on_imu)
            await client.start_notify(BLE_MAG, self._on_mag)
            while not self._stop.is_set() and client.is_connected:
                await asyncio.sleep(0.1)
            try:
                await client.stop_notify(BLE_IMU)
                await client.stop_notify(BLE_MAG)
            except Exception:
                pass

    def _silent_message(self):
        return ("connected over Bluetooth, but no measurements arrived.\n"
                "The sketch on the board is older than this program. Upload imu_stream.ino "
                "again with USE_BLE set to 1.")


async def _scan_names(seconds: float) -> list:
    from bleak import BleakScanner
    found = await BleakScanner.discover(timeout=seconds)
    return sorted({d.name for d in found if (d.name or "").upper().startswith(BLE_NAME)})


def list_ble_devices(seconds: float = BLE_SCAN) -> list:
    """Board names advertising nearby. Blocks for `seconds`, so call it off the UI thread."""
    try:
        import bleak                                              # noqa: F401
    except ImportError:
        return []
    try:
        return asyncio.run(_scan_names(seconds))
    except Exception:
        return []


# ---------------------------------------------------------------
# things that do not exist
# ---------------------------------------------------------------
class Simulated(Source):
    """
    What the three simulators share: a clock that runs in real time when the
    window is open, and a `generate()` that runs it as fast as it can when a
    check wants a whole run at once.
    """

    kind = "SIM"
    truth = True

    def __init__(self, name, hz, keep=60000, realtime=True):
        super().__init__(name, keep)
        # The rate it produces at and the rate it is seen to arrive at are two
        # different numbers, and they must not share a name. They did, and the
        # result was a source that stalled: a busy window slowed the measured
        # rate, the generator read that as its own rate and made fewer samples,
        # which slowed the measured rate further, all the way down to nothing.
        self.rate = float(hz)
        self.hz = float(hz)
        self.dt = 1.0 / self.rate
        self._n = 0
        self._wall = time.perf_counter()
        if realtime:
            threading.Thread(target=self._run, daemon=True).start()

    def generate(self, seconds: float) -> None:
        """Produce that many seconds of samples at once. Only for realtime=False."""
        self._make(int(seconds * self.rate))

    def _make(self, count: int) -> None:
        raise NotImplementedError

    def _run(self):
        # Real time, so that the window behaves the same whether or not a board
        # is plugged in. A page that measures the arrival rate has to see one.
        while not self._stop.is_set():
            now = time.perf_counter()
            due = int((now - self._wall) * self.rate) - self._n
            if due <= 0:
                time.sleep(0.002)
                continue
            self._make(min(due, int(self.rate)))
            time.sleep(0.001)

    def _emit(self, t, values, nbytes=18):
        self._book(self._n, int(t * 1e6) % US_WRAP, nbytes)
        self.queue.append((t, values))


MOTIONS = ("still", "slide and stop", "slow tilt", "turn on the spot",
           "tap", "shake 2 Hz", "shake 8 Hz", "sweep 1-8 Hz")


class SimulatedSource(Simulated):
    """
    A board that does not exist, whose every fault is known because it was put
    there on purpose.

    Real hardware can show that an estimate drifts. It cannot show by how much
    the estimate is wrong, because nothing in the room knows where the board
    truly is. Here the trajectory is written first and the measurements are
    derived from it, so the error is available exactly, and a page can draw the
    truth next to the estimate instead of asking the student to take the drift
    on faith.

    What the knobs do, in the units the datasheet uses:

        bias        a constant added to the gyroscope, in degrees a second.
                    This is what makes attitude drift.
        accel_bias  a constant added to the accelerometer, in milligravity.
                    This is what makes position run away as t squared.
        noise       white noise on both, as a multiple of the real part's own
                    noise density.
    """

    modality = "imu"
    channels = tuple(mod.IMU.channels)

    def __init__(self, hz: float = 100.0, motion: str = "still", bias_dps: float = 0.6,
                 accel_bias_mg: float = 5.0, noise: float = 1.0, keep: int = 60000,
                 realtime: bool = True):
        self.motion = motion
        self.bias = np.array([bias_dps, -0.6 * bias_dps, 0.4 * bias_dps])
        self.accel_bias = np.array([1.0, -0.6, 0.4]) * accel_bias_mg * 1e-3
        self.noise = float(noise)
        self._rng = np.random.default_rng(7)
        # what really happened, kept so that a page can draw it
        self.true_p = np.zeros(3)
        self.true_v = np.zeros(3)
        self.true_yaw = 0.0
        self.true_R = np.eye(3)        # body axes to world, as it really is
        self.origin = np.zeros(3)      # where the estimate was told to start from
        self._trace_t, self._trace_p = [], []
        super().__init__("virtual IMU", hz, keep, realtime)
        self.info = "virtual IMU  %.0f Hz  gyro bias %.2f deg/s  accel bias %.1f mg" % (
            hz, bias_dps, accel_bias_mg)

    # ---- the trajectory the measurements are derived from ----
    #
    # Every motion begins with WARMUP seconds of genuine stillness. Alignment
    # needs a stationary interval and so does any honest zero, and a simulation
    # that starts moving at t = 0 cannot be aligned at all.
    WARMUP = 8.0
    MOVE, REST = 1.5, 4.5           # one slide, and the pause after it
    STEP = 0.5                      # how far each slide goes, in meters

    def _true_accel(self, t):
        """World-frame acceleration of the board, m/s^2, excluding gravity."""
        t -= self.WARMUP
        if t < 0:
            return np.zeros(3)
        m = self.motion
        if m == "tap":
            phase = t % 4.0
            return np.array([12.0, 0.0, 0.0]) if 1.0 <= phase < 1.06 else np.zeros(3)
        if m.startswith("shake"):
            f = 2.0 if "2" in m else 8.0
            return np.array([3.0 * math.sin(2 * math.pi * f * t), 0.0, 0.0])
        if m.startswith("sweep"):
            # 1 Hz rising to 8 Hz over twenty seconds, then again. The phase
            # is the integral of the frequency, not the product.
            u = t % 20.0
            return np.array([2.5 * math.sin(2 * math.pi * (u + 7.0 * u * u / 40.0)), 0.0, 0.0])
        if m == "slide and stop":
            # A minimum-jerk slide: it leaves rest and arrives at rest with
            # both the velocity and the acceleration at zero, which is what a
            # hand actually does. A rectangular push instead leaves a long
            # stretch of constant velocity, and constant velocity is
            # indistinguishable from rest to an accelerometer, so the whole
            # movement would be thrown away by the very method meant to keep it.
            phase = t % (self.MOVE + self.REST)
            if phase >= self.MOVE:
                return np.zeros(3)
            u = phase / self.MOVE
            a = self.STEP / (self.MOVE ** 2) * (60 * u - 180 * u * u + 120 * u ** 3)
            return np.array([a, 0.0, 0.0])
        return np.zeros(3)

    def _true_rate(self, t):
        """
        Body-frame angular rate of the board, deg/s.

        "slow tilt" turns about the board's own y, which tips it. That is the
        motion that matters: a turn about the vertical leaves gravity where it
        was, so an estimator that assumes a fixed attitude survives it, and the
        page would be demonstrating nothing. Tipping is what puts a component
        of gravity into the horizontal, and it is what attitude tracking is
        for.
        """
        t -= self.WARMUP
        if t < 0:
            return np.zeros(3)
        if self.motion == "slow tilt":
            return np.array([0.0, 12.0 * math.sin(2 * math.pi * t / 14.0), 0.0])
        if self.motion == "turn on the spot":
            return np.array([0.0, 0.0, 18.0 * math.sin(2 * math.pi * t / 12.0)])
        if self.motion.startswith("shake"):
            f = 2.0 if "2" in self.motion else 8.0
            return np.array([0.0, 25.0 * math.sin(2 * math.pi * f * t), 0.0])
        if self.motion.startswith("sweep"):
            u = t % 20.0
            return np.array([0.0, 10.0 * math.sin(2 * math.pi * (u + 7.0 * u * u / 40.0)), 0.0])
        if self.motion == "tap":
            phase = t % 4.0
            return np.array([0.0, 60.0, 0.0]) if 1.0 <= phase < 1.06 else np.zeros(3)
        return np.zeros(3)

    def _make(self, count):
        for _ in range(count):
            self._step()

    def _step(self):
        t = self._n * self.dt
        self._n += 1

        a_true = self._true_accel(t)
        w_body = self._true_rate(t)
        self.true_v = self.true_v + a_true * self.dt
        self.true_p = self.true_p + self.true_v * self.dt

        # The attitude is carried as a rotation matrix and turned by the body
        # rate, so that a board which is tipped stays tipped. Yaw alone was not
        # enough: tipping is the whole reason attitude has to be tracked.
        w = np.radians(w_body) * self.dt
        k = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        self.true_R = _orthonormal(self.true_R @ (np.eye(3) + k))
        self.true_yaw = math.degrees(math.atan2(self.true_R[1, 0], self.true_R[0, 0]))

        if self._n % 10 == 0:
            self._trace_t.append(t)
            self._trace_p.append(self.true_p.copy())
            if len(self._trace_t) > 20000:
                del self._trace_t[:10000]
                del self._trace_p[:10000]

        # The accelerometer measures specific force: what it is being pushed
        # with, plus the push the floor gives it against gravity. At rest that
        # is one g straight up, which is why a still board does not read zero.
        a_body = self.true_R.T @ (a_true + np.array([0.0, 0.0, G])) / G      # in g

        a = a_body + self.accel_bias + self._rng.normal(0, 0.0018 * self.noise, 3)
        w = w_body + self.bias + self._rng.normal(0, 0.055 * self.noise, 3)
        m = np.array([22.0, -3.0, -42.0]) + self._rng.normal(0, 0.35 * self.noise, 3)
        self._emit(t, a.tolist() + w.tolist() + m.tolist())

    def set_origin(self) -> None:
        """
        Take where the board is now as the origin of the truth.

        Only the origin. Zeroing the true velocity as well would be a lie: if
        the board happened to be moving, the simulation would then carry on
        from a state it was never in, and every later comparison would be
        against a trajectory that did not happen. Alignment is done from rest
        anyway, so at that moment the velocity is already zero.
        """
        self.origin = self.true_p.copy()
        self._trace_t.clear()
        self._trace_p.clear()

    @property
    def displacement(self) -> np.ndarray:
        """How far the board really is from where the estimate started."""
        return self.true_p - self.origin

    def true_track(self):
        """Time and world position of what really happened, for drawing against."""
        if not self._trace_t:
            return np.zeros(0), np.zeros((0, 3))
        return np.asarray(self._trace_t), np.asarray(self._trace_p) - self.origin


def _orthonormal(R):
    """
    Pull a matrix back onto the rotation group.

    Turning a matrix by a small-angle approximation, ten thousand times, walks
    it off the group: the axes stop being unit length and stop being at right
    angles, and after a minute the simulated board is being stretched as well
    as turned.
    """
    u, _, vt = np.linalg.svd(R)
    return u @ vt


EMG_PATTERNS = ("rest", "bursts", "ramp", "fatigue", "tremor")


class SimulatedEmg(Simulated):
    """
    A muscle that does not exist, contracting on a schedule that is known.

    Surface EMG looks like noise, because it is the sum of thousands of motor
    unit action potentials firing out of step with each other. It is made here
    the way it is usually modeled: white noise shaped to the band a real
    muscle occupies, multiplied by an envelope that is the contraction. That
    envelope is the truth: the RMS the pages estimate is meant to follow it.

        pattern     what the muscle does
        amplitude   the RMS of a full contraction, in millivolts
        hum         mains interference at 50 Hz, in millivolts
        noise       the electrode's own floor, as a multiple of the real one

    "fatigue" holds a contraction while the spectrum slides down, the way a
    tiring muscle's does, so that the median frequency has something to fall
    from. The true center frequency is available for checking against.
    """

    modality = "emg"

    WARMUP = 3.0
    ON, OFF = 2.0, 2.0
    FADE_FROM, FADE_TO, FADE_OVER = 120.0, 70.0, 40.0

    # With two muscles, what each electrode sees is mostly its own muscle
    # and some of the other: x1 = s1 + 0.4 s2, x2 = 0.3 s1 + s2.
    MIX = np.array([[1.0, 0.4], [0.3, 1.0]])

    def __init__(self, hz: float = 1000.0, pattern: str = "bursts", amplitude_mv: float = 1.0,
                 hum_mv: float = 0.05, noise: float = 1.0, mains_hz: float = 50.0,
                 channels: int = 1, keep: int = 120000, realtime: bool = True):
        self.pattern = pattern
        self.amplitude = float(amplitude_mv)
        self.hum = float(hum_mv)
        self.noise = float(noise)
        self.mains = float(mains_hz)
        self.n_ch = 2 if int(channels) >= 2 else 1
        self.channels = [mod.Channel("emg%d_mv" % (i + 1), "emg%d" % (i + 1), "mV",
                                     mod.SERIES[i], "muscle") for i in range(self.n_ch)]
        self._rng = np.random.default_rng(11)
        self._centre = None
        self._shape = None
        self._shape2 = None
        self._gain = 1.0
        self.rate = float(hz)
        self._cap = int(60 * self.rate)
        self._src = np.zeros((self._cap, 2))         # the two sources, by sample number
        self._reshape(self.FADE_FROM)
        super().__init__("virtual muscle", hz, keep, realtime)
        self.info = "virtual muscle  %.0f Hz  %s  %.1f mV  hum %.2f mV%s" % (
            hz, pattern, amplitude_mv, hum_mv, "  two muscles" if self.n_ch == 2 else "")

    # ---- the truth ----
    def true_envelope(self, t):
        """The RMS the signal was made with at time t, in mV."""
        t = np.asarray(t, float) - self.WARMUP
        p = self.pattern
        if p == "bursts":
            phase = np.mod(t, self.ON + self.OFF)
            return np.where((t >= 0) & (phase < self.ON), self.amplitude, 0.0)
        if p == "ramp":
            phase = np.mod(t, 8.0) / 8.0
            tri = 1.0 - np.abs(2 * phase - 1.0)
            return np.where(t >= 0, self.amplitude * tri, 0.0)
        if p == "fatigue":
            return np.where(t >= 0, 0.8 * self.amplitude, 0.0)
        if p == "tremor":
            return np.where(t >= 0,
                            0.5 * self.amplitude * (1 + 0.6 * np.sin(2 * np.pi * 8.0 * t)), 0.0)
        return np.zeros_like(t)

    def true_active(self, t):
        return np.asarray(self.true_envelope(t)) > 0.05 * self.amplitude

    def true_centre(self, t) -> float:
        """Where the spectrum was centered at time t, in Hz."""
        if self.pattern != "fatigue":
            return self.FADE_FROM
        u = min(1.0, max(0.0, (float(t) - self.WARMUP) / self.FADE_OVER))
        return self.FADE_FROM + (self.FADE_TO - self.FADE_FROM) * u

    # ---- making it ----
    def _reshape(self, centre):
        """A band-pass around `center`, and the gain that makes its output unit RMS."""
        if self._centre is not None and abs(centre - self._centre) < 1.0:
            return
        self._centre = centre
        lo, hi = max(20.0, centre * 0.45), min(centre * 2.2, 0.45 * self.rate)
        sos = dsp.butterworth(2, "bandpass", lo, self.rate, hi)
        keep = self._shape.z if self._shape is not None else None
        self._shape = dsp.Sos(sos, 1)
        if keep is not None and keep.shape == self._shape.z.shape:
            self._shape.z = keep
        keep2 = self._shape2.z if self._shape2 is not None else None
        self._shape2 = dsp.Sos(sos, 1)
        if keep2 is not None and keep2.shape == self._shape2.z.shape:
            self._shape2.z = keep2
        probe = dsp.Sos(sos, 1).process(np.random.default_rng(3).normal(size=8000))
        self._gain = 1.0 / max(float(probe[2000:].std()), 1e-9)

    def _make(self, count):
        # Blocks of at most half a second, so that the fatigue slide can move
        # the shaping filter between them.
        while count > 0:
            k = min(count, int(self.rate * 0.5))
            t = (self._n + np.arange(k)) * self.dt
            self._reshape(self.true_centre(float(t[0])))
            white = self._rng.normal(size=k)
            shaped = self._shape.process(white) * self._gain
            env = self.true_envelope(t)
            hum = self.hum * np.sin(2 * np.pi * self.mains * t)
            s1 = shaped * env
            if self.n_ch == 2:
                # the second muscle ramps while the first bursts
                phase = np.mod(t - self.WARMUP, 8.0) / 8.0
                env2 = np.where(t >= self.WARMUP,
                                self.amplitude * (1.0 - np.abs(2 * phase - 1.0)), 0.0)
                s2 = self._shape2.process(self._rng.normal(size=k)) * self._gain * env2
                idx = (self._n + np.arange(k)) % self._cap
                self._src[idx, 0] = s1
                self._src[idx, 1] = s2
                x = np.column_stack([s1, s2]) @ self.MIX.T
                x += hum[:, None] + 0.008 * self.noise * self._rng.normal(size=(k, 2))
                for i in range(k):
                    self._n += 1
                    self._emit(float(t[i]), [float(x[i, 0]), float(x[i, 1])], 32)
            else:
                x = s1 + hum + 0.008 * self.noise * self._rng.normal(size=k)
                for i in range(k):
                    self._n += 1
                    self._emit(float(t[i]), [float(x[i])], 20)
            count -= k

    def true_sources(self, t):
        """The two muscles' own signals at times t, before they were mixed."""
        if self.n_ch < 2:
            return None
        n = np.round(np.asarray(t, float) * self.rate).astype(int)
        return self._src[n % self._cap]


class SimulatedPpg(Simulated):
    """
    A fingertip that does not exist, with a heart rate that is known exactly.

    Each beat is a pulse of the same shape, a systolic peak with a smaller
    dicrotic hump behind it, laid down at intervals set by the heart rate. The
    interval breathes: it lengthens and shortens with a slow sinus rhythm the
    way a real one does, by `hrv` per cent. Two wavelengths are produced, and
    the ratio of their pulsatile parts is set from the oxygen saturation asked
    for, through the same line the page reads it back with.

        bpm        heart rate
        hrv        how much the interval wanders, per cent
        spo2       oxygen saturation the ratio of ratios is set to
        motion     slow wander and the odd bump, as a fraction of the pulse
        hum        mains pick-up, as a fraction of the pulse
        noise      the photodiode's own floor, as a multiple of the real one

    `beat_times` and `true_bpm(t)` are the truth the heart-rate page is checked
    against.
    """

    modality = "ppg"

    DC_IR, DC_RED = 120000.0, 90000.0        # counts, the way an 18-bit oximeter reads
    AC_IR = 0.02                             # the pulse, as a fraction of DC

    def __init__(self, hz: float = 100.0, bpm: float = 72.0, hrv: float = 3.0,
                 spo2: float = 97.0, motion: float = 0.0, hum: float = 0.0,
                 noise: float = 1.0, mains_hz: float = 50.0, keep: int = 60000,
                 realtime: bool = True):
        self.bpm = float(bpm)
        self.hrv = float(hrv)
        self.spo2 = float(spo2)
        self.motion = float(motion)
        self.hum = float(hum)
        self.noise = float(noise)
        self.mains = float(mains_hz)
        self.channels = list(mod.PPG.channels)
        self._rng = np.random.default_rng(5)
        self._phase = 0.0                    # 0..1 through the current beat
        self.beat_times = []                 # when each beat began
        # the ratio of ratios the page should read back: SpO2 = 110 - 25 R
        self.ratio = (110.0 - self.spo2) / 25.0
        super().__init__("virtual fingertip", hz, keep, realtime)
        self.info = "virtual fingertip  %.0f Hz  %.0f BPM  SpO2 %.0f %%" % (hz, bpm, spo2)

    def true_bpm(self, t) -> float:
        """The instantaneous heart rate the signal was made with."""
        return self.bpm / (1.0 + 0.01 * self.hrv * math.sin(2 * math.pi * 0.25 * float(t)))

    @staticmethod
    def pulse(phase):
        """One heartbeat, 0..1 of the way through it, peaking near 0.15."""
        phase = np.asarray(phase, float)
        return (np.exp(-0.5 * ((phase - 0.15) / 0.05) ** 2)
                + 0.35 * np.exp(-0.5 * ((phase - 0.45) / 0.09) ** 2))

    def _make(self, count):
        for _ in range(count):
            t = self._n * self.dt
            self._n += 1
            ibi = 60.0 / self.true_bpm(t)
            self._phase += self.dt / ibi
            if self._phase >= 1.0:
                self._phase -= 1.0
                self.beat_times.append(t)
                if len(self.beat_times) > 4000:
                    del self.beat_times[:2000]
            p = float(self.pulse(self._phase))
            wander = self.motion * (0.6 * math.sin(2 * math.pi * 0.3 * t)
                                    + 0.4 * math.sin(2 * math.pi * 0.07 * t))
            if self.motion > 0 and (t % 9.0) < 0.4:
                wander += 2.0 * self.motion * math.sin(2 * math.pi * 6.0 * t)
            hum = self.hum * math.sin(2 * math.pi * self.mains * t)
            ir = self.DC_IR * (1.0 + self.AC_IR * (p + wander + hum))
            red = self.DC_RED * (1.0 + self.AC_IR * self.ratio * (p + wander + hum))
            ir += self._rng.normal(0, 60.0 * self.noise)
            red += self._rng.normal(0, 60.0 * self.noise)
            self._emit(t, [ir, red], 24)


# ---------------------------------------------------------------
# a recording, played back
# ---------------------------------------------------------------
class FileSource(Source):
    """
    A CSV made earlier, played back as if it were arriving now.

    The first row names the columns, `n,micros,ax_g,...`, which is all this
    needs: the names say what the channels are and the modality follows from
    them, exactly as they do on the wire. The clock is rebuilt from `micros`,
    so the replay runs at the rate the recording was made at, faster if asked,
    and starts again at the end if asked to.
    """

    kind = "FILE"

    def __init__(self, path: str, speed: float = 1.0, loop: bool = True, keep: int = 60000):
        super().__init__(os.path.basename(path), keep)
        self.path = path
        self.speed = float(speed)              # 0 means as fast as possible
        self.loop = bool(loop)
        keys, data = _load_csv(path)
        self.channels = mod.channels_from_keys(keys)
        self.modality = mod.modality_of(self.channels)
        if self.modality == "imu":
            have = {c.key: i for i, c in enumerate(self.channels)}
            order = [have[k] for k in _IMU_ORDER]
            data = np.column_stack([data[:, :2], data[:, 2:][:, order]])
            self.channels = list(mod.IMU.channels)
        self.data = data
        us = data[:, 1]
        dt = np.diff(us) * 1e-6
        dt = dt[(dt > 0) & (dt < 1.0)]
        self.hz = 1.0 / float(np.median(dt)) if len(dt) else 100.0
        self.length = float((us[-1] - us[0]) * 1e-6) if len(us) > 1 else 0.0
        self.info = "%s  %d samples  %.0f Hz  %.1f s  %s" % (
            self.name, len(data), self.hz, self.length, self.badge)
        self._i = 0
        self._pass = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        t_file = (self.data[:, 1] - self.data[0, 1]) * 1e-6
        # unwrap a micros() that turned over during the recording
        step = np.diff(t_file)
        step[step < 0] += US_WRAP * 1e-6
        t_file = np.concatenate([[0.0], np.cumsum(step)])
        span = float(t_file[-1] + (1.0 / self.hz)) if len(t_file) else 0.0
        wall0 = time.perf_counter()
        while not self._stop.is_set():
            if self._i >= len(self.data):
                if not self.loop or len(self.data) == 0:
                    self.info = "%s  finished" % self.name
                    return
                self._i = 0
                self._pass += 1
                wall0 = time.perf_counter()
            offset = self._pass * span
            if self.speed > 0:
                elapsed = (time.perf_counter() - wall0) * self.speed
                due = int(np.searchsorted(t_file, elapsed, side="right"))
                if due <= self._i:
                    time.sleep(0.003)
                    continue
            else:
                due = min(len(self.data), self._i + 200)
                time.sleep(0.001)
            for i in range(self._i, due):
                row = self.data[i]
                t = self._book(row[0], int(row[1]) % US_WRAP, 8 * len(row))
                self.queue.append((offset + t_file[i], row[2:].tolist()))
            self._i = due

    def where(self) -> str:
        return "FILE  %s" % self.name


def _load_csv(path):
    """
    (channel keys, rows) out of a recording. Rows are n, micros, channels.

    Reads both what the Week 2 scripts write, a plain header row, and what a
    sketch prints, a `#COLUMNS` line, and skips any other line beginning with
    `#`. A row that is not all numbers is dropped, not fatal: the first line of
    a recording made from a live port is usually a fragment.
    """
    keys = None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip().lstrip("﻿")
            if not text:
                continue
            found = mod.parse_header(text)
            if found is not None:
                if keys is None:
                    keys = found
                continue
            if text[0] == "#":
                continue
            parts = text.split(",")
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("%s holds no rows of numbers." % os.path.basename(path))
    width = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == width]
    data = np.asarray(rows, float)
    if keys is None:
        # no header at all: unnamed channels, which the generic pages can still draw
        keys = ["ch%d_raw" % (i + 1) for i in range(width - 2)]
    if len(keys) != width - 2:
        raise RuntimeError("%s names %d channels but its rows hold %d."
                           % (os.path.basename(path), len(keys), width - 2))
    return keys, data


class Recorder:
    """
    Write what a source produces to a CSV that FileSource, and the Week 2
    scripts, can read back.

    The format is the one already in use: a header row `n,micros,<channels>`,
    then one row per sample. Nothing else, so that numpy.loadtxt with
    skiprows=1 reads it and so does a spreadsheet.
    """

    def __init__(self, path: str, channels):
        self.path = path
        self.channels = list(channels)
        self.n = 0
        self.started = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(mod.column_names(self.channels))

    def feed(self, samples) -> None:
        for t, values in samples:
            self.n += 1
            # Eight significant figures: an oximeter's 120,000 counts keep two
            # decimals, a gyroscope's 0.061 keeps them all, and the file is
            # still half the size of one written with repr().
            self._w.writerow([self.n, int(round(t * 1e6))]
                             + ["%.8g" % v for v in values[:len(self.channels)]])

    @property
    def seconds(self) -> float:
        return time.time() - self.started

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def open_source(kind: str, target: str = "", **kw) -> Source:
    """
    One call for the toolbar.

        kind    "USB", "BLE", "SIM" or "FILE"
        target  the port, the board's name, or the file
        kw      for SIM: modality="imu" | "emg" | "ppg" and that simulator's knobs
                for USB: expect="auto" | "imu" | "emg" | "ppg"
                for FILE: speed, loop
    """
    if kind == "USB":
        return SerialSource(target or "auto", expect=kw.get("expect", "auto"))
    if kind == "BLE":
        return BleSource(target)
    if kind == "FILE":
        return FileSource(target, speed=kw.get("speed", 1.0), loop=kw.get("loop", True))
    which = kw.pop("modality", "imu")
    if which == "emg":
        return SimulatedEmg(**kw)
    if which == "ppg":
        return SimulatedPpg(**kw)
    return SimulatedSource(**kw)
