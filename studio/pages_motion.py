# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_motion.py
The two pages about where the board is and which way it is pointing.

    Orientation   attitude, and the three things that keep it from drifting
    Position      the double integration, and exactly which term ruins it

Both draw in three dimensions with an orthographic projection worked out here
rather than with an OpenGL scene, for two reasons. A perspective view
foreshortens the far side of the volume, and a trajectory drawn in one cannot
be measured by eye. And a lecture room is the wrong place to discover that a
laptop has no usable OpenGL driver.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import core, theme
from .shell import Every, Page, Slider, YRange, card, decimate, label, row, steps


# ---------------------------------------------------------------
# a small three-dimensional view, drawn with flat lines
# ---------------------------------------------------------------
class Scene3D(pg.PlotWidget):
    """
    World-frame lines, projected. Drag to turn it.

    Everything added is a named line whose points are given in meters in the
    world frame; the view owns the projection and nothing else has to know
    about it.
    """

    def __init__(self, half=1.0):
        super().__init__()
        theme.plot(self)
        pi = self.getPlotItem()
        pi.hideAxis("left")
        pi.hideAxis("bottom")
        pi.setAspectLocked(True)
        pi.setMouseEnabled(False, False)
        pi.hideButtons()
        self.setBackground(theme.SURFACE)
        self.elev, self.azim = 22.0, -58.0
        self.half = half
        self._drag = None
        self._items = {}
        self._world = {}
        self.add("grid", theme.LINE_SOFT, 1.0)
        self.add("box", theme.LINE, 1.0)
        self.add("ref", theme.TRUTH, 1.2, dash=True)
        self.add("track", theme.ACCENT, 1.8)
        self.add("drop", theme.LINE, 1.0, dash=True)
        self.add("body", theme.INK_SOFT, 1.6)
        for i, c in enumerate((theme.X_AXIS, theme.Y_AXIS, theme.Z_AXIS)):
            self.add("axis%d" % i, c, 2.6)
        self.mark = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(theme.ACCENT),
                                       pen=pg.mkPen(theme.SURFACE, width=2))
        pi.addItem(self.mark)
        self.set_extent(half)

    def add(self, name, colour, width, dash=False):
        item = self.getPlotItem().plot(pen=theme.pen(colour, width, dash))
        self._items[name] = item
        self._world[name] = np.zeros((0, 3))
        return item

    def set(self, name, points):
        self._world[name] = np.asarray(points, float).reshape(-1, 3)
        self._project(name)

    def _project(self, name):
        p = self._world[name]
        if len(p) == 0:
            self._items[name].setData([], [])
            return
        xy = core.project(p, self.elev, self.azim)
        self._items[name].setData(xy[:, 0], xy[:, 1])

    def refresh(self):
        for name in self._items:
            self._project(name)
        o = core.project(np.zeros((1, 3)), self.elev, self.azim)
        self.mark.setData(o[:, 0], o[:, 1])

    def set_extent(self, half):
        """Size the volume. Everything fixed inside it is redrawn to match."""
        self.half = float(half)
        h = self.half
        grid = []
        for v in np.linspace(-h, h, 7):
            grid += [[v, -h, -h], [v, h, -h], [np.nan] * 3]
            grid += [[-h, v, -h], [h, v, -h], [np.nan] * 3]
        self.set("grid", grid)
        self.set("box", core.board_edges(np.eye(3), 1.0) * 0 + _box(h))
        self.refresh()
        r = h * 1.45
        self.getPlotItem().setRange(xRange=(-r, r), yRange=(-r, r), padding=0)

    # dragging turns the view, which is what everyone tries first
    def mousePressEvent(self, ev):
        self._drag = ev.position()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        d = ev.position() - self._drag
        self._drag = ev.position()
        self.azim -= d.x() * 0.4
        self.elev = max(-89.0, min(89.0, self.elev + d.y() * 0.4))
        self.refresh()

    def mouseReleaseEvent(self, ev):
        self._drag = None


def _box(h):
    """The twelve edges of the volume, as one path with gaps."""
    c = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    e = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
    out = []
    for i, j in e:
        out += [c[i], c[j], [np.nan] * 3]
    return np.array(out)


def _euler_of(R):
    """Roll, pitch and heading of a rotation matrix, in degrees."""
    import math
    return np.degrees([math.atan2(R[2, 1], R[2, 2]),
                       -math.asin(max(-1.0, min(1.0, R[2, 0]))),
                       math.atan2(R[1, 0], R[0, 0])])


# ---------------------------------------------------------------
# orientation
# ---------------------------------------------------------------
class OrientationPage(Page):
    title = "Orientation"
    subtitle = "Attitude estimation from gyroscope and accelerometer"
    modalities = ("imu",)

    def __init__(self):
        super().__init__()
        self.view = Scene3D(half=1.0)
        self.view.set_extent(1.0)
        self.view.set("ref", [])
        self.view.set("track", [])
        self.view.set("drop", [])

        self.euler = pg.PlotWidget()
        ep = theme.plot(self.euler, "seconds before now", "angle", "", "°")
        ep.setTitle("Roll, pitch and heading", color=theme.INK_SOFT,
                    size="%dpt" % theme.SIZE_SMALL, justify="left")
        ep.setXRange(-30, 0, padding=0)
        theme.legend(ep)
        self.e_curves = [ep.plot(pen=theme.pen(c, 1.6), name=n) for c, n in
                         ((theme.X_AXIS, "roll"), (theme.Y_AXIS, "pitch"),
                          (theme.Z_AXIS, "heading"))]

        self.set_level = QtWidgets.QPushButton("Set level here")
        self.set_level.setObjectName("Primary")
        self.set_level.clicked.connect(self._level)
        self.clear_level = QtWidgets.QPushButton("Clear")
        self.clear_level.clicked.connect(self._unlevel)
        self.gyro_only = QtWidgets.QCheckBox("also show the gyroscope integrated on its own")
        self.gyro_only.setChecked(True)
        self.naive = ep.plot(pen=theme.pen(theme.GUILTY, 1.4, dash=True),
                             name="heading, gyroscope alone")
        self.euler.setMinimumHeight(120)
        self.ey = YRange(ep, floor=5.0)
        self.slow = Every(6)

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)

        left, _ = card(self.view)
        right, rl = card(label("ATTITUDE", "Section"), self.readout, self.state)
        rl.addStretch(1)
        rl.addWidget(label("Lay the board down and press this. Whatever pose it is in "
                           "becomes zero, so a board propped up by its own cable does "
                           "not read as tilted.", "Caption", wrap=True))
        rl.addLayout(row(self.set_level, self.clear_level, None))
        right.setFixedWidth(360)

        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addLayout(row(left, right), 3)
        bottom, _ = card(self.euler, row(self.gyro_only, None))
        col.addWidget(bottom, 2)
        self.body(col)
        self._naive_yaw = 0.0
        self._naive_t = None
        self._hist_t, self._hist_naive = [], []

    def on_show(self, session, source):
        self._session = session
        self.explain_changed.emit()

    def _level(self):
        s = getattr(self, "_session", None)
        if s is not None:
            s.set_level()

    def _unlevel(self):
        s = getattr(self, "_session", None)
        if s is not None:
            s.clear_level()

    def on_hide(self):
        self._naive_yaw = 0.0
        self._naive_t = None
        self._hist_t.clear()
        self._hist_naive.clear()

    def tick(self, session, source):
        d = session.last(30.0)
        if len(d) < 3:
            return

        # the board, drawn in the attitude the filter believes, against the
        # pose the student called level
        R = session.relative(d[-1, core.QUAT])
        size = self.view.half * 1.15
        self.view.set("body", core.board_edges(R, size))
        for i in range(3):
            arm = R @ (np.eye(3)[i] * size * 0.72)
            self.view.set("axis%d" % i, [[0, 0, 0], arm])
        self.view.refresh()

        e = np.array([_euler_of(session.relative(q))
                      for q in d[::max(1, len(d) // 400), core.QUAT]])
        t = d[::max(1, len(d) // 400), core.T]
        t = t - d[-1, core.T]
        for i in range(3):
            self.e_curves[i].setData(t, e[:, i])
        self.ey.fit(e)

        # the same heading with nothing correcting it: the gyroscope, integrated
        last_t = d[-1, core.T]
        if self._naive_t is None:
            self._naive_t = d[0, core.T]
        new = d[d[:, core.T] > self._naive_t]
        for r in new:
            dt = r[core.T] - self._naive_t
            self._naive_t = r[core.T]
            if 0 < dt < 1.0:
                self._naive_yaw += r[core.GZ] * dt
            self._hist_t.append(r[core.T])
            self._hist_naive.append(self._naive_yaw)
        if len(self._hist_t) > 40000:
            del self._hist_t[:20000]
            del self._hist_naive[:20000]
        if self.gyro_only.isChecked() and self._hist_t:
            ht = np.asarray(self._hist_t) - last_t
            keep = ht > -30.0
            self.naive.setData(ht[keep], np.asarray(self._hist_naive)[keep])
        else:
            self.naive.setData([], [])

        if not self.slow():
            return
        bias = session.gyro_bias()
        rpy = _euler_of(R)
        self.readout.setText(
            "<table cellspacing=5>"
            "<tr><td style='color:%s'>roll</td><td align=right><b>%7.2f</b></td>"
            "<td style='color:%s'>°</td></tr>"
            "<tr><td style='color:%s'>pitch</td><td align=right><b>%7.2f</b></td>"
            "<td style='color:%s'>°</td></tr>"
            "<tr><td style='color:%s'>heading</td><td align=right><b>%7.2f</b></td>"
            "<td style='color:%s'>°</td></tr>"
            "<tr><td colspan=3><hr></td></tr>"
            "<tr><td style='color:%s'>gyro bias x</td><td align=right>%7.3f</td>"
            "<td style='color:%s'>°/s</td></tr>"
            "<tr><td style='color:%s'>y</td><td align=right>%7.3f</td>"
            "<td style='color:%s'>°/s</td></tr>"
            "<tr><td style='color:%s'>z</td><td align=right>%7.3f</td>"
            "<td style='color:%s'>°/s</td></tr>"
            "</table>"
            % (theme.INK_FAINT, rpy[0], theme.INK_FAINT,
               theme.INK_FAINT, rpy[1], theme.INK_FAINT,
               theme.INK_FAINT, rpy[2], theme.INK_FAINT,
               theme.INK_FAINT, bias[0], theme.INK_FAINT,
               theme.INK_FAINT, bias[1], theme.INK_FAINT,
               theme.INK_FAINT, bias[2], theme.INK_FAINT))

        bits = []
        if not session.settled():
            bits.append("Still settling. The filter runs a fast gain for the first "
                        "few seconds so that it does not start from nowhere.")
        if session.rejecting():
            bits.append("<span style='color:%s'>The accelerometer is being ignored: "
                        "it is reading something other than gravity, so the gyroscope "
                        "is carrying the attitude alone.</span>" % theme.WARN)
        drift = float(self._hist_naive[-1] - rpy[2]) if self._hist_naive else 0.0
        if abs(drift) > 2.0:
            bits.append("The gyroscope on its own is <b>%.1f°</b> from the corrected "
                        "heading by now." % drift)
        if session.level is not None:
            bits.append("<span style='color:%s'>Zeroed to the pose you put it down in. "
                        "Press Clear to read against true vertical again.</span>"
                        % theme.GOOD)
        self.state.setText("<br>".join(bits) or
                           "<span style='color:%s'>Roll and pitch are being corrected "
                           "against gravity every sample, so they do not drift at all. "
                           "Heading has nothing correcting it.</span>" % theme.INK_SOFT)

    def explain(self):
        return dict(
            head="Attitude estimation",
            equation="<span style='font-size:16pt'>q&#775; &nbsp;=&nbsp; "
                     "&#189; q &#8855; ( &#969; "
                     "<span style='color:%s'>&minus; b&#770;</span> ) "
                     "&nbsp;+&nbsp; <span style='color:%s'>k ( a&#770; &#215; g&#770; )"
                     "</span></span>" % (theme.ACCENT, theme.ACCENT),
            terms=(("<span style='color:%s'>b&#770;</span>" % theme.ACCENT,
                    "gyroscope offset, learnt while stationary"),
                   ("<span style='color:%s'>a&#770; &#215; g&#770;</span>" % theme.ACCENT,
                    "correction towards gravity"),
                   ("<span style='color:%s'>heading</span>" % theme.GUILTY,
                    "unobserved without a magnetometer")),
            why="Gravity corrects roll and pitch every sample. Nothing observes a turn "
                "about the vertical, so heading is left to the gyroscope and drifts.",
            tip="Turn the board a full circle and put it back: roll and pitch return, "
                "heading does not.",
            source=("Attitude and heading reference system, Wikipedia",
                    "https://en.wikipedia.org/wiki/Attitude_and_heading_reference_system"),
            sources=(("x-io Fusion, the open-source AHRS this program uses",
                      "https://github.com/xioTechnologies/Fusion"),))


# ---------------------------------------------------------------
# position
# ---------------------------------------------------------------
class PositionPage(Page):
    title = "Position"
    subtitle = "Dead reckoning: velocity and position by integration"
    modalities = ("imu",)

    def __init__(self):
        super().__init__()
        self.estimators = core.make_estimators()
        self.chosen = 0
        self.tolerance = 0.5
        self._align = None
        self._a0 = None
        self._rot0 = np.eye(3)

        self.view = Scene3D(half=0.8)
        self.view.set("drop", [])

        self.dist = pg.PlotWidget()
        dp = theme.plot(self.dist, "seconds since this estimator started",
                        "displacement |p|", "", "m")
        dp.setLogMode(False, True)
        self.dist_curves = []
        for est in self.estimators:
            self.dist_curves.append(dp.plot(pen=theme.pen(est.colour, 1.9), name=est.name))
        # The y axis is in log mode, so a horizontal line sits at the logarithm
        # of the value it stands for. Passing the value itself put the
        # half-meter line at three meters and labeled it nought.
        self.limit = pg.InfiniteLine(angle=0, pen=theme.pen(theme.TRUTH, 1.3, dash=True),
                                     label="",
                                     labelOpts={"position": 0.04, "color": theme.INK_FAINT})
        dp.addItem(self.limit)
        self.dist.setMinimumHeight(120)
        self.dp = dp
        self._span = None
        self._decade = None
        self.slow = Every(6)
        self.truth_curve = dp.plot(pen=theme.pen(theme.GOOD, 1.6, dash=True))

        self.buttons = []
        brow = QtWidgets.QHBoxLayout()
        brow.setSpacing(theme.GAP)
        for i, est in enumerate(self.estimators):
            b = QtWidgets.QPushButton(est.name)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=i: self.choose(k))
            self.buttons.append(b)
            brow.addWidget(b)
        brow.addStretch(1)
        self.align_btn = QtWidgets.QPushButton("Align and start")
        self.align_btn.setObjectName("Primary")
        self.align_btn.clicked.connect(self.start)
        brow.addWidget(self.align_btn)

        # A bar, not a percentage in a sentence. Two seconds of nothing moving
        # is long enough to look like a program that has stopped.
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setVisible(False)

        self.tau = Slider("leak τ", 0.2, 10.0, 1.0, 0.1, "s", 1, width=170)
        self.tau.changed.connect(self._set_tau)
        self.tol = Slider("tolerance", 0.05, 5.0, 0.5, 0.05, "m", 2, width=170)
        self.tol.changed.connect(self._set_tol)

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 2))
        self.state = label("", "Caption", wrap=True)

        left, _ = card(self.view)
        right, rl = card(label("ESTIMATE", "Section"), self.readout, self.bar, self.state)
        rl.addStretch(1)
        rl.addLayout(row(self.tau, self.tol))
        right.setFixedWidth(380)

        top = row(left, right)
        bottom, _ = card(self.dist, brow)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addLayout(top, 3)
        col.addWidget(bottom, 2)
        self.body(col)
        self.choose(0)
        self._set_tol(self.tolerance)

    # ---- controls ----
    def _set_tau(self, v):
        for est in self.estimators:
            if isinstance(est, core.LeakyIntegration):
                est.leak = v
        self.explain_changed.emit()

    def _set_tol(self, v):
        self.tolerance = v
        self.limit.setValue(float(np.log10(max(v, 1e-9))))
        self.limit.label.setFormat("tolerance %.3g m" % v)

    def choose(self, k):
        self.chosen = k
        for i, b in enumerate(self.buttons):
            b.setChecked(i == k)
            b.setStyleSheet("" if i != k else
                            "background:%s; border-color:%s; color:white; font-weight:600;"
                            % (self.estimators[k].colour, self.estimators[k].colour))
        self.explain_changed.emit()

    ALIGN_SECONDS = 1.2        # of stillness. One pass, not three.

    def start(self):
        """
        Align, then set every estimator going from rest at the same instant.

        One window of stillness gives everything at once: the mean gyroscope
        reading is its offset, the mean accelerometer reading is the vertical,
        and both go straight into the filter. There is nothing to converge to
        afterwards, because the answer was measured rather than approached.

        All four estimators start together, on identical input, so any
        difference between them belongs to the processing.
        """
        self._align = {"held": 0.0, "rows": [], "seen": -1e18, "began": None}
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.align_btn.setEnabled(False)

    def _draw_board(self, d, session, here):
        """The board, at the estimated position, in the attitude it is lying in."""
        R = session.relative(d[-1, core.QUAT])
        size = self.view.half * 0.22
        self.view.set("body", core.board_edges(R, size) + here)
        self.view.set("drop", [[here[0], here[1], here[2]],
                               [here[0], here[1], -self.view.half]])
        for i in range(3):
            arm = here + R @ (np.eye(3)[i] * size * 0.7)
            self.view.set("axis%d" % i, [here, arm])
        self.view.refresh()

    def _step_alignment(self, d, session, source):
        """One frame of the alignment. Any movement starts it again."""
        job = self._align
        if job["began"] is None:
            job["began"] = d[-1, core.T]
        for r in d[d[:, core.T] > job["seen"]]:
            if r[core.STILL] < 0.5:
                job["held"], job["rows"] = 0.0, []
                continue
            job["held"] += 1.0 / max(session.hz, 1.0)
            job["rows"].append(r)
        job["seen"] = d[-1, core.T]

        want = self.ALIGN_SECONDS
        self.bar.setValue(int(100 * min(1.0, job["held"] / want)))
        waited = d[-1, core.T] - job["began"]

        if job["held"] < want or not job["rows"]:
            if waited > 8.0:
                self.state.setText(
                    "<span style='color:%s'>It has not been still yet. Put the board "
                    "down and let go.</span>" % theme.WARN)
            else:
                self.state.setText("Aligning. Hands off the board.")
            return

        block = np.asarray(job["rows"])
        self._a0 = block[:, core.A].mean(axis=0)
        self._rot0 = core.attitude_from_gravity(self._a0)
        session.set_gyro_bias(block[:, core.W].mean(axis=0))
        session.set_attitude(self._rot0)
        session.set_level(self._rot0)
        # With the attitude set from this reading, what the filter will report
        # as earth acceleration for it is exactly this. Subtracting it takes
        # off the accelerometer's own offset, the same offset the other route
        # removes by subtracting the whole resting vector.
        self._e0 = self._rot0 @ self._a0 - np.array([0.0, 0.0, 1.0])
        for est in self.estimators:
            est.start(d[-1, core.T], self._a0, self._rot0, self._e0)
        if source is not None and getattr(source, "truth", False):
            source.set_origin()
        self._fed = None
        self._align = None
        self.bar.setVisible(False)
        self.align_btn.setEnabled(True)

    # ---- the frame ----
    def tick(self, session, source):
        d = session.last(2.0)
        if len(d) < 5:
            return

        if self._align is not None:
            # The live view keeps running while this happens. A page that goes
            # blank for a second and a half is a page that looks broken.
            self._draw_board(d, session, np.zeros(3))
            self._step_alignment(d, session, source)
            return

        if self.estimators[0].t0 is None:
            self.state.setText("Put the board down, hands off, and press "
                               "<b>Align and start</b>. A second of stillness gives the "
                               "gyroscope offset, the vertical and the origin.")
            return

        # feed every estimator the samples it has not seen
        fresh = session.last(2.0)
        done = self.estimators[0]
        last = getattr(self, "_fed", None)
        rows = fresh if last is None else fresh[fresh[:, core.T] > last]
        prev = last
        for r in rows:
            dt = 1.0 / session.hz if prev is None else r[core.T] - prev
            prev = r[core.T]
            if dt <= 0 or dt > 1.0:
                continue
            for est in self.estimators:
                est.step(r, dt)
        self._fed = prev

        top, run = 0.0, 0.0
        for i, est in enumerate(self.estimators):
            t, p = est.track()
            if len(t) < 2:
                continue
            r = np.maximum(np.linalg.norm(p, axis=1), 1e-5)
            top = max(top, float(r[-1]))
            run = max(run, float(t[-1]))
            self.dist_curves[i].setData(*decimate(t, r))

        # Both ranges move in steps, so the axis is rebuilt when the run gets
        # longer or the error crosses a decade, and not on every frame.
        span = steps(max(run, 16.0), 2.0)
        if span != self._span:
            self._span = span
            self.dp.setXRange(0, span, padding=0)
        decade = int(np.ceil(np.log10(max(top, self.tolerance) * 3)))
        if decade != self._decade:
            self._decade = decade
            self.dp.setYRange(-4.0, float(decade), padding=0)

        if source is not None and getattr(source, "truth", False):
            tt, tp = source.true_track()
            if len(tt) > 2:
                self.truth_curve.setData(tt - tt[0],
                                         np.maximum(np.linalg.norm(tp, axis=1), 1e-5))
        else:
            self.truth_curve.setData([], [])

        est = self.estimators[self.chosen]
        t, p = est.track()
        reach = float(np.abs(p).max()) if len(p) else 0.0
        want = max(self.tolerance * 1.6, 2.0 ** np.ceil(np.log2(max(1.35 * reach, 1e-6))))
        if abs(want - self.view.half) > 1e-9:
            self.view.set_extent(want)
        self.view._items["track"].setPen(theme.pen(est.colour, 1.9))
        self.view.set("track", p if len(p) else np.zeros((0, 3)))
        self.view.set("ref", _sphere(self.tolerance)
                      if self.tolerance < self.view.half * 0.9 else [])
        self._draw_board(d, session, est.p)

        if not self.slow():
            return
        rows_html = []
        for e in self.estimators:
            mark = "<b>&#8250;</b>&nbsp;" if e is est else "&nbsp;&nbsp;"
            rows_html.append(
                "<tr><td style='color:%s'>&#8212;</td><td style='color:%s'>%s%s</td>"
                "<td align=right><b>%s</b></td></tr>"
                % (e.colour, theme.INK_SOFT, mark,
                   e.name.replace(" ", "&nbsp;"), _metres(e.distance)))
        elapsed = max(0.0, float(est.track()[0][-1])) if len(est.track()[0]) else 0.0
        self.readout.setText(
            "<table cellspacing=5>%s"
            "<tr><td colspan=3><hr></td></tr>"
            "<tr><td colspan=2 style='color:%s'>running for</td>"
            "<td align=right>%.0f s</td></tr></table>"
            % ("".join(rows_html), theme.INK_FAINT, elapsed))

        if source is not None and getattr(source, "truth", False):
            err = float(np.linalg.norm(est.p - source.displacement))
            self.state.setText(
                "<span style='color:%s'>&#8212;</span> what really happened.<br>"
                "The simulated board really moved <b>%s</b> from the origin. "
                "This estimator says <b>%s</b>, so it is <b>%s</b> wrong."
                % (theme.GOOD, _metres(float(np.linalg.norm(source.displacement))),
                   _metres(est.distance), _metres(err)))
        else:
            self.state.setText(
                "A real board's true position is unknown. Leave it still, and every "
                "metre on the plot is error.")

    def explain(self):
        est = self.estimators[self.chosen]
        live = ""
        if est.t0 is not None and len(est.track()[0]):
            t = est.track()[0][-1]
            live = ("<br><span style='color:%s'>now: t = %.1f s, |p| = %s</span>"
                    % (theme.INK_FAINT, t, _metres(est.distance)))
        return dict(head=est.name, equation=est.equation, terms=est.terms,
                    why=est.why, live=live, source=est.source, sources=est.sources,
                    tip="Press each of the four with the board still, then tilt it and "
                        "press them again.")


def _sphere(r, n=41):
    u = np.linspace(0, 2 * np.pi, n)
    o, c, s = np.zeros_like(u), r * np.cos(u), r * np.sin(u)
    gap = np.array([[np.nan] * 3])
    return np.concatenate([np.column_stack([c, s, o]), gap,
                           np.column_stack([c, o, s]), gap,
                           np.column_stack([o, c, s])])


def _metres(v: float) -> str:
    """A distance a person can read, over the eight decades this page spans."""
    if v < 0.001:
        return "%.2f mm" % (v * 1000)
    if v < 1.0:
        return "%.0f mm" % (v * 1000)
    if v < 1000:
        return "%.2f m" % v
    return "%.1f km" % (v / 1000)
