# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
pages_frames.py
The IMU's own axes against the world's: the rotation that takes gravity off.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import core, theme
from .pages_motion import Scene3D, _euler_of
from .shell import Every, Page, YRange, card, decimate, label, row

BODY, WORLD = "#e08a1e", theme.ACCENT


class FramesPage(Page):
    title = "Frames"
    subtitle = "Body frame to world frame: the rotation that removes gravity"
    modalities = ("imu",)

    def __init__(self):
        super().__init__()
        self.view = Scene3D(half=1.0)
        self.view.set_extent(1.0)
        for name in ("ref", "track", "drop"):
            self.view.set(name, [])
        self.view.add("body_vec", BODY, 2.6)
        self.view.add("world_vec", WORLD, 2.6)
        self.view.add("gravity", theme.TRUTH, 1.4, dash=True)

        self.bplot = pg.PlotWidget()
        bp = theme.plot(self.bplot, "seconds before now", "body frame", "", "g")
        theme.title(bp, "a, in the board's own axes")
        theme.legend(bp)
        self.b_curves = [bp.plot(pen=theme.pen(c, 1.4), name=n) for c, n in
                         zip((theme.X_AXIS, theme.Y_AXIS, theme.Z_AXIS), "xyz")]
        self.by = YRange(bp, floor=0.5)

        self.wplot = pg.PlotWidget()
        wp = theme.plot(self.wplot, "seconds before now", "world frame", "", "g")
        theme.title(wp, "R a − g: the same reading in the world's axes, gravity taken off")
        theme.legend(wp)
        self.w_curves = [wp.plot(pen=theme.pen(c, 1.4), name=n) for c, n in
                         zip((theme.X_AXIS, theme.Y_AXIS, theme.Z_AXIS), ("east", "north", "up"))]
        self.wy = YRange(wp, floor=0.05)
        for p in (bp, wp):
            p.setXRange(-10, 0, padding=0)
        # under a 3-D view: shorter than the house minimum, or a 640-pixel
        # window cannot hold the page
        self.bplot.setMinimumHeight(120)
        self.wplot.setMinimumHeight(120)

        self.readout = label("", "Readout")
        self.readout.setStyleSheet("font-size: %dpt;" % (theme.SIZE_BODY + 1))
        self.state = label("", "Caption", wrap=True)
        side, sl = card(label("ROTATION", "Section"), self.readout, self.state)
        sl.addStretch(1)
        side.setFixedWidth(360)

        scene, _ = card(self.view)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(theme.GAP)
        col.addLayout(row(scene, side), 3)
        plots = QtWidgets.QHBoxLayout()
        plots.setSpacing(theme.GAP)
        plots.addWidget(card(self.bplot)[0])
        plots.addWidget(card(self.wplot)[0])
        col.addLayout(plots, 2)
        self.body(col)
        self.slow = Every(6)

    def on_show(self, session, source):
        self.explain_changed.emit()

    def tick(self, session, source):
        d = session.last(10.0)
        if len(d) < 3 or not hasattr(session, "relative"):
            return
        q = d[-1, core.QUAT]
        R = core.quat_to_matrix(q)                 # body axes to world axes
        a_body = d[-1, core.A]
        a_world = R @ a_body
        earth = d[-1, core.EARTH]

        size = self.view.half * 0.9
        self.view.set("body", core.board_edges(R, size))
        for i in range(3):
            self.view.set("axis%d" % i, [[0, 0, 0], R @ (np.eye(3)[i] * size * 0.6)])
        # the reading as it is in the body frame, drawn as if the body axes were
        # the world's: it tips with the board. The same reading rotated by R
        # stays vertical, because that is what gravity does.
        self.view.set("body_vec", [[0, 0, 0], a_body * size])
        self.view.set("world_vec", [[0, 0, 0], a_world * size])
        self.view.set("gravity", [[0, 0, 0], [0, 0, -size]])
        self.view.refresh()

        t = d[:, core.T] - d[-1, core.T]
        step = max(1, len(d) // 400)
        for i in range(3):
            self.b_curves[i].setData(t[::step], d[::step, core.AX + i])
            self.w_curves[i].setData(t[::step], d[::step, core.EX + i])
        self.by.fit(d[:, core.A])
        self.wy.fit(d[:, core.EARTH])
        if not self.slow():
            return

        rpy = _euler_of(R)
        faint = theme.INK_FAINT
        m = "".join("<tr>%s</tr>" % "".join("<td align=right>%6.3f</td>" % v for v in R[i])
                    for i in range(3))
        self.readout.setText(
            "<table cellspacing=4>"
            "<tr><td style='color:%s'>a<sub>body</sub></td>%s<td style='color:%s'>g</td></tr>"
            "<tr><td style='color:%s'>R a<sub>body</sub></td>%s<td style='color:%s'>g</td></tr>"
            "<tr><td style='color:%s'>R a &minus; g</td>%s<td style='color:%s'>g</td></tr>"
            "<tr><td colspan=5><hr></td></tr>"
            "<tr><td style='color:%s' valign=top>R</td><td colspan=4><table cellspacing=2>%s"
            "</table></td></tr>"
            "<tr><td colspan=5><hr></td></tr>"
            "<tr><td style='color:%s'>roll, pitch, yaw</td>"
            "<td align=right>%.1f</td><td align=right>%.1f</td><td align=right>%.1f</td>"
            "<td style='color:%s'>°</td></tr></table>"
            % (faint, "".join("<td align=right>%7.3f</td>" % v for v in a_body), faint,
               faint, "".join("<td align=right>%7.3f</td>" % v for v in a_world), faint,
               faint, "".join("<td align=right>%7.3f</td>" % v for v in earth), faint,
               faint, m, faint, rpy[0], rpy[1], rpy[2], faint))
        left = float(np.linalg.norm(earth))
        self.state.setText(
            "<span style='color:%s'>&#9632;</span> the reading in body axes &nbsp; "
            "<span style='color:%s'>&#9632;</span> the same reading rotated into world "
            "axes.<br>Tilt the board: the orange arrow tips with it, the green one stays "
            "vertical. What is left after gravity is %.0f mg." % (BODY, WORLD, 1000 * left))

    def explain(self):
        return dict(
            head="Coordinate frames",
            equation="<span style='font-size:16pt'>a<sub>world</sub> &nbsp;=&nbsp; R(q) "
                     "a<sub>body</sub> &nbsp;&nbsp;&nbsp;&nbsp; "
                     "a<sub>motion</sub> &nbsp;=&nbsp; a<sub>world</sub> &minus; "
                     "<span style='color:%s'>g</span></span>" % theme.ACCENT,
            terms=(("R(q)", "the 3×3 rotation from the board's axes to the world's, "
                            "from the attitude quaternion"),
                   ("<span style='color:%s'>g</span>" % theme.ACCENT,
                    "(0, 0, 1) g in world axes, whichever way the board lies"),
                   ("<span style='color:%s'>R wrong by &#948;&#952;</span>" % theme.GUILTY,
                    "leaves g sin &#948;&#952; of gravity in the horizontal")),
            why="The accelerometer reports in the board's axes. Gravity is fixed in the "
                "world's. Only after rotating into the world's axes can one constant be "
                "subtracted and the rest be called motion.",
            tip="Set the simulated motion to slow tilt and watch the left plot swing while "
                "the right one stays near zero.",
            source=("Rotation matrix, Wikipedia", "https://en.wikipedia.org/wiki/Rotation_matrix"),
            sources=(("Quaternions and spatial rotation, Wikipedia",
                      "https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation"),
                     ("Euler angles, Wikipedia", "https://en.wikipedia.org/wiki/Euler_angles")))
