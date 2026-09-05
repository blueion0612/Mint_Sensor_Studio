# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
check_studio.py
Drive every page of Sensor Studio, on every kind of sensor, without anyone watching.

Opens the real window, connects each simulated sensor in turn, walks the
rail, works every control on every page that applies, and renders a picture
of each one. It fails loudly on three things,
because each has shipped before:

    an exception on any page          the window would die mid-lecture
    a frame that costs too long       the window would feel slow
    two pieces of text overlapping    the complaint that started this rewrite

Then it records a few seconds, plays the recording back through the fourth
source, and checks that beats were found in it; opens the dark palette in the
smallest window the layout is designed for and looks for collisions again;
and, if a board is plugged in, opens the cable.

The overlap check is the interesting one. Qt layouts cannot overlap by
construction, but a fixed-size panel whose text is too long, or a plot title
written over a legend, still can. So it walks the real widget tree and compares
the on-screen rectangles of everything that draws text.
"""

import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)                # the repository root
SHOTS = os.path.join(os.environ.get("SCRATCH", os.path.join(HERE, "_runs")), "studio")

# A frame goes out every 45 ms. The budget was set on a laptop; a shared CI runner
# is two to three times slower, so STUDIO_BUDGET_MS lets it say so without
# loosening the local check.
BUDGET_MS = float(os.environ.get("STUDIO_BUDGET_MS", "26.0"))
sys.path.insert(0, APP)

# STUDIO_HIDDEN=1 keeps the windows off the screen while the system's own fonts
# still draw. Off-screen Qt on Windows has no font database and renders every
# glyph as a box, which changes the text widths this check measures.
HIDDEN = os.environ.get("STUDIO_HIDDEN") == "1"

# the three simulated sensors, and what they are told to do
SOURCES = [
    ("imu", dict(modality="imu", motion="shake 2 Hz", bias_dps=0.8, accel_bias_mg=6.0,
                 noise=1.0)),
    ("emg", dict(modality="emg", pattern="bursts", amplitude_mv=1.0, hum_mv=0.1,
                 channels=2)),
    ("ppg", dict(modality="ppg", bpm=72.0, hrv=3.0, spo2=96.0, motion=0.2)),
]


def main():
    os.makedirs(SHOTS, exist_ok=True)
    for f in os.listdir(SHOTS):
        if f.endswith(".png"):
            os.remove(os.path.join(SHOTS, f))

    from PySide6 import QtWidgets, QtCore
    from studio import theme, sources, catalogue
    from studio.shell import Studio, RECORDINGS

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    theme.apply(app)
    fails = []

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.004)

    def measure(win, page, kind):
        cost = []
        for _ in range(40):
            t0 = time.perf_counter()
            win._frame()
            app.processEvents()
            cost.append((time.perf_counter() - t0) * 1e3)
        cost.sort()
        mean, worst = sum(cost) / len(cost), cost[-1]
        notes = []
        if win.last_error is not None:
            err, win.last_error = win.last_error, None
            win.timer.start(win.pace.interval)
            notes.append("RAISED")
            fails.append("%s on %s raised: %s" % (page.title, kind,
                                                  err.strip().splitlines()[-1]))
        if mean > BUDGET_MS:
            notes.append("SLOW")
            fails.append("%s on %s: %.1f ms a frame, over the %.0f ms budget"
                         % (page.title, kind, mean, BUDGET_MS))
        bad = overlaps(page)
        if bad:
            notes.append("OVERLAP")
            fails.append("%s on %s: %s" % (page.title, kind, "; ".join(bad[:3])))
        info = page.explain()
        if not info:
            notes.append("no explanation")
            fails.append("%s has no explanation" % page.title)
        elif not (info.get("source") or info.get("sources")):
            notes.append("no link")
            fails.append("%s explains but links to nothing" % page.title)
        return mean, worst, notes

    win = Studio()
    win.resize(1440, 900)
    win.quiet_errors = True          # collect tracebacks, never a modal box
    if HIDDEN:
        win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    win.show()
    print("  groups: " + ", ".join("%s (%d)" % (g.name, len(g.pages)) for g in catalogue.GROUPS))

    for kind, kw in SOURCES:
        print()
        print("  %s   %s" % (kind.upper(), ", ".join("%s=%s" % i for i in kw.items()
                                                    if i[0] != "modality")))
        win._open("SIM", "", dict(kw))
        if win.source is None:
            fails.append("the simulated %s would not open" % kind)
            continue
        if win.badge.text() != kind.upper():
            fails.append("toolbar badge says %r for a %s source" % (win.badge.text(), kind))
        pump(12.0 if kind == "imu" else 8.0)     # let the buffers fill
        print("  %-14s %8s %8s   %s" % ("page", "frame", "worst", "checks"))
        print("  " + "-" * 68)
        for i, page in enumerate(win.pages):
            item = win._items[i]
            enabled = bool(item.flags() & QtCore.Qt.ItemIsEnabled)
            if page.fits(win.session) != enabled:
                fails.append("%s on %s: the rail %s it" % (
                    page.title, kind, "greys" if not enabled else "offers"))
            if not enabled:
                continue
            win.goto(i)
            pump(1.2)
            work = exercise(page, app, pump)
            pump(4.0 if kind == "imu" else 2.5)
            # A page that needs an alignment cannot be measured until it has
            # one, and an alignment needs a board that is standing still.
            if hasattr(page, "estimators"):
                win._open("SIM", "", dict(modality="imu", motion="slide and stop",
                                          bias_dps=0.8, accel_bias_mg=6.0))
                win.goto(i)
                pump(3.0)
                page.start()
                end = time.time() + 25
                while page.estimators[0].t0 is None and time.time() < end:
                    pump(0.3)
                if page.estimators[0].t0 is None:
                    fails.append("%s never finished aligning on a board that was "
                                 "standing still" % page.title)
                pump(6.0)
            mean, worst, notes = measure(win, page, kind)
            notes.append(work)
            print("  %-14s %6.1f ms %6.1f ms   %s"
                  % (page.title, mean, worst, ", ".join(n for n in notes if n) or "ok"))
            win.grab().save(os.path.join(SHOTS, "%s_%d_%s.png"
                                         % (kind, i + 1, page.title.lower())))

    # --- record, then play the recording back through the fourth source ---
    print()
    win._open("SIM", "", dict(modality="ppg", bpm=80.0))
    pump(2.0)
    win.rec.click()
    path = win.recorder.path if win.recorder else None
    pump(6.0)
    win.rec.click()
    if not path or not os.path.isfile(path):
        fails.append("Record did not write a file")
    else:
        win._open("FILE", path, dict(speed=4.0, loop=True))
        if win.source is None or win.source.modality != "ppg":
            fails.append("the recording did not play back as PPG")
        else:
            for i, page in enumerate(win.pages):
                if page.title == "Heart rate":
                    win.goto(i)
            pump(8.0)
            h = win.session.heart(10.0)
            if h["n"] < 4 or abs(h["mean"] - 80.0) > 4.0:
                fails.append("played back, but read %.1f BPM from %d beats where 80 was "
                             "recorded" % (h["mean"], h["n"]))
            print("  recorded %s, played it back at 4x: %d beats, %.1f BPM"
                  % (os.path.basename(path), h["n"], h["mean"]))
            win.grab().save(os.path.join(SHOTS, "file_heartrate.png"))
        os.remove(path)
        try:
            os.rmdir(RECORDINGS)         # only if the check made it and it is empty
        except OSError:
            pass

    # --- the cable, if the board happens to be plugged in ---
    print()
    boards = [p for p in sources.list_serial_ports() if p[2]]
    if boards:
        win._open("USB", boards[0][0], {})
        if win.source is not None:
            pump(3.0)
            win.goto(0)
            pump(1.5)
            print("  USB   %s   %s   %s" % (win.source.where(), win.source.badge,
                                           win.source.rate_text()))
            win.grab().save(os.path.join(SHOTS, "usb.png"))
        else:
            fails.append("the board is plugged in but the USB source would not open")
    else:
        print("  USB   no board plugged in, not checked")
    win._close()
    win.close()

    # --- the dark palette, in the smallest window the layout is designed for ---
    print()
    print("  the dark palette in a 1024 x 640 window")
    theme.apply(app, dark=True)
    dark = Studio()
    dark.quiet_errors = True
    dark.resize(1024, 640)
    if HIDDEN:
        dark.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    dark.show()
    small = 0
    for kind, kw in SOURCES:
        dark._open("SIM", "", dict(kw))
        pump(3.0)
        for i, page in enumerate(dark.pages):
            if not page.fits(dark.session):
                continue
            dark.goto(i)
            pump(0.6)
            for _ in range(4):
                dark._frame()
                app.processEvents()
            if dark.last_error is not None:
                fails.append("%s in the dark palette raised: %s"
                             % (page.title, dark.last_error.strip().splitlines()[-1]))
                dark.last_error = None
                dark.timer.start(dark.pace.interval)
            bad = overlaps(page)
            if bad:
                small += 1
                fails.append("%s at 1024x640: %s" % (page.title, "; ".join(bad[:2])))
            dark.grab().save(os.path.join(SHOTS, "small_dark_%s_%d_%s.png"
                                          % (kind, i + 1, page.title.lower())))
    print("  %s" % ("nothing collided" if not small else "%d pages collided" % small))
    dark._close()
    dark.close()
    theme.apply(app, dark=False)

    print()
    if fails:
        print("  %d problem(s):" % len(fails))
        for f in fails:
            print("    - " + f)
        return 1
    print("  every page drew on every sensor, stayed inside the frame budget, had no")
    print("  overlapping text, and a recording came back through the file source")
    print("  pictures in %s" % SHOTS)
    return 0


def exercise(page, app, pump):
    """Work whatever controls the page has, so a broken one shows up here."""
    from PySide6 import QtWidgets
    from studio.shell import Slider

    touched = 0
    for w in page.findChildren(QtWidgets.QComboBox):
        for k in range(w.count()):
            w.setCurrentIndex(k)
            app.processEvents()
            touched += 1
    for w in page.findChildren(Slider):
        lo = w._bar.minimum()
        hi = w._bar.maximum()
        for v in (lo, (lo + hi) // 2, hi, (lo + hi) // 3):
            w._bar.setValue(v)
            app.processEvents()
            touched += 1
    for w in page.findChildren(QtWidgets.QCheckBox):
        w.toggle()
        app.processEvents()
        w.toggle()
        app.processEvents()
        touched += 2
    for w in page.findChildren(QtWidgets.QPushButton):
        w.click()
        app.processEvents()
        touched += 1
        pump(0.25)
    return "%d controls worked" % touched


def overlaps(page):
    """
    Pairs of text widgets whose rectangles intersect on screen.

    Only visible widgets with something in them count, and a widget that
    contains the other is a parent, not a collision.
    """
    from PySide6 import QtWidgets
    import pyqtgraph as pg

    kinds = (QtWidgets.QLabel, QtWidgets.QPushButton, QtWidgets.QCheckBox,
             QtWidgets.QComboBox, pg.PlotWidget)
    boxes = []
    for w in page.findChildren(QtWidgets.QWidget):
        if not isinstance(w, kinds) or not w.isVisible():
            continue
        if isinstance(w, pg.PlotWidget):
            # A plot squeezed below its minimum paints over whatever the layout
            # put under it. That is a collision too, and it has shipped.
            text = "plot"
        else:
            text = w.text() if hasattr(w, "text") else w.currentText()
        if not text or not text.strip():
            continue
        r = w.rect().translated(w.mapTo(page, w.rect().topLeft()))
        if r.width() < 2 or r.height() < 2:
            continue
        boxes.append((w, text, r))

    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            wa, ta, ra = boxes[i]
            wb, tb, rb = boxes[j]
            if wb.isAncestorOf(wa) or wa.isAncestorOf(wb):
                continue
            hit = ra.intersected(rb)
            if hit.width() > 2 and hit.height() > 2:
                out.append("%r over %r" % (ta[:22], tb[:22]))
    return out


if __name__ == "__main__":
    sys.exit(main())
