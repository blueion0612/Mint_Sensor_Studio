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

# A frame goes out every 45 ms. The budget was set on a laptop; a shared CI
# runner is two to three times slower, so STUDIO_BUDGET_MS lets it say so
# without loosening the local check.
BUDGET_MS = float(os.environ.get("STUDIO_BUDGET_MS", "26.0"))
sys.path.insert(0, APP)

# STUDIO_HIDDEN=1 keeps the windows off the screen while the system's own
# fonts still draw. Off-screen Qt on Windows has no font database and renders
# every glyph as a box, which changes the text widths this check measures.
HIDDEN = os.environ.get("STUDIO_HIDDEN") == "1"
# The standard windows are laid out for a real screen. Qt's off-screen
# platform reports 800 x 600, which would switch the window into its
# small-screen scrolling mode and test that instead; the short-screen check
# does so on purpose, with its own rectangle.
BIG_SCREEN = (0, 0, 1920, 1080)


def open_window(Studio, QtCore, screen=BIG_SCREEN):
    """A Studio laid out for `screen`, hidden if asked, with the override reset."""
    Studio.SCREEN_OVERRIDE = QtCore.QRect(*screen) if screen else None
    try:
        win = Studio()
    finally:
        Studio.SCREEN_OVERRIDE = None
    win.quiet_errors = True
    if HIDDEN:
        win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    return win

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
    Studio.AUTO_CONNECT = False      # a board on the bench must not take a test from the simulator

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    theme.apply(app)
    fails = []

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.004)

    def frames(win, n=40):
        cost = []
        for _ in range(n):
            t0 = time.perf_counter()
            win._frame()
            app.processEvents()
            cost.append((time.perf_counter() - t0) * 1e3)
        cost.sort()
        return sum(cost) / len(cost), cost[-1]

    def measure(win, page, kind):
        mean, worst = frames(win)
        notes = []
        if mean > BUDGET_MS:
            # Once more before believing it. The machine running this has
            # other things to do, and a page that sits a few ms under the
            # budget has gone over it on a build and back under on the next.
            # A page that is really too slow is slow both times.
            again, worst2 = frames(win)
            notes.append("re-measured %.1f then %.1f ms" % (mean, again))
            mean, worst = min(mean, again), min(worst, worst2)
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

    win = open_window(Studio, QtCore)   # collects tracebacks, never a modal box
    win.resize(1440, 900)
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
    dark = open_window(Studio, QtCore)
    dark.resize(1024, 640)
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

    # --- a screen shorter than the layout: a 1366 x 768 laptop at 125 % ---
    # scaling has 1093 x 614 logical pixels, 574 once the taskbar is off. The
    # window has to fit that and the bottom panel has to stay reachable.
    print()
    print("  a 1093 x 574 screen (a 1366 x 768 laptop at 125 % scaling, taskbar off)")
    tiny = open_window(Studio, QtCore, (0, 0, 1093, 574))
    tiny.show()
    app.processEvents()
    if tiny.width() > 1093 or tiny.height() > 574:
        fails.append("on a 1093 x 574 screen the window opened %d x %d"
                     % (tiny.width(), tiny.height()))
    scroll = tiny.findChild(QtWidgets.QScrollArea, "Cramped")
    if scroll is None:
        fails.append("on a screen shorter than the layout the window did not fall "
                     "back to scrolling")
    else:
        tiny._open("SIM", "", dict(SOURCES[0][1]))
        pump(3.0)
        cramped = 0
        for i, page in enumerate(tiny.pages):
            if not page.fits(tiny.session):
                continue
            tiny.goto(i)
            pump(0.5)
            for _ in range(4):
                tiny._frame()
                app.processEvents()
            if tiny.last_error is not None:
                fails.append("%s on the short screen raised: %s"
                             % (page.title, tiny.last_error.strip().splitlines()[-1]))
                tiny.last_error = None
                tiny.timer.start(tiny.pace.interval)
            bad = overlaps(page)
            if bad:
                cramped += 1
                fails.append("%s on the short screen: %s" % (page.title, "; ".join(bad[:2])))
        # the bottom panel has to be reachable: scroll to the end and look for it
        bar = scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        app.processEvents()
        view = scroll.viewport()
        bottom = tiny.explain.mapTo(view, QtCore.QPoint(0, tiny.explain.height())).y()
        if bar.maximum() <= 0 or bottom > view.height() + 1:
            fails.append("on the short screen the bottom panel cannot be scrolled into "
                         "view (scroll range %d, panel bottom at %d of %d)"
                         % (bar.maximum(), bottom, view.height()))
        # The page must be the layout's floor and no taller: the first build of
        # this fallback scrolled 1228 px, a wrapped label's stale size hint.
        want = theme.MIN_H - view.height()
        if abs(bar.maximum() - want) > 2:
            fails.append("on the short screen the page scrolls %d px where the floor "
                         "needs %d" % (bar.maximum(), want))
        print("  window %d x %d, scrolls %d px, %s"
              % (tiny.width(), tiny.height(), bar.maximum(),
                 "nothing collided" if not cramped else "%d pages collided" % cramped))
        tiny.grab().save(os.path.join(SHOTS, "short_screen.png"))
        tiny._close()
    tiny.close()

    # --- nothing connected: every page is offered, and the theory pages, which
    # read no sensor, have to work in full ---
    print()
    print("  nothing connected")
    bare = open_window(Studio, QtCore)
    bare.resize(1440, 900)
    bare.show()
    app.processEvents()
    theory_titles = [p.title for g, _k, p in bare.entries if g.key == "theory"]
    if len(theory_titles) < 5:
        fails.append("the rail has %d theory pages, expected five" % len(theory_titles))
    for i, page in enumerate(bare.pages):
        if not (bare._items[i].flags() & QtCore.Qt.ItemIsEnabled):
            fails.append("%s is greyed with nothing connected; every page should be offered"
                         % page.title)
            continue
        bare.goto(i)
        pump(0.3)
        work = exercise(page, app, pump) if page.title in theory_titles else ""
        for _ in range(4):
            bare._frame()
            app.processEvents()
        if bare.last_error is not None:
            fails.append("%s with nothing connected raised: %s"
                         % (page.title, bare.last_error.strip().splitlines()[-1]))
            bare.last_error = None
            bare.timer.start(bare.pace.interval)
        if page.title in theory_titles:
            bad = overlaps(page)
            if bad:
                fails.append("%s with nothing connected: %s" % (page.title, "; ".join(bad[:2])))
            info = page.explain()
            if not info or not (info.get("source") or info.get("sources")):
                fails.append("%s explains nothing, or links to nothing" % page.title)
            print("  %-18s %s" % (page.title, work))
            bare.grab().save(os.path.join(SHOTS, "theory_%s.png"
                                          % page.title.lower().replace(" ", "_")))
    bare.close()

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
