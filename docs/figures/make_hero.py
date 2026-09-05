"""Take the README hero: the Orientation page, on a simulated IMU, in both palettes.

    python docs/figures/make_hero.py

Opens the real window off screen, connects the simulated IMU, waits for the
buffers to fill, and saves the window as it is drawn. Nothing is staged; the
picture is what the program shows. Writes hero_orientation.png and
hero_orientation-dark.png, and the three pages under docs/img/.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

# On a machine with a display the native platform is used and the window is kept
# off the screen, so the system's fonts render. Off-screen Qt on Windows has no
# font database and draws every glyph as a box. On a headless runner set
# QT_QPA_PLATFORM=offscreen, where fontconfig supplies the fonts.
HIDDEN = os.environ.get("QT_QPA_PLATFORM", "") != "offscreen"

from PySide6 import QtCore, QtWidgets  # noqa: E402

from studio import theme  # noqa: E402
from studio.shell import Studio  # noqa: E402

IMU = dict(modality="imu", motion="shake 2 Hz", bias_dps=0.8, accel_bias_mg=6.0, noise=1.0)
EMG = dict(modality="emg", pattern="bursts", amplitude_mv=1.0, hum_mv=0.1, channels=2)
PPG = dict(modality="ppg", bpm=72.0, hrv=3.0, spo2=96.0, motion=0.2)
IMG = os.path.join(os.path.dirname(HERE), "img")


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.004)


def shoot(app, dark, out, source=IMU, page_title="Orientation", settle=14.0):
    theme.apply(app, dark=dark)
    win = Studio()
    win.quiet_errors = True
    win.resize(1280, 800)
    if HIDDEN:
        win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    win.show()
    win._open("SIM", "", dict(source))
    pump(app, 1.0)
    for i, page in enumerate(win.pages):
        if page.title == page_title:
            win.goto(i)
    pump(app, settle)                    # a trace needs history behind it
    win.grab().save(out)
    win._close()
    win.close()
    print("wrote", out)


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    shoot(app, False, os.path.join(HERE, "hero_orientation.png"))
    shoot(app, True, os.path.join(HERE, "hero_orientation-dark.png"))
    # the three pages the README shows below the hero
    shoot(app, False, os.path.join(IMG, "envelope.png"), EMG, "Envelope", 10.0)
    shoot(app, False, os.path.join(IMG, "heartrate.png"), PPG, "Heart rate", 12.0)
    shoot(app, True, os.path.join(IMG, "spectrogram.png"), IMU, "Spectrogram", 14.0)
