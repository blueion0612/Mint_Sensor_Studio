# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
The entry point: what has to be installed, the --check report, and opening the window.

    sensor-studio                 open the window
    sensor-studio --dark          the same window in the dark palette
    sensor-studio --check         say what is installed, and stop

`python -m studio` and `python sensor_studio.py` at the repository root run the
same function.
"""

import sys

# What has to be there, and what only takes something away if it is not.
NEEDED = (("numpy", "numpy", "the arrays every page reads"),
          ("PySide6", "PySide6", "the window itself"),
          ("pyqtgraph", "pyqtgraph", "the plots"))
OPTIONAL = (("serial", "pyserial", "the USB cable. Without it: the radio, the simulated "
                                   "sensors and recordings"),
            ("bleak", "bleak", "the radio. Without it: the cable, the simulated sensors "
                               "and recordings"),
            ("imufusion", "imufusion", "the reference attitude filter. Without it, the "
                                       "numpy one in studio/core.py, which must agree"),
            ("scipy", "scipy", "faster filters. Without it, the biquad cascade in "
                               "studio/dsp.py, which gives the same answer"))


def no_wheel(package):
    """
    Is this optional package one PyPI has no pre-built wheel for on this
    machine? pip would then try to compile it, fail for want of a compiler,
    and say so in a way that does not look like this is the reason.

    imufusion ships wheels for x86-64 Windows, both kinds of Mac and Linux,
    for Python 3.9 to 3.14. Elsewhere the numpy filter in studio/core.py,
    which is required to agree with it, is used.
    """
    import platform
    if package == "imufusion":
        on_windows_arm = sys.platform == "win32" and platform.machine().upper() == "ARM64"
        return on_windows_arm or sys.version_info[:2] >= (3, 15)
    return False


def report():
    """Say what is installed and what is not, then stop."""
    import importlib
    import platform
    import sys as _sys

    print()
    print("  MINT Sensor Studio")
    print("  %s, Python %s, %s" % (platform.system(), platform.python_version(),
                                   platform.machine()))
    # "installed but still missing" is nearly always two different Pythons
    print("  running:  %s" % _sys.executable)
    print()

    warnings = []
    if _sys.version_info[:2] < (3, 10):
        warnings.append("Python %s is too old for the current PySide6. Make the "
                        "course environment: conda create -n imu python=3.12"
                        % platform.python_version())

    # The current Qt is built for macOS 13 and newer. On an older Mac pip finds
    # no PySide6 it can take, and its message does not say why.
    if _sys.platform == "darwin":
        release = platform.mac_ver()[0]
        try:
            major = int(release.split(".")[0])
        except (ValueError, IndexError):
            major = 0
        if 0 < major < 11:
            warnings.append("macOS %s is older than any PySide6 that runs this "
                            "window. The window needs macOS 11 or newer."
                            % release)
        elif 0 < major < 13:
            pin = "6.9.3" if major == 12 else "6.7.3"
            warnings.append(
                'macOS %s cannot take the current PySide6, which is built for 13 '
                'and newer.\n           Install the last one for this system '
                'first:  pip install "PySide6==%s"' % (release, pin))

    missing = []
    for group, items in (("required", NEEDED), ("optional", OPTIONAL)):
        print("  %s" % group)
        for module, package, why in items:
            try:
                importlib.import_module(module)
                mark = "[ ok ]"
            except ImportError:
                if group == "optional" and no_wheel(package):
                    # Not offered below: pip would try to compile it and fail.
                    mark = "[ -- ]"
                    why += ". No pre-built package for this machine; not needed"
                else:
                    mark = "[    ]" if group == "optional" else "[FAIL]"
                    missing.append((package, group))
            print("    %s %-12s %s" % (mark, package, why))
        print()
    if missing:
        print("  install what is missing with:")
        print("      pip install " + " ".join(p for p, _ in missing))
    else:
        print("  everything is here.")
    for text in warnings:
        print()
        print("  note:  %s" % text)
    print()
    return 1 if any(g == "required" for _, g in missing) else 0


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if "--check" in argv:
        return report()
    for module, package, _why in NEEDED:
        try:
            __import__(module)
        except ImportError:
            raise SystemExit(
                "\n[ERR] %s is not installed, so the window cannot open.\n"
                "      Run:  pip install %s\n"
                "      Or:   python sensor_studio.py --check   to see the whole list.\n"
                % (package, package))
    from PySide6 import QtWidgets

    from studio import theme
    from studio.shell import Studio

    app = QtWidgets.QApplication(argv)
    app.setApplicationName("MINT Sensor Studio")
    theme.apply(app, dark="--dark" in argv)

    win = Studio()
    win.show()
    return app.exec()
