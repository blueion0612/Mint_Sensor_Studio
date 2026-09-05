# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
sensor_studio.py
Start here, from a checkout, without installing anything:

    python sensor_studio.py              open the window
    python sensor_studio.py --dark       the same window in the dark palette
    python sensor_studio.py --check      say what is installed, and stop

The implementation is studio/app.py. After `pip install -e .` the same thing is
the `sensor-studio` command, or `python -m studio`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from studio.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
