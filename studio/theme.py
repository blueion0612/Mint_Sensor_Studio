# Copyright (c) 2026 Yuhyeon Lee
# SPDX-License-Identifier: MIT
"""
theme.py
One palette, one type scale, one set of spacings.

Everything visual is decided here so that no page invents its own. Two rules
hold the look together:

    the data is the brightest thing on screen
    the accent color is used for exactly one thing at a time

Colors are named for what they are for, not for what they look like, so that
changing the palette does not mean rereading every page. There are two
palettes, light and dark, chosen once at start-up (`--dark`); the pages read
the names and never the values.
"""

import sys

import pyqtgraph as pg

# --- the two palettes ---------------------------------------------------
LIGHT = dict(
    ACCENT="#4e8f57",          # MINT Lab green, darkened for text contrast
    ACCENT_SOFT="#e8f2e9",
    ACCENT_LINE="#8fc09a",
    ACCENT_HOVER="#437c4b",
    BG="#f4f5f6",              # window
    SURFACE="#ffffff",         # panels and plots
    LINE="#dcdee1",            # borders
    LINE_SOFT="#eceef0",       # grid
    INK="#1c1f23",             # primary text
    INK_SOFT="#5b6169",        # secondary text
    INK_FAINT="#8b9198",       # labels, units
    TRUTH="#8b9198",           # references, targets, the ideal
    LEGEND_BG=(255, 255, 255, 220),
)
DARK = dict(
    ACCENT="#6cbf78",
    ACCENT_SOFT="#22332a",
    ACCENT_LINE="#3d7a48",
    ACCENT_HOVER="#7fcf8a",
    BG="#141719",
    SURFACE="#1d2226",
    LINE="#2f363c",
    LINE_SOFT="#262c31",
    INK="#e6e8eb",
    INK_SOFT="#aab1b8",
    INK_FAINT="#7d858d",
    TRUTH="#7d858d",
    LEGEND_BG=(29, 34, 38, 220),
)
IS_DARK = False
globals().update(LIGHT)

# --- meaning -----------------------------------------------------------
X_AXIS, Y_AXIS, Z_AXIS = "#d4453c", "#2e9e4f", "#2f6fd0"
WARN = "#c8501e"
GOOD = "#2e7d4f"
GUILTY = "#d4453c"          # the term in an equation that causes the trouble
RECORD = "#d4453c"

SERIES = ("#d4453c", "#e08a1e", "#2f6fd0", "#7a4fbf", "#2e9e4f")

# one color per kind of sensor, worn as a badge in the toolbar and on the rail
MODALITY = {"imu": "#4e8f57", "emg": "#e08a1e", "ppg": "#d4453c", "generic": "#5b6169"}

# --- type --------------------------------------------------------------
# Named per platform rather than left to chance. Qt will substitute something
# if a face is absent, but its guess for a missing "Segoe UI" on a Mac is not
# the face a Mac user expects to read, and the substitution changes every
# metric on the page.
if sys.platform == "darwin":
    UI_FONT, MONO_FONT = "SF Pro Text", "Menlo"
elif sys.platform.startswith("linux"):
    UI_FONT, MONO_FONT = "DejaVu Sans", "DejaVu Sans Mono"
else:
    UI_FONT, MONO_FONT = "Segoe UI", "Consolas"

SIZE_TITLE = 15
SIZE_BODY = 11
SIZE_SMALL = 10
SIZE_READOUT = 22

# --- spacing, on an 8 px grid -----------------------------------------
GAP = 8
PAD = 16
RAIL_W = 236
EXPLAIN_H = 210

# The smallest window the layout is designed to hold, which is what fits on a
# 1366 x 768 laptop with its taskbar. Everything below is a minimum, not a
# size: the panels grow with the window and never shrink past these.
MIN_W, MIN_H = 1000, 620
WANT_W, WANT_H = 1440, 900
SIDE_W = 330                # the readout column on a page


def apply(app, dark=False):
    """Style the whole application once, at start-up, in one of the two palettes."""
    global IS_DARK
    IS_DARK = bool(dark)
    globals().update(DARK if dark else LIGHT)
    pg.setConfigOptions(antialias=True, background=SURFACE, foreground=INK_SOFT,
                        imageAxisOrder="row-major")
    app.setStyleSheet(sheet())


def plot(widget, x_label="", y_label="", x_unit="", y_unit=""):
    """The house style for a plot. Called instead of setting this up by hand."""
    widget.setBackground(SURFACE)
    # A layout will happily give a plot two pixels if something else in the
    # same card asks for room. Below this height the axis labels collide with
    # the data. It is a floor for the smallest window, not a size: Qt cannot
    # shrink a widget below its minimum, so a floor higher than the window
    # can hold pushes the plot over the controls under it, and two plots at
    # 160 with their controls did exactly that at 640 pixels.
    widget.setMinimumHeight(90)
    pi = widget.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.12)
    pi.getViewBox().setDefaultPadding(0.02)
    for side in ("left", "bottom"):
        ax = pi.getAxis(side)
        ax.setPen(pg.mkPen(LINE, width=1))
        ax.setTextPen(pg.mkPen(INK_FAINT))
        ax.setStyle(tickFont=_tick_font(), tickTextOffset=6)
    for side in ("top", "right"):
        pi.showAxis(side, False)
    if y_label:
        pi.setLabel("left", y_label, units=y_unit or None,
                    **{"color": INK_SOFT, "font-size": "%dpt" % SIZE_SMALL})
    if x_label:
        pi.setLabel("bottom", x_label, units=x_unit or None,
                    **{"color": INK_SOFT, "font-size": "%dpt" % SIZE_SMALL})
    return pi


def title(pi, text):
    """A plot's title, in the house style, top left."""
    pi.setTitle(text, color=INK_SOFT, size="%dpt" % SIZE_SMALL, justify="left")


def axis_label(pi, side, text, unit=None):
    pi.setLabel(side, text, units=unit or None,
                **{"color": INK_SOFT, "font-size": "%dpt" % SIZE_SMALL})


_TICK_FONT = None


def _tick_font():
    global _TICK_FONT
    if _TICK_FONT is None:
        from PySide6 import QtGui
        _TICK_FONT = QtGui.QFont(UI_FONT, SIZE_SMALL - 1)
    return _TICK_FONT


def pen(colour, width=1.6, dash=False):
    p = pg.mkPen(colour, width=width)
    if dash:
        p.setDashPattern([6, 5])
    return p


def legend(plot_item, offset=(-10, 10)):
    lg = plot_item.addLegend(offset=offset, labelTextSize="%dpt" % SIZE_SMALL,
                             labelTextColor=INK_SOFT, brush=pg.mkBrush(*LEGEND_BG),
                             pen=pg.mkPen(LINE))
    lg.setColumnCount(3)
    return lg


def fill(colour, alpha=40):
    """A translucent brush of a color, for the area under a curve."""
    c = pg.mkColor(colour)
    return pg.mkBrush(c.red(), c.green(), c.blue(), alpha)


SHEET_TEMPLATE = """
* {{ font-family: "{ui}"; font-size: {body}pt; color: {ink}; }}

QMainWindow, QWidget#Page, QDialog {{ background: {bg}; }}

QToolBar#TopBar {{
    background: {surface};
    border: 0px;
    border-bottom: 1px solid {line};
    padding: 6px {pad}px;
    spacing: {gap}px;
}}

QLabel#Product {{ font-size: {title}pt; font-weight: 600; color: {ink}; }}
QLabel#Tagline  {{ color: {faint}; }}
QLabel#Caption  {{ color: {faint}; font-size: {small}pt; }}
QLabel#Readout  {{ font-family: "{mono}"; color: {ink}; }}
QLabel#Section  {{ color: {faint}; font-size: {small}pt;
                   font-weight: 600; letter-spacing: 1px; }}
QLabel#Badge    {{ color: #ffffff; font-size: {small}pt; font-weight: 700;
                   letter-spacing: 1px; padding: 2px 8px; border-radius: 4px; }}
QLabel#Tip      {{ color: {accent}; }}

QTreeWidget#Rail {{
    background: {surface};
    border: 0px;
    border-right: 1px solid {line};
    outline: 0;
    padding: {gap}px 0px;
}}
QTreeWidget#Rail::item {{
    padding: 8px 10px;
    border-left: 3px solid transparent;
    color: {inksoft};
}}
QTreeWidget#Rail::item:has-children {{
    color: {faint}; font-size: {small}pt; font-weight: 600; letter-spacing: 1px;
    padding-top: 12px;
}}
QTreeWidget#Rail::item:selected {{
    background: {accent_soft};
    border-left: 3px solid {accent};
    color: {ink};
    font-weight: 600;
}}
QTreeWidget#Rail::item:hover:!selected {{ background: {bg}; }}
QTreeWidget#Rail::item:disabled {{ color: {faint}; background: transparent; }}
QTreeWidget#Rail::branch {{ background: {surface}; border-image: none; image: none; }}

QFrame#Card {{
    background: {surface};
    border: 1px solid {line};
    border-radius: 6px;
}}

QFrame#Explain {{
    background: {surface};
    border: 0px;
    border-top: 1px solid {line};
}}

QPushButton {{
    background: {surface};
    border: 1px solid {line};
    border-radius: 5px;
    padding: 6px 14px;
    color: {ink};
}}
QPushButton:hover {{ border-color: {accent_line}; }}
QPushButton:pressed {{ background: {bg}; }}
QPushButton:disabled {{ color: {faint}; border-color: {linesoft}; }}
QPushButton#Primary {{
    background: {accent}; border-color: {accent}; color: #ffffff; font-weight: 600;
}}
QPushButton#Primary:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton:checked {{
    background: {accent_soft}; border-color: {accent}; color: {ink}; font-weight: 600;
}}
QPushButton#Record {{ color: {record}; border-color: {line}; font-weight: 600; }}
QPushButton#Record:checked {{ background: {record}; border-color: {record}; color: #ffffff; }}

QComboBox {{
    background: {surface}; border: 1px solid {line}; border-radius: 5px;
    padding: 5px 10px; min-width: 150px;
}}
QComboBox:hover {{ border-color: {accent_line}; }}
QComboBox::drop-down {{ border: 0px; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {surface}; border: 1px solid {line};
    selection-background-color: {accent_soft}; selection-color: {ink};
    outline: 0;
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {surface}; border: 1px solid {line}; border-radius: 5px; padding: 5px 8px;
}}

QTabWidget::pane {{ border: 1px solid {line}; border-radius: 6px; background: {surface}; }}
QTabBar::tab {{
    background: {bg}; color: {inksoft}; border: 1px solid {line}; border-bottom: 0px;
    padding: 7px 16px; margin-right: 2px;
    border-top-left-radius: 5px; border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{ background: {surface}; color: {ink}; font-weight: 600; }}

QSlider::groove:horizontal {{
    height: 4px; background: {linesoft}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {accent_line}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {surface}; border: 2px solid {accent};
    width: 12px; height: 12px; margin: -6px 0; border-radius: 8px;
}}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {line};
    border-radius: 3px; background: {surface};
}}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QRadioButton::indicator {{
    width: 14px; height: 14px; border: 1px solid {line}; border-radius: 8px;
    background: {surface};
}}
QRadioButton::indicator:checked {{ background: {accent}; border: 3px solid {surface};
                                   outline: 1px solid {accent}; }}

QProgressBar {{ background: {linesoft}; border: 0px; border-radius: 3px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}

QScrollArea {{ background: transparent; border: 0px; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

QStatusBar {{ background: {surface}; border-top: 1px solid {line}; color: {faint}; }}
QStatusBar::item {{ border: 0px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {line}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
QToolTip {{
    background: {ink}; color: {surface}; border: 0px; padding: 5px 8px; border-radius: 4px;
}}
"""


def sheet():
    return SHEET_TEMPLATE.format(
        ui=UI_FONT, mono=MONO_FONT, body=SIZE_BODY, title=SIZE_TITLE, small=SIZE_SMALL,
        bg=BG, surface=SURFACE, line=LINE, linesoft=LINE_SOFT, ink=INK,
        inksoft=INK_SOFT, faint=INK_FAINT, accent=ACCENT, accent_soft=ACCENT_SOFT,
        accent_line=ACCENT_LINE, accent_hover=ACCENT_HOVER, record=RECORD,
        pad=PAD, gap=GAP)
