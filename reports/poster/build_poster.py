"""Build the A1 poster (594 x 841 mm) on the department template.

    python reports/poster/build_poster.py          # writes poster_draft.pptx + placements.json
    powershell -File reports/poster/finish_poster.ps1   # PowerPoint inserts the SVG figures, checks
                                                        # text fit, saves the PPTX and exports the PDF

Every number in the text is read from results/ by numbers() below, never typed in.
Text heights are measured with the real font files, so the layout is computed, not guessed.
"""

import copy
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import HALF, ROUTE_PICK, SIZES  # noqa: E402  (slot sizes = figure sizes)
ROOT = HERE.parents[1]
RES = ROOT / "results"
FIG = HERE / "figures"

# ------------------------------------------------------------------ style
BLUE = "0070C0"        # headings and takeaways, as in the reference poster
NAVY = "1F3973"
INK = "111111"
PALE = "E8F1FA"
FOOT = "F1AA84"
F_TITLE, F_HEAD, F_BODY, F_EQ, F_HINDI = "Times New Roman", "Trebuchet MS", "Calibri", "Cambria", "Nirmala UI"
S_TITLE, S_SUB, S_HDR, S_AUTH, S_HEAD, S_SUBHEAD, S_BODY, S_CAP, S_TAKE, S_SMALL = 40, 24, 22, 28, 28, 20, 20, 16, 17, 15
S_GA_T, S_GA_S, S_TILE_N, S_TILE_L, S_EQ_L, S_EQ = 20, 17, 26, 16, 18, 22
LNSP = 0.95            # line spacing used for every text block

PAGE_W, PAGE_H = 594.12, 840.98
MARGIN = 15.0
GUTTER = 15.0
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
COL_X = (MARGIN, MARGIN + COL_W + GUTTER)
HEADER_H = 60.0
FOOTER_H = 16.0
TOP = HEADER_H + 5.0
BOTTOM = PAGE_H - FOOTER_H - 4.0

FONT_FILES = {
    ("Calibri", False, False): "calibri.ttf", ("Calibri", True, False): "calibrib.ttf",
    ("Calibri", False, True): "calibrii.ttf", ("Calibri", True, True): "calibriz.ttf",
    ("Trebuchet MS", True, False): "trebucbd.ttf", ("Trebuchet MS", False, False): "trebuc.ttf",
    ("Times New Roman", False, False): "times.ttf", ("Times New Roman", True, False): "timesbd.ttf",
    ("Times New Roman", False, True): "timesi.ttf",
    ("Cambria", False, False): "cambria.ttc", ("Cambria", False, True): "cambriai.ttf",
    ("Cambria", True, False): "cambriab.ttf",
}
LINE_FACTOR = {"Calibri": 1.2207, "Trebuchet MS": 1.1621, "Times New Roman": 1.1499, "Cambria": 1.1724,
               "Nirmala UI": 1.33}
_fcache = {}


def mm(v):
    return Emu(int(round(v * 36000)))


def _font(face, bold, italic, size):
    key = (face, bold, italic)
    if key not in FONT_FILES:
        key = (face, bold, False) if (face, bold, False) in FONT_FILES else (face, False, False)
    k = key + (size,)
    if k not in _fcache:
        _fcache[k] = ImageFont.truetype(f"C:/Windows/Fonts/{FONT_FILES[key]}", size * 20)
    return _fcache[k]


def width_mm(text, face, size, bold=False, italic=False):
    return _font(face, bold, italic, size).getlength(text) / 20 * 25.4 / 72


def line_mm(face, size, spacing=LNSP):
    return LINE_FACTOR[face] * size * spacing * 25.4 / 72


# ------------------------------------------------------------------ numbers from results/
def numbers():
    n = {}
    rp = pd.read_csv(RES / "s01_route_pairs.csv").set_index("grade")
    n["fc4140"], n["ac4140"] = rp.loc["AISI 4140", "UTS_MPa_FC"], rp.loc["AISI 4140", "UTS_MPa_AC"]
    b = pd.read_csv(RES / "s04_model_benchmark.csv").set_index(["target", "model"])
    n["an_uts"] = b.loc[("UTS_MPa", "gated_ensemble"), "RMSE_inAD"]
    n["an_uts_txt"] = b.loc[("UTS_MPa", "physics_lit"), "RMSE_inAD"]
    n["an_cov"] = pd.read_csv(RES / "s04_conformal.csv").set_index("target").loc["UTS_MPa", "LOGO_coverage_90"]
    q = pd.read_csv(RES / "s09_qt_benchmark.csv").set_index(["model", "target"])
    n["qt_uts"], n["qt_uts_txt"] = q.loc[("qt_ensemble", "UTS_MPa"), "RMSE"], q.loc[("qt_physics", "UTS_MPa"), "RMSE"]
    n["qt_cov"] = q.loc[("qt_ensemble", "UTS_MPa"), "coverage_nested"]
    qo = pd.read_csv(RES / "s09_qt_oof.csv")
    n["qt_bars"], n["qt_grades"] = len(qo), qo.grade.nunique()
    c = pd.read_csv(RES / "s05_category_summary.csv").set_index("category")
    g = c.UTS_MPa_delta_median
    n["gain_low"], n["gain_med"], n["gain_high"] = g["Low-carbon steel"], g["Medium-carbon steel"], g["High-carbon steel"]
    n["gain_crmo"] = g["Mo / Cr-Mo steel"]
    n["el_high"] = c.El_pct_delta_median["High-carbon steel"]
    j = pd.read_csv(RES / "s08_kinetic_calibration_cv_metrics.csv")
    j = j[j.distances == "all distances"].set_index("kinetics")
    n["jom_pub"], n["jom_cal"] = j.loc["HRC_published", "RMSE_HRC"], j.loc["HRC_calibrated_cv", "RMSE_HRC"]
    n["jom_pts"] = int(j.loc["HRC_calibrated_cv", "n"])
    n["jom_grades"] = pd.read_csv(RES / "s08_kinetic_calibration_cv.csv").grade.nunique()
    bs = pd.read_csv(RES / "s06_bar_sizes_metrics.csv")
    r = bs[bs.check.str.startswith("bar-size effect")].iloc[0]
    n["bs_n"], n["bs_sign"], n["bs_pred"], n["bs_meas"] = int(r.n), r.sign_agreement, r.mean_predicted, r.mean_measured
    hc = pd.read_csv(RES / "s09_hardenability_check.csv")
    n["di_ratio"] = float(np.median(hc.DI_model_calibrated_mm / hc.DI_A255_mm))
    ft = pd.read_csv(RES / "s09_five_treatments.csv")
    f = ft[ft.grade.astype(str) == "4140"].set_index("treatment")
    n["ft"] = f
    n["ft_min"] = f.UTS_MPa_pred.min()
    n["ft_max"] = f.UTS_MPa_pred.max()
    ds = pd.read_csv(RES / "s01_dataset_summary.csv").set_index("dataset")
    n["hb_grades"] = int(ds.loc["properties_handbook", "steels_or_grades"])
    n["members"] = int(ds.loc["structural_external", "rows"])
    lit = pd.read_csv(ROOT / "data/raw/literature_compilation.csv")
    n["lit_rows"] = len(lit)
    n["cct_pts"] = int(pd.read_csv(RES / "s03_cct_atlas_summary.csv").set_index("subset").loc["all points", "n"])
    n["cct_steels"] = int(pd.read_csv(RES / "s03_cct_atlas_summary.csv").set_index("subset").loc["all points", "steels"])
    bars = pd.read_csv(RES / "s06_bar_sizes_predictions.csv")
    n["an_bars"] = len(bars[bars.route.isin(["FC", "AC"])]) if "route" in bars else None
    tc = pd.read_csv(RES / "s09_tempering_curves.csv")
    t = tc[tc.grade.astype(str) == "4140"]
    n["routes"] = (rp.UTS_MPa_AC - rp.UTS_MPa_FC).to_dict()
    n["cool"] = pd.read_csv(RES / "s03_cooling_paths.csv").set_index("path").rate_700C_K_s.to_dict()
    n["temp_max"], n["temp_min"] = t.UTS_MPa_pred.max(), t.UTS_MPa_pred.min()
    m = qo[(qo.grade.astype(str) == "4140") & (qo.section_mm == 25) & (qo.quench_medium == "oil")]
    n["temp_meas_lo"], n["temp_meas_hi"] = m.temper_T_C.min(), m.temper_T_C.max()
    cct = json.loads((HERE / "data/cct_4140.json").read_text())
    c45 = json.loads((HERE / "data/cct_1045.json").read_text())["routes"]
    n["cct45_same"] = all(abs(c45["FC"][k] - c45["AC"][k]) < 0.005 for k in "FPBM") and c45["AC"]["B"] < 0.01
    n["cct_ac_B"] = cct["routes"]["AC"]["B"]
    n["cct_fc_F"], n["cct_fc_P"] = cct["routes"]["FC"]["F"], cct["routes"]["FC"]["P"]
    return n


# ------------------------------------------------------------------ low-level pptx helpers
def _set_bodypr(tf, l=0, t=0, r=0, b=0, anchor=MSO_ANCHOR.TOP, wrap=True):
    tf.word_wrap = wrap
    tf.auto_size = None
    tf.margin_left, tf.margin_top, tf.margin_right, tf.margin_bottom = mm(l), mm(t), mm(r), mm(b)
    tf.vertical_anchor = anchor
    bp = tf._txBody.find(qn("a:bodyPr"))
    for child in list(bp):
        bp.remove(child)
    etree.SubElement(bp, qn("a:noAutofit"))


def _ppr(p, align=PP_ALIGN.LEFT, spacing=LNSP, after_pt=0.0, before_pt=0.0, bullet=None, indent_mm=0.0):
    p.alignment = align
    pPr = p._p.get_or_add_pPr()
    for tag in ("a:lnSpc", "a:spcBef", "a:spcAft", "a:buClr", "a:buFont", "a:buChar", "a:buNone"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    ln = etree.SubElement(pPr, qn("a:lnSpc"))
    etree.SubElement(ln, qn("a:spcPct"), val=str(int(spacing * 100000)))
    sb = etree.SubElement(pPr, qn("a:spcBef"))
    etree.SubElement(sb, qn("a:spcPts"), val=str(int(before_pt * 100)))
    sa = etree.SubElement(pPr, qn("a:spcAft"))
    etree.SubElement(sa, qn("a:spcPts"), val=str(int(after_pt * 100)))
    if bullet:
        pPr.set("marL", str(int(indent_mm * 36000)))
        pPr.set("indent", str(-int(indent_mm * 36000)))
        bc = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bc, qn("a:srgbClr"), val=BLUE)
        etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
        etree.SubElement(pPr, qn("a:buChar"), char=bullet)
    else:
        etree.SubElement(pPr, qn("a:buNone"))


UNIT_RE = re.compile(r"(\d) (%|MPa|HRC|HV|HB|mm|°C|K/s|h(?![A-Za-z]))")


def nbsp(text):
    return UNIT_RE.sub(lambda m: m.group(1) + " " + m.group(2), text)


def _run(p, text, face, size, bold=False, italic=False, color=INK, baseline=None, cs_face=None):
    r = p.add_run()
    r.text = nbsp(text)
    f = r.font
    f.name, f.size, f.bold, f.italic = face, Pt(size), bold, italic
    f.color.rgb = RGBColor.from_string(color)
    rPr = r._r.get_or_add_rPr()
    if baseline:
        rPr.set("baseline", str(baseline))
    if cs_face:
        for tag in ("a:ea", "a:cs"):
            e = rPr.find(qn(tag))
            if e is None:
                e = etree.SubElement(rPr, qn(tag))
            e.set("typeface", cs_face)
    return r


def textbox(slide, x, y, w, h, name):
    tb = slide.shapes.add_textbox(mm(x), mm(y), mm(w), mm(h))
    tb.name = name
    _set_bodypr(tb.text_frame)
    return tb


def rect(slide, x, y, w, h, fill=None, line=None, line_pt=0.0, shape=MSO_SHAPE.RECTANGLE, name=None, radius=None):
    s = slide.shapes.add_shape(shape, mm(x), mm(y), mm(w), mm(h))
    if name:
        s.name = name
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = RGBColor.from_string(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = RGBColor.from_string(line)
        s.line.width = Pt(line_pt)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if s.has_text_frame:
        _set_bodypr(s.text_frame)
    return s


# ------------------------------------------------------------------ rich text: **bold**, [[blue bold]]
def parse_rich(text):
    """'**x**' bold, '[[x]]' bold blue. Returns [(text, bold, color)]."""
    out, i, bold, blue, buf = [], 0, False, False, ""
    while i < len(text):
        if text.startswith("**", i):
            if buf:
                out.append((buf, bold or blue, BLUE if blue else INK))
            buf, bold, i = "", not bold, i + 2
        elif text.startswith("[[", i) or text.startswith("]]", i):
            if buf:
                out.append((buf, bold or blue, BLUE if blue else INK))
            buf, blue, i = "", text.startswith("[[", i), i + 2
        else:
            buf += text[i]
            i += 1
    if buf:
        out.append((buf, bold or blue, BLUE if blue else INK))
    return out


def wrap_count(runs, width, face, size):
    """Number of lines a paragraph of rich runs takes in `width` mm (greedy word wrap)."""
    words = []
    runs = [(nbsp(t), b, c) for t, b, c in runs]
    for t, bold, _ in runs:
        for k, wtxt in enumerate(t.split(" ")):
            words.append((wtxt, bold, k > 0 or (t.startswith(" "))))
    space = width_mm(" ", face, size)
    lines, cur = 1, 0.0
    for wtxt, bold, _ in words:
        if not wtxt:
            continue
        wl = width_mm(wtxt, face, size, bold)
        if cur == 0:
            cur = wl
        elif cur + space + wl <= width:
            cur += space + wl
        else:
            lines += 1
            cur = wl
    return lines


def paragraphs(slide, x, y, w, items, size=S_BODY, face=F_BODY, bullet="•", indent=5.0, after_pt=3.0,
               align=PP_ALIGN.LEFT, name="text", spacing=LNSP, slack=1.5):
    """Bulleted (or plain) rich paragraphs; returns the height used in mm."""
    usable = w - (indent if bullet else 0) - 0.5
    lh = line_mm(face, size, spacing)
    nlines = [wrap_count(parse_rich(it), usable, face, size) for it in items]
    h = sum(nlines) * lh + (len(items) - 1) * after_pt * 25.4 / 72 + slack
    tb = textbox(slide, x, y, w, h, name)
    tf = tb.text_frame
    for k, it in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        _ppr(p, align=align, spacing=spacing, after_pt=after_pt if k < len(items) - 1 else 0,
             bullet=bullet, indent_mm=indent)
        for t, bold, col in parse_rich(it):
            _run(p, t, face, size, bold=bold, color=col)
    return h


EXTRA = {0: 0.0, 1: 0.0}   # extra space before each heading, set by a second layout pass
EXTRA_ROW = 0.0            # extra space between rows of result figures


def figure_block(slide, x, y, w, h, name, cap, take, placements, cap_lines=None, take_lines=None):
    """Figure slot + centred caption + bold blue takeaway; returns the height used (mm)."""
    placements.append({"kind": "svg", "name": name, "x": x, "y": y, "w": w, "h": h})
    rect(slide, x, y, w, h, line="BBBBBB", line_pt=0.5, name=f"figure slot {name}")
    cl = cap_lines or wrap_count(parse_rich(cap), w - 1, F_BODY, S_CAP)
    tl = take_lines or wrap_count([(take, True, BLUE)], w - 1, F_BODY, S_TAKE)
    cy = y + h + 1.2
    tb = textbox(slide, x, cy, w, cl * line_mm(F_BODY, S_CAP) + 1, f"caption {name}")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, align=PP_ALIGN.CENTER)
    if cap.startswith("Fig "):
        fig_no, rest = cap.split(":", 1)
        _run(p, fig_no + ":", F_BODY, S_CAP, bold=True)
        _run(p, rest, F_BODY, S_CAP)
    else:
        _run(p, cap, F_BODY, S_CAP)
    ty = cy + cl * line_mm(F_BODY, S_CAP) + 0.6
    tb = textbox(slide, x, ty, w, tl * line_mm(F_BODY, S_TAKE) + 1, f"takeaway {name}")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, align=PP_ALIGN.CENTER)
    _run(p, take, F_BODY, S_TAKE, bold=True, color=BLUE)
    return h + 1.2 + cl * line_mm(F_BODY, S_CAP) + 0.6 + tl * line_mm(F_BODY, S_TAKE) + 1.0


def tile_strip(slide, x, y, w, items, name, gap=3.0):
    """A row of pale rounded tiles: big blue number over a small label. Returns the height (mm)."""
    tw = (w - (len(items) - 1) * gap) / len(items)
    lab_lines = max(wrap_count([(lab, False, INK)], tw - 3, F_BODY, S_TILE_L) for _, lab in items)
    th = 2.0 + line_mm(F_BODY, S_TILE_N, 0.85) + lab_lines * line_mm(F_BODY, S_TILE_L, 0.9) + 2.0
    for k, (num, lab) in enumerate(items):
        tx = x + k * (tw + gap)
        rect(slide, tx, y, tw, th, fill=PALE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"{name} tile {k + 1}",
             radius=0.08)
        tb = textbox(slide, tx + 1.5, y + 1.5, tw - 3, th - 2.0, f"{name} tile text {k + 1}")
        tf = tb.text_frame
        p = tf.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.85)
        _run(p, num, F_BODY, S_TILE_N, bold=True, color=BLUE)
        p = tf.add_paragraph()
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
        _run(p, lab, F_BODY, S_TILE_L, color=INK)
    return th


def heading(slide, col, y, text, first=False):
    before = 0.0 if first else 4.0 + EXTRA[col]
    h = line_mm(F_HEAD, S_HEAD, 0.9) + 1.0
    tb = textbox(slide, COL_X[col], y + before, COL_W, h, f"heading: {text}")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
    _run(p, text, F_HEAD, S_HEAD, bold=True, color=BLUE)
    return before + h + 1.0


def subheading(slide, x, y, w, text):
    h = line_mm(F_BODY, S_SUBHEAD, 0.9) + 0.5
    tb = textbox(slide, x, y, w, h, f"subheading: {text}")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, spacing=0.9)
    _run(p, text, F_BODY, S_SUBHEAD, bold=True, color=BLUE)
    return h + 1.0


# ------------------------------------------------------------------ equations: _{sub} ^{sup} ~italic~
def eq_runs(src):
    runs, i, ital = [], 0, False
    while i < len(src):
        ch = src[i]
        if ch in "_^" and i + 1 < len(src) and src[i + 1] == "{":
            j = src.index("}", i)
            inner = src[i + 2:j]
            parts = inner.split("~")
            for k, part in enumerate(parts):
                if part:
                    runs.append((part, (k % 2 == 1), -25000 if ch == "_" else 30000))
            i = j + 1
        elif ch == "~":
            ital = not ital
            i += 1
        else:
            j = i
            while j < len(src) and src[j] not in "_^~":
                j += 1
            runs.append((src[i:j], ital, None))
            i = j
    return runs


def equation_row(slide, x, y, w, label_w, label, src, size=S_EQ):
    h = line_mm(F_EQ, size, 1.0) + 1.2
    tb = textbox(slide, x, y, label_w, h, f"eq label: {label}")
    tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tb.text_frame.paragraphs[0]
    _ppr(p, spacing=0.9)
    _run(p, label, F_BODY, S_EQ_L, bold=True, color=NAVY)
    tb = textbox(slide, x + label_w, y, w - label_w, h, f"equation: {label}")
    tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tb.text_frame.paragraphs[0]
    _ppr(p, spacing=1.0)
    for t, it, base in eq_runs(src):
        _run(p, t, F_EQ, size, italic=it, baseline=base)
    return h


# ------------------------------------------------------------------ template frame
def prepare_template(prs):
    slide = prs.slides[0]
    grad, logo = None, None
    for shp in list(slide.shapes):
        if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
            for sub in shp.shapes:
                if sub.shape_type == MSO_SHAPE_TYPE.GROUP:
                    for s2 in sub.shapes:
                        if s2.name == "Rectangle 3":
                            grad = copy.deepcopy(s2._element.spPr.find(qn("a:gradFill")))
            shp._element.getparent().remove(shp._element)
        elif shp.shape_type == MSO_SHAPE_TYPE.PICTURE:
            logo = shp
    return slide, grad, logo


def build(out_pptx, placements_json):
    N = numbers()
    prs = Presentation(str(HERE / "Template.pptx"))
    slide, grad, logo = prepare_template(prs)
    spTree = slide.shapes._spTree
    placements = []

    # header band with the template's own gradient
    band = rect(slide, 0, 0, PAGE_W, HEADER_H, fill=PALE, name="header band")
    spPr = band._element.spPr
    for tag in ("a:solidFill", "a:noFill", "a:gradFill"):
        for e in spPr.findall(qn(tag)):
            spPr.remove(e)
    spPr.insert(2, grad)
    spTree.remove(band._element)
    spTree.insert(2, band._element)
    # logo from the template (vector)
    logo.left, logo.top, logo.width, logo.height = mm(14), mm(6), mm(48), mm(48)
    spTree.remove(logo._element)
    spTree.append(logo._element)

    # header text
    tb = textbox(slide, MARGIN, 4.5, PAGE_W - 2 * MARGIN, HEADER_H - 6, "title block")
    tf = tb.text_frame
    lines = [("Furnace or Air Cooling? Predicting What Changes in a Steel", F_TITLE, S_TITLE, True, False, INK,
              0.95),
             ("Physics-informed machine learning for five heat treatments of steel", F_TITLE, S_SUB, False, True,
              NAVY, 1.0),
             ("B.Tech Project (MTP-400A), Autumn 2026   |   Department of Metallurgical and Materials "
              "Engineering, Indian Institute of Technology Roorkee", F_TITLE, S_HDR, False, False, INK, 1.0),
             ("Shibasish Banerjee, Aviral Singh   |   Supervisor: Dr. Anish Karmakar", F_TITLE, S_AUTH, False,
              False, INK, 1.0)]
    for k, (t, face, size, bold, ital, col, sp) in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        _ppr(p, align=PP_ALIGN.CENTER, spacing=sp, before_pt=3 if k in (1, 3) else 0)
        _run(p, t, face, size, bold=bold, italic=ital, color=col)
    hdr_need = sum(line_mm(f, s, sp) for _, f, s, *_, sp in lines) + 6 * 25.4 / 72
    print(f"header text needs {hdr_need:.1f} mm of {HEADER_H - 6:.1f}")

    # page border, column divider, footer
    rect(slide, 0.6, 0.6, PAGE_W - 1.2, PAGE_H - 1.2, line="0E2841", line_pt=2.5, name="page border")
    div = slide.shapes.add_connector(1, mm(PAGE_W / 2), mm(TOP), mm(PAGE_W / 2), mm(BOTTOM))
    div.line.color.rgb = RGBColor.from_string("156082")
    div.line.width = Pt(1.5)
    div.name = "column divider"
    fy = PAGE_H - FOOTER_H
    rect(slide, 0, fy, PAGE_W, FOOTER_H, fill=FOOT, name="footer band")
    for k, (t, face, cs) in enumerate((("श्रमं विना न किमपि साध्यम्!", F_HINDI, F_HINDI),
                                       ("Nothing can be achieved without hard work!", F_BODY, None))):
        tb = textbox(slide, k * PAGE_W / 2, fy, PAGE_W / 2, FOOTER_H, f"footer {k}")
        tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tb.text_frame.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=1.0)
        _run(p, t, face, S_BODY, bold=(k == 1), color=INK, cs_face=cs)

    # ============================== LEFT COLUMN
    x, w, y = COL_X[0], COL_W, TOP
    y += heading(slide, 0, y, "Introduction and Motivation", first=True)
    y += paragraphs(slide, x, y, w, [
        f"**Same steel, very different properties:** AISI 4140 is {N['fc4140']:.0f} MPa annealed "
        f"but {N['ac4140']:.0f} MPa normalised",
        "Handbooks give a few conditions per grade; open tools with honest uncertainty are lacking",
        "Pure ML ignores processing; pure physics is not accurate enough",
        "**Our supervisor's question:** furnace or air cooling – what changes, how much, and why?",
    ], name="intro")
    y += 2.0
    fw, fh = SIZES["routes"]
    gains = [N["routes"][g] for g, _ in ROUTE_PICK]
    y += figure_block(slide, x, y, fw, fh, "f0_handbook_routes",
                      "Fig 1: Handbook tensile strength of the same grades, annealed vs normalised",
                      f"The cooling route alone adds +{min(gains):.0f} to +{max(gains):.0f} MPa",
                      placements)

    # graphical abstract
    y += heading(slide, 0, y, "Graphical Abstract")
    steps = [("Inputs", "composition, heat treatment, bar size"),
             ("Transformation", "heat flow + Li kinetics"),
             ("Microstructure", "ferrite, pearlite, bainite, martensite"),
             ("Physics strength", "Gladman, Maynier, Hollomon–Jaffe"),
             ("GP ensemble", "learns what the physics misses"),
             ("Properties", "YS, UTS, El, HB with 90 % intervals")]
    aw = 9.0
    bw = (w - 2 * aw) / 3
    inner = bw - 5
    t_lines = max(wrap_count([(t, True, INK)], inner, F_BODY, S_GA_T) for t, _ in steps)
    s_lines = max(wrap_count([(s, False, INK)], inner, F_BODY, S_GA_S) for _, s in steps)
    bh = 4.5 + t_lines * line_mm(F_BODY, S_GA_T, 0.95) + s_lines * line_mm(F_BODY, S_GA_S, 0.95) + 2.5
    rowgap = 7.0
    pos = []
    for k in range(6):
        row, colk = divmod(k, 3)
        if row == 1:
            colk = 2 - colk
        pos.append((x + colk * (bw + aw), y + 2.5 + row * (bh + rowgap)))
    for k, ((t, s), (bx, by)) in enumerate(zip(steps, pos)):
        last = k == 5
        b = rect(slide, bx, by, bw, bh, fill=("FDE9D9" if last else PALE), line=BLUE, line_pt=1.5,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"GA box {k + 1}", radius=0.12)
        tb = textbox(slide, bx + 2.5, by + 4.0, bw - 5, bh - 4.5, f"GA text {k + 1}")
        tf = tb.text_frame
        p = tf.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.95)
        _run(p, t, F_BODY, S_GA_T, bold=True, color=NAVY)
        p = tf.add_paragraph()
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.95)
        _run(p, s, F_BODY, S_GA_S, color=INK)
        c = rect(slide, bx - 2.5, by - 2.5, 7.5, 7.5, fill=BLUE, shape=MSO_SHAPE.OVAL, name=f"GA number {k + 1}")
        tf = c.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=1.0)
        _run(p, str(k + 1), F_BODY, 15, bold=True, color="FFFFFF")
    for k in range(5):
        (ax0, ay0), (ax1, ay1) = pos[k], pos[k + 1]
        if k == 2:   # down arrow at the right
            a = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, mm(ax0 + bw / 2 - 3.5), mm(ay0 + bh + 1),
                                       mm(7), mm(rowgap - 2))
        elif k < 2:
            a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, mm(ax0 + bw + 1.2), mm(ay0 + bh / 2 - 3.5),
                                       mm(aw - 2.4), mm(7))
        else:
            a = slide.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, mm(ax1 + bw + 1.2), mm(ay0 + bh / 2 - 3.5),
                                       mm(aw - 2.4), mm(7))
        a.fill.solid()
        a.fill.fore_color.rgb = RGBColor.from_string(NAVY)
        a.line.fill.background()
        a.shadow.inherit = False
        a.name = f"GA arrow {k + 1}"
    y += 2.5 + 2 * bh + rowgap + 4
    y += subheading(slide, x, y, w, "Key results, on grades the models never saw")
    y += tile_strip(slide, x, y, w, [
        (f"{N['an_uts']:.0f} MPa", "UTS error, annealed / normalised"),
        (f"{N['qt_uts']:.0f} MPa", "UTS error, quenched + tempered"),
        (f"{100 * N['an_cov']:.1f} %", "inside the 90 % interval (A/N)"),
        (f"{N['jom_cal']:.1f} HRC", f"Jominy error (published: {N['jom_pub']:.1f})"),
    ], "key result") + 1.0

    # methodology: data strip + model chain
    y += heading(slide, 0, y, "Methodology")
    y += subheading(slide, x, y, w, "Open data (every row has its heat treatment)")
    tiles = [(f"{N['hb_grades']}", "handbook grades"),
             (f"{N['qt_bars']}", f"Q+T bars, {N['qt_grades']} grades"),
             (f"{N['jom_pts']}", "Jominy band points"),
             (f"{N['cct_pts']}", "CCT-atlas points"),
             (f"{N['lit_rows']}", "literature values")]
    y += tile_strip(slide, x, y, w, tiles, "data") + 3.0
    y += subheading(slide, x, y, w, "Model chain")
    cw, chh = SIZES["cooling"]
    furn = N["cool"]["furnace cooling"]
    h_chart = figure_block(slide, x + w - cw, y + 1.0, cw, chh, "f_methods_cooling_rates",
                           "Model cooling rate at 700 °C (quenches: bar centre)",
                           f"A 25 mm bar in air cools {N['cool']['still air, 25 mm bar'] / furn:.0f}× faster "
                           "than the furnace",
                           placements) + 1.0
    h_text = paragraphs(slide, x, y, w - cw - 6, [
        "**Thermodynamics:** Ae1, Ae3, Acm, lever rule",
        "**Heat flow:** furnace, bar in still air, radial quench",
        "**Kinetics:** Li et al.; bainite calibrated on Jominy bands (MCMC)",
        "**Quench + temper (Q+T):** Hollomon–Jaffe master curve",
        "**Properties:** Gaussian processes anchored on physics strength",
        "**Validation:** leave-one-grade-out; split-conformal 90 % intervals",
    ], name="methods", after_pt=4.0)
    y += max(h_text, h_chart)

    # key equations box
    y += 4.0
    eqs = [("Bainite rate", "~k~_{B} = ~f~_{B} 2^{0.29~G~} (~B~_{s} − ~T~)^{2} e^{−~Q~/~RT~} / BC"),
           ("Physics strength", "σ_{phys} = ~f~_{FP} σ_{Gladman} + ~f~_{BM} · 3.3 HV_{Maynier}"),
           ("Anchored GP", "UTS = ~c~ σ_{phys} exp[δ(x)],   δ ∼ GP"),
           ("Tempering", "~P~ = ~T~ (20 + log_{10} ~t~)"),
           ("90 % interval", "ŷ ± ~q~ σ(x)")]
    row_h = line_mm(F_EQ, S_EQ, 1.0) + 1.2
    box_h = 2 + line_mm(F_HEAD, S_HEAD, 0.9) + 1 + len(eqs) * row_h + 3
    rect(slide, x, y, w, box_h, fill="FFFFFF", line=BLUE, line_pt=2.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         name="equations box", radius=0.06)
    yy = y + 2
    tb = textbox(slide, x, yy, w, line_mm(F_HEAD, S_HEAD, 0.9) + 1, "heading: Key Equations")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
    _run(p, "Key Equations", F_HEAD, S_HEAD, bold=True, color=BLUE)
    yy += line_mm(F_HEAD, S_HEAD, 0.9) + 1
    for k, (lab, src) in enumerate(eqs):
        yy += equation_row(slide, x + 8, yy, w - 14, 58, f"{k + 1}. {lab}", src)
    y += box_h

    # software tool
    y += heading(slide, 0, y, "Software Tool: SteelRoute")
    y += paragraphs(slide, x, y, w, [
        "Python package and Gradio app: furnace vs air, or all five treatments, for any grade",
        "Open data and open code; every number on this poster is read from results/",
    ], after_pt=4.0, name="software")
    cmds = [("make setup", "virtual environment and requirements"),
            ("make pipeline", "stages 1–11 from the raw open data to results/ and figures/"),
            ("make notebooks", "11 executed notebooks that walk through the work"),
            ("make report", "report, slide deck and manuscript from results/")]
    lh = line_mm(F_BODY, 17, 1.0)
    bh = len(cmds) * lh + 5
    rect(slide, x, y + 1.5, w, bh, fill="F2F2F2", shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="commands box",
         radius=0.06)
    for k, (cmd, what) in enumerate(cmds):
        tb = textbox(slide, x + 5, y + 4 + k * lh, 52, lh, f"command {k + 1}")
        p = tb.text_frame.paragraphs[0]
        _ppr(p, spacing=1.0)
        _run(p, cmd, "Consolas", 17, bold=True, color=NAVY)
        tb = textbox(slide, x + 58, y + 4 + k * lh, w - 62, lh, f"command note {k + 1}")
        p = tb.text_frame.paragraphs[0]
        _ppr(p, spacing=1.0)
        _run(p, what, F_BODY, 17, color=INK)
    y += bh + 1.5

    # project team
    y += heading(slide, 0, y, "Project Team")
    y += paragraphs(slide, x, y, w, [
        "**Shibasish Banerjee** (23118071) and **Aviral Singh** (23118017)",
        "Supervisor: **Dr. Anish Karmakar**, Dept. of Metallurgical and Materials Engineering, IIT Roorkee",
    ], bullet=None, align=PP_ALIGN.CENTER, after_pt=2.0, name="team")
    y += heading(slide, 0, y, "Acknowledgements")
    y += paragraphs(slide, x, y, w, [
        "We thank Dr. Anish Karmakar for the question behind this work and for always asking how we knew a "
        "result was true, and the authors who shared their data openly.",
        "**AI use:** AI tools assisted with language, structure and code; we reviewed and validated all output. "
        "The content and conclusions are our own; no data were generated or altered by AI.",
    ], size=S_SMALL, bullet=None, after_pt=3.0, name="acknowledgements")
    left_end = y

    # ============================== RIGHT COLUMN
    x, y = COL_X[1], TOP
    y += heading(slide, 1, y, "Results and Interpretation", first=True)
    half = HALF
    gap = COL_W - 2 * HALF
    y += figure_block(slide, x, y, COL_W, SIZES["carbon"][1], "f1_furnace_vs_air_carbon",
                      "Fig 2: Plain-carbon steels (0.75 % Mn), furnace vs 25 mm bar in still air; lines: model "
                      "with 90 % band",
                      f"Air cooling adds +{N['gain_low']:.0f} MPa (low C) to +{N['gain_high']:.0f} MPa (high C); "
                      f"high-C elongation falls by {abs(N['el_high']):.1f} points", placements) + 3.0 + EXTRA_ROW
    y += figure_block(slide, x, y, COL_W, SIZES["cct"][1], "f2_cct_4140",
                      "Fig 3: Model CCT diagrams with the same furnace path and 25 mm still-air path",
                      ("1045 stays ferrite + pearlite; " if N["cct45_same"] else "")
                      + f"4140 forms {100 * N['cct_ac_B']:.0f} % bainite in air: a different steel",
                      placements) + 3.0 + EXTRA_ROW
    figs = [
        [("f3_parity_unseen_grades", SIZES["parity"][1],
          "Fig 4: Grade held out; A/N: annealed, normalised; bars: 90 % intervals",
          f"UTS within {N['an_uts']:.0f} MPa (A/N) and {N['qt_uts']:.0f} MPa (Q+T)"),
         ("f4_jominy_4140H", SIZES["jominy"][1],
          "Fig 5: Jominy hardness of 4140H, published vs calibrated kinetics",
          f"{N['jom_grades']} unseen grades: {N['jom_pub']:.1f} → {N['jom_cal']:.1f} HRC")],
        [("f5_five_treatments_4140", SIZES["five"][1],
          "Fig 6: Five treatments of a 25 mm 4140 bar; whiskers: 90 % interval",
          f"Same steel: {N['ft_min']:.0f} to {N['ft_max']:.0f} MPa by treatment alone"),
         ("f6_tempering_4140", SIZES["tempering"][1],
          "Fig 7: Tempering a 25 mm 4140 bar (oil, 1 h); circles: measured; shaded: TME, TE",
          f"Tracks the measured bars from {N['temp_meas_lo']:.0f} to {N['temp_meas_hi']:.0f} °C")],
    ]
    for row in figs:
        fh = max(f[1] for f in row)
        cap_l = max(wrap_count(parse_rich(f[2]), half - 1, F_BODY, S_CAP) for f in row)
        tk_l = max(wrap_count([(f[3], True, BLUE)], half - 1, F_BODY, S_TAKE) for f in row)
        used = 0.0
        for j, (name, h, cap, take) in enumerate(row):
            used = figure_block(slide, x + j * (half + gap), y, half, fh, name, cap, take, placements,
                                cap_lines=cap_l, take_lines=tk_l)
        y += used + 3.0 + EXTRA_ROW

    y += heading(slide, 1, y, "Conclusion and Future Work")
    y += paragraphs(slide, x, y, COL_W, [
        f"[[Furnace vs air:]] low-C steels barely change (+{N['gain_low']:.0f} MPa); medium and high C gain "
        f"+{N['gain_med']:.0f} and +{N['gain_high']:.0f} MPa from finer pearlite",
        "[[Hardenable steels:]] Cr-Mo and Ni-Cr-Mo turn bainitic in air, so bar size matters as much as the route",
        f"[[Accuracy:]] unseen-grade UTS within {N['an_uts']:.0f} MPa (A/N) and {N['qt_uts']:.0f} MPa (Q+T), "
        f"against {N['an_uts_txt']:.0f} and {N['qt_uts_txt']:.0f} MPa for textbook laws",
        f"[[Honest intervals:]] 90 % intervals hold {100 * N['an_cov']:.1f} % and {100 * N['qt_cov']:.0f} % of "
        "unseen values",
        f"[[Limits:]] {N['di_ratio']:.1f}× too hardenable vs Grossmann's ideal diameter; stress relief untested (no open data)",
        "[[Future work:]] lab tests of Indian grades (EN3B to EN42), predictions fixed in advance",
    ], name="conclusion", after_pt=4.0)

    y += heading(slide, 1, y, "References")
    y += paragraphs(slide, x, y, COL_W, [
        "M.V. Li et al., Metall. Mater. Trans. B 29, 661 (1998)",
        "T. Gladman, I.D. McIvor, F.B. Pickering, J. Iron Steel Inst. 210, 916 (1972)",
        "J.H. Hollomon, L.D. Jaffe, Trans. AIME 162, 223 (1945)",
        "N. Birbilis, M. Ghorbani, S. Zhao, Mendeley Data, doi:10.17632/jmwb9ddd43.1 (2025)",
        "J. Lei et al., J. Am. Stat. Assoc. 113, 1094 (2018)",
    ], size=S_SMALL, after_pt=1.0, indent=4.0, name="references")

    right_end = y

    print(f"left column ends at {left_end:.1f} mm, right at {right_end:.1f} mm, limit {BOTTOM:.1f} mm")
    prs.save(str(out_pptx))
    Path(placements_json).write_text(json.dumps(placements, indent=1))
    return left_end, right_end


if __name__ == "__main__":
    ends = build(HERE / "poster_draft.pptx", HERE / "placements.json")
    n_head = {0: 5, 1: 2}   # headings after the first one in each column
    EXTRA[0] = max(0.0, (BOTTOM - ends[0]) / n_head[0] - 0.3)
    globals()["EXTRA_ROW"] = max(0.0, (BOTTOM - ends[1]) / (n_head[1] + 4) - 0.3)
    EXTRA[1] = EXTRA_ROW
    print("extra space before headings:", {k: round(v, 1) for k, v in EXTRA.items()}, "between rows:",
          round(EXTRA_ROW, 1))
    build(HERE / "poster_draft.pptx", HERE / "placements.json")
