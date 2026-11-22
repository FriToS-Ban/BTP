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
from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
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
S_TITLE, S_SUB, S_HDR, S_HEAD, S_SUBHEAD, S_BODY, S_CAP, S_TAKE, S_SMALL = 80, 34, 32, 48, 31, 28, 22, 28, 22
LNSP = 0.90            # line spacing used for every text block

PAGE_W, PAGE_H = 594.12, 840.98
MARGIN = 15.0
GUTTER = 15.0
COL_W = (PAGE_W - 2 * MARGIN - GUTTER) / 2
COL_X = (MARGIN, MARGIN + COL_W + GUTTER)
HEADER_H = 114.0
FOOTER_H = 22.0
TOP = HEADER_H + 6.0
BOTTOM = PAGE_H - FOOTER_H - 5.0

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
    n["temp_max"], n["temp_min"] = t.UTS_MPa_pred.max(), t.UTS_MPa_pred.min()
    m = qo[(qo.grade.astype(str) == "4140") & (qo.section_mm == 25) & (qo.quench_medium == "oil")]
    n["temp_meas_lo"], n["temp_meas_hi"] = m.temper_T_C.min(), m.temper_T_C.max()
    cct = json.loads((HERE / "data/cct_4140.json").read_text()) if (HERE / "data/cct_4140.json").exists() else None
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


def paragraphs(slide, x, y, w, items, size=S_BODY, face=F_BODY, bullet="•", indent=7.5, after_pt=5.0,
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


def heading(slide, col, y, text, first=False):
    before = 0.0 if first else 3.0 + EXTRA[col]
    h = line_mm(F_HEAD, S_HEAD, 0.9) + 1.0
    tb = textbox(slide, COL_X[col], y + before, COL_W, h, f"heading: {text}")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
    _run(p, text, F_HEAD, S_HEAD, bold=True, color=BLUE)
    return before + h + 1.5


def subheading(slide, x, y, w, text):
    h = line_mm(F_BODY, S_SUBHEAD, 0.9) + 0.5
    tb = textbox(slide, x, y, w, h, f"subheading: {text}")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, spacing=0.9)
    _run(p, text, F_BODY, S_SUBHEAD, bold=True, color=BLUE)
    return h + 1.5


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


def equation_row(slide, x, y, w, label_w, label, src, size=S_BODY):
    h = line_mm(F_EQ, size, 1.0) + 0.5
    tb = textbox(slide, x, y, label_w, h, f"eq label: {label}")
    tb.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tb.text_frame.paragraphs[0]
    _ppr(p, spacing=0.9)
    _run(p, label, F_BODY, 24, bold=True, color=NAVY)
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
    logo.left, logo.top, logo.width, logo.height = mm(13), mm(9), mm(62), mm(62)
    spTree.remove(logo._element)
    spTree.append(logo._element)

    # header text
    tb = textbox(slide, MARGIN, 6.0, PAGE_W - 2 * MARGIN, HEADER_H - 8, "title block")
    tf = tb.text_frame
    lines = [("Furnace or Air Cooling?", F_TITLE, S_TITLE, True, False, INK, 0.86),
             ("Predicting What Changes in a Steel", F_TITLE, S_TITLE, True, False, INK, 0.86),
             ("Physics-informed machine learning for five heat treatments of steel", F_TITLE, S_SUB, False, True,
              NAVY, 1.0),
             ("B.Tech Project (MTP-400A), Autumn 2026   |   Department of Metallurgical and Materials "
              "Engineering, IIT Roorkee", F_TITLE, S_HDR, False, False, INK, 1.0),
             ("Shibasish Banerjee, Aviral Singh   |   Supervisor: Dr. Anish Karmakar", F_TITLE, 36, False, False,
              INK, 1.0)]
    for k, (t, face, size, bold, ital, col, sp) in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        _ppr(p, align=PP_ALIGN.CENTER, spacing=sp, before_pt=4 if k in (2, 4) else 0)
        _run(p, t, face, size, bold=bold, italic=ital, color=col)
    hdr_need = sum(line_mm(f, s, sp) for _, f, s, *_, sp in lines) + 8 * 25.4 / 72
    print(f"header text needs {hdr_need:.1f} mm of {HEADER_H - 8:.1f}")

    # page border, column divider, footer
    rect(slide, 0.6, 0.6, PAGE_W - 1.2, PAGE_H - 1.2, line="0E2841", line_pt=3.0, name="page border")
    div = slide.shapes.add_connector(1, mm(PAGE_W / 2), mm(TOP), mm(PAGE_W / 2), mm(BOTTOM))
    div.line.color.rgb = RGBColor.from_string("156082")
    div.line.width = Pt(2.0)
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

    # graphical abstract
    y += heading(slide, 0, y, "Graphical Abstract")
    steps = [("Inputs", "composition, heat treatment, bar size"),
             ("Transformation", "heat flow + Li kinetics"),
             ("Microstructure", "ferrite, pearlite, bainite, martensite"),
             ("Physics strength", "Gladman, Maynier, Hollomon–Jaffe"),
             ("GP ensemble", "learns what the physics misses"),
             ("Properties", "YS, UTS, El, HB with 90 % intervals")]
    aw = 12.0
    bw = (w - 2 * aw) / 3
    inner = bw - 6
    t_lines = max(wrap_count([(t, True, INK)], inner, F_BODY, 26) for t, _ in steps)
    s_lines = max(wrap_count([(s, False, INK)], inner, F_BODY, 24) for _, s in steps)
    bh = 6.5 + t_lines * line_mm(F_BODY, 26, 0.9) + s_lines * line_mm(F_BODY, 24, 0.9) + 2.5
    rowgap = 10.0
    pos = []
    for k in range(6):
        row, colk = divmod(k, 3)
        if row == 1:
            colk = 2 - colk
        pos.append((x + colk * (bw + aw), y + 3 + row * (bh + rowgap)))
    for k, ((t, s), (bx, by)) in enumerate(zip(steps, pos)):
        last = k == 5
        b = rect(slide, bx, by, bw, bh, fill=("FDE9D9" if last else PALE), line=BLUE, line_pt=2.0,
                 shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"GA box {k + 1}", radius=0.12)
        tb = textbox(slide, bx + 3, by + 5.5, bw - 6, bh - 6.5, f"GA text {k + 1}")
        tf = tb.text_frame
        p = tf.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
        _run(p, t, F_BODY, 26, bold=True, color=NAVY)
        p = tf.add_paragraph()
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
        _run(p, s, F_BODY, 24, color=INK)
        c = rect(slide, bx - 3.5, by - 3.5, 11, 11, fill=BLUE, shape=MSO_SHAPE.OVAL, name=f"GA number {k + 1}")
        tf = c.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=1.0)
        _run(p, str(k + 1), F_BODY, 24, bold=True, color="FFFFFF")
    for k in range(5):
        (ax0, ay0), (ax1, ay1) = pos[k], pos[k + 1]
        if k == 2:   # down arrow at the right
            a = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, mm(ax0 + bw / 2 - 5), mm(ay0 + bh + 1),
                                       mm(10), mm(rowgap - 2))
        elif k < 2:
            a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, mm(ax0 + bw + 1.5), mm(ay0 + bh / 2 - 5),
                                       mm(aw - 3), mm(10))
        else:
            a = slide.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, mm(ax1 + bw + 1.5), mm(ay0 + bh / 2 - 5),
                                       mm(aw - 3), mm(10))
        a.fill.solid()
        a.fill.fore_color.rgb = RGBColor.from_string(NAVY)
        a.line.fill.background()
        a.shadow.inherit = False
        a.name = f"GA arrow {k + 1}"
    y += 3 + 2 * bh + rowgap + 4

    # methodology: data strip + model chain
    y += heading(slide, 0, y, "Methodology")
    y += subheading(slide, x, y, w, "Open data (every row has its heat treatment)")
    tiles = [(f"{N['hb_grades']}", "handbook grades"),
             (f"{N['qt_bars']}", f"Q+T bars, {N['qt_grades']} grades"),
             (f"{N['jom_pts']}", "Jominy band points"),
             (f"{N['cct_pts']}", "CCT-atlas points"),
             (f"{N['lit_rows']}", "literature values")]
    tg = 3.0
    tw = (w - 4 * tg) / 5
    lab_lines = max(wrap_count([(lab, False, INK)], tw - 3, F_BODY, 24) for _, lab in tiles)
    th = 2.5 + line_mm(F_BODY, 38, 0.85) + lab_lines * line_mm(F_BODY, 24, 0.88) + 2.5
    for k, (num, lab) in enumerate(tiles):
        tx = x + k * (tw + tg)
        rect(slide, tx, y, tw, th, fill=PALE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"data tile {k + 1}",
             radius=0.08)
        tb = textbox(slide, tx + 1.5, y + 2.0, tw - 3, th - 2.5, f"data tile text {k + 1}")
        tf = tb.text_frame
        p = tf.paragraphs[0]
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.85)
        _run(p, num, F_BODY, 38, bold=True, color=BLUE)
        p = tf.add_paragraph()
        _ppr(p, align=PP_ALIGN.CENTER, spacing=0.88)
        _run(p, lab, F_BODY, 24, color=INK)
    y += th + 3.5
    y += subheading(slide, x, y, w, "Model chain")
    y += paragraphs(slide, x, y, w, [
        "**Thermodynamics:** Ae1, Ae3, Acm, lever rule",
        "**Heat flow:** furnace, bar in still air, radial quench",
        "**Kinetics:** Li et al.; bainite calibrated on Jominy bands (MCMC)",
        "**Quench + temper (Q+T):** Hollomon–Jaffe master curve",
        "**Properties:** Gaussian processes anchored on physics strength",
        "**Validation:** leave-one-grade-out; split-conformal 90 % intervals",
    ], name="methods", after_pt=3.0)

    # key equations box
    y += 4.0
    eqs = [("Bainite rate", "~k~_{B} = ~f~_{B} 2^{0.29~G~} (~B~_{s} − ~T~)^{2} e^{−~Q~/~RT~} / BC"),
           ("Physics strength", "σ_{phys} = ~f~_{FP} σ_{Gladman} + ~f~_{BM} · 3.3 HV_{Maynier}"),
           ("Anchored GP", "UTS = ~c~ σ_{phys} exp[δ(x)],   δ ∼ GP"),
           ("Tempering", "~P~ = ~T~ (20 + log_{10} ~t~)"),
           ("90 % interval", "ŷ ± ~q~ σ(x)")]
    row_h = line_mm(F_EQ, S_BODY, 1.0) + 0.5
    box_h = 2 + line_mm(F_HEAD, S_HEAD, 0.9) + 1 + len(eqs) * row_h + 3
    rect(slide, x, y, w, box_h, fill="FFFFFF", line=BLUE, line_pt=3.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         name="equations box", radius=0.06)
    yy = y + 2
    tb = textbox(slide, x, yy, w, line_mm(F_HEAD, S_HEAD, 0.9) + 1, "heading: Key Equations")
    p = tb.text_frame.paragraphs[0]
    _ppr(p, align=PP_ALIGN.CENTER, spacing=0.9)
    _run(p, "Key Equations", F_HEAD, S_HEAD, bold=True, color=BLUE)
    yy += line_mm(F_HEAD, S_HEAD, 0.9) + 1
    for k, (lab, src) in enumerate(eqs):
        yy += equation_row(slide, x + 6, yy, w - 10, 74, f"{k + 1}. {lab}", src)
    y += box_h

    # software tool
    y += heading(slide, 0, y, "Software Tool: SteelRoute")
    y += paragraphs(slide, x, y, w, [
        "Python package and Gradio app: furnace vs air, or all five treatments, for any grade",
        "**make pipeline** reruns all 11 stages from open data; every number on this poster is read from "
        "results/",
    ], after_pt=4.0, name="software")

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
    half = (COL_W - 10.0) / 2
    figs = [
        [("f1_furnace_vs_air_carbon", 100,
          "Fig 1: Plain-carbon steels; circles are handbook values",
          f"Air cooling adds +{N['gain_low']:.0f} MPa (low C) to +{N['gain_high']:.0f} MPa (high C)"),
         ("f2_cct_4140", 100,
          f"Fig 2: AISI 4140 CCT; the furnace path gives {100 * N['cct_fc_F']:.0f} % F + {100 * N['cct_fc_P']:.0f} % P",
          f"In air, 4140 forms {100 * N['cct_ac_B']:.0f} % bainite: a different steel")],
        [("f3_parity_unseen_grades", 100,
          "Fig 3: Grade held out; A/N: annealed, normalised; bars: 90 % intervals",
          f"UTS within {N['an_uts']:.0f} MPa (A/N) and {N['qt_uts']:.0f} MPa (Q+T)"),
         ("f4_jominy_4140H", 100,
          "Fig 4: Jominy hardness of 4140H, published vs calibrated kinetics",
          f"{N['jom_grades']} unseen grades: {N['jom_pub']:.1f} → {N['jom_cal']:.1f} HRC")],
        [("f5_five_treatments_4140", 92,
          "Fig 5: Five treatments of a 25 mm 4140 bar; whiskers: 90 % interval",
          f"Same steel: {N['ft_min']:.0f} to {N['ft_max']:.0f} MPa by treatment alone"),
         ("f6_tempering_4140", 92,
          "Fig 6: Tempering a 25 mm 4140 bar (oil, 1 h); circles: measured; shaded: TME, TE",
          f"Tracks the measured bars from {N['temp_meas_lo']:.0f} to {N['temp_meas_hi']:.0f} °C")],
    ]
    for row in figs:
        fh = max(f[1] for f in row)
        cap_l = max(wrap_count(parse_rich(f[2]), half - 1, F_BODY, S_CAP) for f in row)
        tk_l = max(wrap_count([(f[3], True, BLUE)], half - 1, F_BODY, S_TAKE) for f in row)
        for j, (name, h, cap, take) in enumerate(row):
            fx = x + j * (half + 10.0)
            placements.append({"kind": "svg", "name": name, "x": fx, "y": y, "w": half, "h": h})
            ph = rect(slide, fx, y, half, h, line="BBBBBB", line_pt=0.5, name=f"figure slot {name}")
            cy = y + fh + 1.5
            tb = textbox(slide, fx, cy, half, cap_l * line_mm(F_BODY, S_CAP) + 1, f"caption {name}")
            p = tb.text_frame.paragraphs[0]
            _ppr(p, align=PP_ALIGN.CENTER)
            fig_no, rest = cap.split(":", 1)
            _run(p, fig_no + ":", F_BODY, S_CAP, bold=True)
            _run(p, rest, F_BODY, S_CAP)
            ty = cy + cap_l * line_mm(F_BODY, S_CAP) + 1.0
            tb = textbox(slide, fx, ty, half, tk_l * line_mm(F_BODY, S_TAKE) + 1, f"takeaway {name}")
            p = tb.text_frame.paragraphs[0]
            _ppr(p, align=PP_ALIGN.CENTER)
            _run(p, take, F_BODY, S_TAKE, bold=True, color=BLUE)
        y += fh + 1.5 + cap_l * line_mm(F_BODY, S_CAP) + 1.0 + tk_l * line_mm(F_BODY, S_TAKE) + 5.0

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
    ], size=S_SMALL, after_pt=1.5, indent=5.5, name="references")

    right_end = y

    print(f"left column ends at {left_end:.1f} mm, right at {right_end:.1f} mm, limit {BOTTOM:.1f} mm")
    prs.save(str(out_pptx))
    Path(placements_json).write_text(json.dumps(placements, indent=1))
    return left_end, right_end


if __name__ == "__main__":
    ends = build(HERE / "poster_draft.pptx", HERE / "placements.json")
    n_head = {0: 5, 1: 2}   # headings after the first one in each column
    for col in (0, 1):
        EXTRA[col] = max(0.0, (BOTTOM - ends[col]) / n_head[col] - 0.3)
    print("extra space before headings:", {k: round(v, 1) for k, v in EXTRA.items()})
    build(HERE / "poster_draft.pptx", HERE / "placements.json")
