"""Write the poster as an editable PPTX on the department template.

    python reports/poster_v2/export_pptx.py      # after build_poster.py; writes BTP_poster_A1_v2.pptx

build_poster.py lays the page out in Chrome and saves where every text block, box and figure landed
(.cache/layout.json). This script puts the same blocks at the same places on Template.pptx: the template's own
gradient band and logo, native text boxes (Calibri, Trebuchet MS, Times New Roman, Cambria), native shapes, and
the figures as 300 dpi pictures. The PDF is the print master; the PPTX is for editing in PowerPoint.
"""

import copy
import json
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / ".cache"
sys.path.insert(0, str(HERE))
from build_poster import BOTTOM, HEADER_H, OUT_NAME, PAGE_H, PAGE_W, TOP  # noqa: E402

FONT_MAP = {"Carlito": "Calibri", "STIX Two Text": "Cambria", "Times": "Times New Roman",
            "Kohinoor Devanagari": "Nirmala UI"}
BLUE, NAVY = "0070C0", "1F3973"
SVG_NS = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
SVG_EXT = "{96DAC541-7B7A-43D3-8B79-37D633B846F1}"
SLACK = 0.9      # mm added to the width of every text box, so a hair of extra width never wraps a line


def mm(v):
    return Emu(int(round(v * 36000)))


def rgb(h):
    return RGBColor.from_string(h)


def prepare_template(prs):
    """Keep the template's logo and take its gradient; its placeholder text boxes are replaced."""
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


def add_rect(slide, s, shape=MSO_SHAPE.RECTANGLE):
    if s.get("round"):
        shape = MSO_SHAPE.OVAL
    elif s.get("radius", 0) > 0:
        shape = MSO_SHAPE.ROUNDED_RECTANGLE
    sp = slide.shapes.add_shape(shape, mm(s["x"]), mm(s["y"]), mm(s["w"]), mm(s["h"]))
    sp.name = s["name"]
    if s.get("fill"):
        sp.fill.solid()
        sp.fill.fore_color.rgb = rgb(s["fill"])
    else:
        sp.fill.background()
    if s.get("line") and s.get("line_pt", 0) > 0:
        sp.line.color.rgb = rgb(s["line"])
        sp.line.width = Pt(s["line_pt"])
    else:
        sp.line.fill.background()
    sp.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sp.adjustments[0] = min(0.5, s["radius"] / min(s["w"], s["h"]))
    return sp


def set_body(tf, anchor):
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    bp = tf._txBody.find(qn("a:bodyPr"))
    for child in list(bp):
        bp.remove(child)
    etree.SubElement(bp, qn("a:noAutofit"))


def set_par(p, par):
    p.alignment = {"center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}.get(par["align"], PP_ALIGN.LEFT)
    pPr = p._p.get_or_add_pPr()
    ln = etree.SubElement(pPr, qn("a:lnSpc"))
    etree.SubElement(ln, qn("a:spcPts"), val=str(int(round(par["line"] * 100))))
    sb = etree.SubElement(pPr, qn("a:spcBef"))
    etree.SubElement(sb, qn("a:spcPts"), val=str(int(round(par["before"] * 100))))
    sa = etree.SubElement(pPr, qn("a:spcAft"))
    etree.SubElement(sa, qn("a:spcPts"), val=str(int(round(par["after"] * 100))))
    if par["bullet"]:
        ind = par["indent"]
        pPr.set("marL", str(int(ind * 36000)))
        pPr.set("indent", str(-int((ind - 0.6) * 36000)))
        bc = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bc, qn("a:srgbClr"), val=BLUE)
        etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
        etree.SubElement(pPr, qn("a:buChar"), char="•")
    else:
        etree.SubElement(pPr, qn("a:buNone"))


def add_run(p, r):
    run = p.add_run()
    run.text = r["t"]
    f = run.font
    face = FONT_MAP.get(r["font"], r["font"])
    f.name, f.size, f.bold, f.italic = face, Pt(r["size"]), r["bold"], r["italic"]
    f.color.rgb = rgb(r["color"])
    rPr = run._r.get_or_add_rPr()
    if r["base"]:
        rPr.set("baseline", str(r["base"]))
    if face == "Nirmala UI":
        for tag in ("a:ea", "a:cs"):
            etree.SubElement(rPr, qn(tag)).set("typeface", face)


def add_text(slide, b):
    centred = b["pars"] and b["pars"][0]["align"] == "center"
    x = b["x"] - (SLACK / 2 if centred else 0)
    tb = slide.shapes.add_textbox(mm(x), mm(b["y"]), mm(b["w"] + SLACK), mm(b["h"] + 0.6))
    tb.name = b["name"]
    tf = tb.text_frame
    set_body(tf, MSO_ANCHOR.MIDDLE if b["valign"] == "middle" else MSO_ANCHOR.TOP)
    for k, par in enumerate(b["pars"]):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        set_par(p, par)
        for r in par["runs"]:
            add_run(p, r)
    return tb


def attach_svg(slide, pic, svg_path, name):
    """Give a picture its vector original, the way PowerPoint stores an SVG: the PNG stays as the fallback."""
    part = Part(PackURI(f"/ppt/media/{name}.svg"), "image/svg+xml", slide.part.package, svg_path.read_bytes())
    rid = slide.part.relate_to(part, RT.IMAGE)
    blip = pic._element.blipFill.find(qn("a:blip"))
    ext = etree.SubElement(etree.SubElement(blip, qn("a:extLst")), qn("a:ext"), uri=SVG_EXT)
    etree.SubElement(ext, f"{{{SVG_NS}}}svgBlip", nsmap={"asvg": SVG_NS}).set(qn("r:embed"), rid)


def build(out_path):
    lay = json.loads((CACHE / "layout.json").read_text(encoding="utf-8"))
    prs = Presentation(str(HERE / "Template.pptx"))
    slide, grad, logo = prepare_template(prs)
    spTree = slide.shapes._spTree
    slide.background.fill.solid()                      # the template leaves the page unfilled; make it white
    slide.background.fill.fore_color.rgb = rgb("FFFFFF")

    # header band with the template's own gradient, behind everything
    band = add_rect(slide, {"name": "header band", "x": 0, "y": 0, "w": PAGE_W, "h": HEADER_H, "fill": "E8F1FA"})
    spPr = band._element.spPr
    for tag in ("a:solidFill", "a:noFill", "a:gradFill"):
        for e in spPr.findall(qn(tag)):
            spPr.remove(e)
    spPr.insert(2, grad)
    spTree.remove(band._element)
    spTree.insert(2, band._element)

    for s in lay["shapes"]:
        if s["name"] in ("header band", "page border", "column divider"):
            continue
        add_rect(slide, s)
    for a in lay["arrows"]:
        shape = {"r": MSO_SHAPE.RIGHT_ARROW, "l": MSO_SHAPE.LEFT_ARROW, "d": MSO_SHAPE.DOWN_ARROW}[a["dir"]]
        sp = slide.shapes.add_shape(shape, mm(a["x"]), mm(a["y"]), mm(a["w"]), mm(a["h"]))
        sp.fill.solid()
        sp.fill.fore_color.rgb = rgb(NAVY)
        sp.line.fill.background()
        sp.shadow.inherit = False
        sp.name = "GA arrow"
    for p in lay["pics"]:
        pic = slide.shapes.add_picture(str(HERE / "figures" / f"{p['name']}.png"), mm(p["x"]), mm(p["y"]),
                                       mm(p["w"]), mm(p["h"]))
        pic.name = f"figure {p['name']}"
        svg = HERE / "figures" / f"{p['name']}.svg"
        if svg.exists():
            attach_svg(slide, pic, svg, p["name"])
    for b in lay["boxes"]:
        add_text(slide, b)

    # the template's vector logo, with a PNG fallback for viewers that do not draw SVG
    lg = lay["logo"]
    logo.left, logo.top, logo.width, logo.height = mm(lg["x"]), mm(lg["y"]), mm(lg["w"]), mm(lg["h"])
    png = CACHE / "logo.png"
    if png.exists():
        _, rid = slide.part.get_or_add_image_part(str(png))
        blip = logo._element.blipFill.find(qn("a:blip"))
        if blip.get(qn("r:embed")) is None:
            blip.set(qn("r:embed"), rid)
    spTree.remove(logo._element)
    spTree.append(logo._element)

    div = slide.shapes.add_connector(1, mm(PAGE_W / 2), mm(TOP), mm(PAGE_W / 2), mm(BOTTOM))
    div.line.color.rgb = rgb("156082")
    div.line.width = Pt(1.5)
    div.name = "column divider"
    border = add_rect(slide, {"name": "page border", "x": 0.6, "y": 0.6, "w": PAGE_W - 1.2, "h": PAGE_H - 1.2,
                              "line": "0E2841", "line_pt": 2.5})
    border.name = "page border"

    prs.save(str(out_path))
    n = {"text boxes": len(lay["boxes"]), "shapes": len(lay["shapes"]) + len(lay["arrows"]),
         "figures": len(lay["pics"])}
    print("wrote", out_path.name, n)
    return n


def logo_png(width=1400):
    """Rasterise the template's SVG logo once (Chrome), for the PNG fallback."""
    png = CACHE / "logo.png"
    if png.exists() or not (CACHE / "logo.svg").exists():
        return
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": width})
        holder = CACHE / "logo.html"       # a file page, so that Chrome may load the local SVG
        holder.write_text(f'<body style="margin:0;background:transparent"><img src="logo.svg" '
                          f'style="width:{width}px;height:{width}px"></body>')
        page.goto(holder.as_uri())
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(png), omit_background=True)
        browser.close()


if __name__ == "__main__":
    logo_png()
    build(HERE / f"{OUT_NAME}.pptx")
