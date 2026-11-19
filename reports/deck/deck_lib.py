"""Small HTML slide toolkit for the project deck (rendered to PDF by Chrome).

Every slide is a fixed 1280 x 720 px page. Numbers on the slides are read
from results/*.csv by build_deck.py at build time, so the deck cannot drift
from the analysis.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"

CSS = r"""
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fcfcfb; }
body { font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif; color: #0b0b0b;
       -webkit-font-smoothing: antialiased; font-variant-numeric: tabular-nums; }
.slide { width: 1280px; height: 720px; position: relative; overflow: hidden; page-break-after: always;
         break-after: page; padding: 40px 60px 46px; background: #fcfcfb; display: flex; flex-direction: column; }
.slide:last-child { page-break-after: auto; break-after: auto; }
.kicker { font-size: 12.5px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #256abf;
          margin-bottom: 6px; }
h1 { font-size: 40px; line-height: 1.12; font-weight: 700; letter-spacing: -0.015em; margin: 0 0 18px; }
h2 { font-size: 27px; line-height: 1.22; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 16px;
     max-width: 1140px; }
h3 { font-size: 17px; font-weight: 700; margin: 0 0 8px; color: #0b0b0b; }
p, li { font-size: 17.5px; line-height: 1.45; margin: 0 0 9px; }
ul { margin: 0; padding-left: 22px; }
li { margin-bottom: 7px; }
.small, .small li, .small p { font-size: 14.5px; line-height: 1.4; }
.tiny { font-size: 12px; color: #52514e; line-height: 1.35; }
.muted { color: #52514e; }
.content { flex: 1; display: flex; flex-direction: column; min-height: 0; }
.row { display: flex; gap: 30px; align-items: flex-start; min-height: 0; }
.col { flex: 1; min-width: 0; }
.col-2 { flex: 2; min-width: 0; }
.col-3 { flex: 3; min-width: 0; }
.center { display: flex; justify-content: center; align-items: center; }
figure { margin: 0; display: flex; flex-direction: column; align-items: center; min-height: 0; }
figure img { max-width: 100%; object-fit: contain; display: block; }
figcaption { font-size: 12px; color: #52514e; margin-top: 6px; line-height: 1.35; text-align: left; width: 100%; }
.footer { position: absolute; left: 60px; right: 60px; bottom: 16px; display: flex; justify-content: space-between;
          font-size: 11px; color: #898781; }
.callout { border-left: 4px solid #256abf; background: #f1f5fb; padding: 12px 16px; border-radius: 4px; }
.callout p, .callout li { margin-bottom: 5px; }
.warn { border-left-color: #eda100; background: #fdf7ea; }
.good { border-left-color: #1baf7a; background: #eef8f3; }
.stat { display: flex; flex-direction: column; padding: 12px 16px; border: 1px solid #e1e0d9; border-radius: 8px;
        background: #ffffff; }
.stat .v { font-size: 30px; font-weight: 700; letter-spacing: -0.01em; line-height: 1.1; }
.stat .l { font-size: 13px; color: #52514e; margin-top: 4px; line-height: 1.3; }
.stats { display: flex; gap: 14px; }
.stats .stat { flex: 1; }
table.data { border-collapse: collapse; width: 100%; font-size: 14px; }
table.data th { text-align: left; font-weight: 700; color: #52514e; border-bottom: 1.5px solid #c3c2b7;
                padding: 6px 8px; font-size: 12.5px; letter-spacing: .02em; }
table.data td { padding: 5.5px 8px; border-bottom: 1px solid #e9e8e2; vertical-align: top; }
table.data td.num, table.data th.num { text-align: right; white-space: nowrap; }
table.data tr.hl td { background: #f1f5fb; font-weight: 650; }
.fc { color: #2a78d6; font-weight: 700; }
.ac { color: #eb6834; font-weight: 700; }
.pill { display: inline-block; font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
        background: #eceae3; color: #52514e; margin-right: 6px; }
.eq { font-family: "Iowan Old Style", "Charter", Georgia, serif; font-size: 17px; background: #f4f3ee;
      border-radius: 4px; padding: 6px 10px; margin: 4px 0 8px; display: block; }
.chain { display: flex; align-items: stretch; gap: 0; margin: 6px 0 4px; }
.chain .box { flex: 1; border: 1.5px solid #c3c2b7; border-radius: 8px; padding: 10px 12px; background: #fff;
              font-size: 14px; line-height: 1.35; }
.chain .box b { display: block; font-size: 15px; margin-bottom: 3px; }
.chain .arrow { width: 26px; display: flex; align-items: center; justify-content: center; color: #898781;
                font-size: 20px; }
.title-slide { justify-content: center; padding: 70px 90px; }
.title-slide .content { justify-content: center; }
.title-slide .meta { font-size: 18px; line-height: 1.6; color: #52514e; margin-top: 26px; }
.rq { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.rq .q { border: 1px solid #e1e0d9; border-radius: 8px; padding: 12px 16px; background: #fff; }
.rq .q b { color: #256abf; }
.section-slide { justify-content: center; }
.section-slide .num { font-size: 15px; font-weight: 700; color: #256abf; letter-spacing: .1em; }
.section-slide h1 { font-size: 44px; }
"""


def esc(s):
    return html.escape(str(s))


def img(name, height=None, width=None, caption=None, embed=True):
    """A figure from figures/ (embedded as base64 so the HTML is self-contained)."""
    path = FIG / name
    if embed:
        data = base64.b64encode(path.read_bytes()).decode()
        src = f"data:image/png;base64,{data}"
    else:
        src = str(path)
    style = []
    if height:
        style.append(f"max-height:{height}px")
    if width:
        style.append(f"max-width:{width}px")
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" style="{";".join(style)}">{cap}</figure>'


def table(header, rows, num_cols=(), hl_rows=(), col_widths=None):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}"'
                 f'{f" style=width:{col_widths[i]}" if col_widths else ""}>{h}</th>' for i, h in enumerate(header))
    body = []
    for r_i, r in enumerate(rows):
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r))
        body.append(f'<tr class="{"hl" if r_i in hl_rows else ""}">{tds}</tr>')
    return f'<table class="data"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def stat(value, label):
    return f'<div class="stat"><div class="v">{value}</div><div class="l">{label}</div></div>'


class Deck:
    def __init__(self, footer_left, footer_right=""):
        self.slides = []
        self.footer_left = footer_left
        self.footer_right = footer_right

    def add(self, body, title=None, kicker=None, cls="", footer=True):
        n = len(self.slides) + 1
        head = ""
        if kicker:
            head += f'<div class="kicker">{kicker}</div>'
        if title:
            head += f"<h2>{title}</h2>"
        foot = (f'<div class="footer"><span>{self.footer_left}</span><span>{self.footer_right}'
                f'&nbsp;&nbsp;{n}</span></div>') if footer else ""
        self.slides.append(f'<section class="slide {cls}">{head}<div class="content">{body}</div>{foot}</section>')

    def html(self, title="deck"):
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
                f"<style>{CSS}</style></head><body>{''.join(self.slides)}</body></html>")


def render_pdf(html_path, pdf_path, png_dir=None):
    """Print the HTML deck to PDF with Chrome (Playwright); optionally save PNG previews."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(Path(html_path).resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(path=str(pdf_path), width="1280px", height="720px", print_background=True,
                 prefer_css_page_size=True)
        if png_dir is not None:
            Path(png_dir).mkdir(parents=True, exist_ok=True)
            n = page.evaluate("document.querySelectorAll('section.slide').length")
            for i in range(n):
                el = page.query_selector_all("section.slide")[i]
                el.screenshot(path=str(Path(png_dir) / f"slide_{i + 1:02d}.png"))
        browser.close()


def render_pptx(html_path, pptx_path, scale=2):
    """Save the deck as a 16:9 PowerPoint file, one full-slide picture per slide (looks identical to the PDF)."""
    import io

    from PIL import Image
    from playwright.sync_api import sync_playwright
    from pptx import Presentation
    from pptx.util import Emu

    shots = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:
            browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=scale)
        page.goto(Path(html_path).resolve().as_uri())
        page.wait_for_load_state("networkidle")
        for el in page.query_selector_all("section.slide"):
            shots.append(el.screenshot(type="png"))
        browser.close()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 13.333 x 7.5 in
    for png in shots:
        buf = io.BytesIO()
        Image.open(io.BytesIO(png)).convert("RGB").save(buf, "JPEG", quality=90, optimize=True)
        buf.seek(0)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(buf, 0, 0, prs.slide_width, prs.slide_height)
    prs.save(str(pptx_path))
