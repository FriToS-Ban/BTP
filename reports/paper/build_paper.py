"""Build the manuscript (reports/paper/manuscript.pdf) from paper.template.html and results/.

    .venv/bin/python reports/paper/build_paper.py

Numbers come from the same tokens as the report (reports/build_report.py), so
the paper, the report and the deck always agree with the last pipeline run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports"))
from build_report import tokens  # noqa: E402

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: "Iowan Old Style", "Charter", Georgia, serif; font-size: 10.6pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 8pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt; }
h3 { font-size: 11pt; margin: 11pt 0 4pt; font-style: italic; font-weight: 600; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
.authors { font-size: 11pt; margin-bottom: 2pt; }
.affil { font-size: 9.5pt; color: #444; margin-bottom: 12pt; }
.box { border: 0.6pt solid #999; padding: 8pt 10pt; margin: 8pt 0 12pt; background: #fafaf7; }
.box p { margin-bottom: 4pt; }
.kw { font-size: 9.5pt; color: #333; }
figure { margin: 10pt 0 12pt; page-break-inside: avoid; break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 9.2pt; line-height: 1.35; color: #222; margin-top: 4pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 4pt 0 6pt; page-break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #bbb; padding: 3pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 1pt solid #333; font-weight: 600; }
.tcap { font-size: 9.2pt; margin-top: 10pt; page-break-after: avoid; }
.eq { font-size: 10pt; margin: 4pt 0 8pt 16pt; }
ol.refs { font-size: 8.9pt; padding-left: 18pt; }
ol.refs li { margin-bottom: 2.5pt; }
ul { margin-top: 0; }
"""


def md_to_html_table(md):
    """Convert the pipe tables produced by build_report.tokens() to HTML."""
    lines = [ln for ln in md.strip().splitlines() if ln.startswith("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines if not set(ln) <= set("|-: ")]
    head, body = rows[0], rows[1:]
    bold = lambda c: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", c)  # noqa: E731
    html = "<table><thead><tr>" + "".join(f"<th>{bold(c)}</th>" for c in head) + "</tr></thead><tbody>"
    html += "".join("<tr>" + "".join(f"<td>{bold(c)}</td>" for c in r) + "</tr>" for r in body)
    rest = md.strip().split("\n\n", 1)
    note = f"<p style='font-size:8.6pt'>{rest[1]}</p>" if len(rest) > 1 else ""
    return html + "</tbody></table>" + note


def main():
    T = tokens()
    for k in ("ladder_table", "transfer_table", "category_table"):
        T[k] = md_to_html_table(T[k])
    tpl = (HERE / "paper.template.html").read_text()
    missing = sorted(set(re.findall(r"«(\w+)»", tpl)) - set(T))
    if missing:
        raise KeyError(f"tokens without a value: {missing}")
    body = re.sub(r"«(\w+)»", lambda m: T[m.group(1)], tpl)
    fig = (ROOT / "figures").as_uri()
    body = body.replace('src="FIG/', f'src="{fig}/')
    html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
    out_html = HERE / "manuscript.html"
    out_html.write_text(html)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="chrome")
        except Exception:
            b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(out_html.as_uri())
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(HERE / "manuscript.pdf"), format="A4", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template='<div style="font-size:8pt;width:100%;text-align:center;color:#666">'
                               '<span class="pageNumber"></span></div>',
               margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
        b.close()
    print("wrote", HERE / "manuscript.pdf")


if __name__ == "__main__":
    main()
