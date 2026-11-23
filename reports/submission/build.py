#!/usr/bin/env python
r"""Build the submission report, report.pdf, from the HTML sources in this folder.

    .venv/bin/python reports/submission/build.py

front.html   title page, declaration, acknowledgement, abstract, abbreviations
chapters/    one file per chapter, then the appendices
refs.py      the bibliography, cited in the text as <c k="key"/> and numbered in order of first citation
style.css    page and type styles (A4, Times New Roman, plain centred bold headings)

Markup used in the chapters:
    <h1 id="ch:x">Title</h1>                 chapter title (first line of each chapter file)
    <h2 id="sec:x">Title</h2>                numbered section
    <fig id="fig:x" src=".." w="90%" short="..">caption</fig>
    <tab id="tab:x" short=".."><cap>caption</cap><table>..</table><note>..</note></tab>
    <eq id="eq:x">LaTeX</eq>                 numbered display equation;  \( .. \) is inline math
    <ref to="fig:x"/>  ->  Figure 5.3        <ref to="fig:x" n/>  ->  5.3
    <c k="key1,key2"/> ->  [4, 7]

The body is rendered first, with invisible markers, so that the page of every heading, figure and
table is known. Then the front matter is rendered with the contents and the lists, twice, so that its
own page numbers are right, and the two parts are joined with page labels and bookmarks.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

from latex2mathml.converter import convert as tex_to_mathml
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "_build"
sys.path.insert(0, str(HERE))
from refs import REFS  # noqa: E402

TITLE = "Classify, Heat-Treat, Predict"
SUBTITLE = "Machine Learning for Steels"
AUTHORS = "Aviral Singh, Shibasish Banerjee"
CHAPTERS = ["ch1_introduction.html", "ch2_literature.html", "ch3_data.html", "ch4_methodology.html",
            "ch5_results.html", "ch6_tool.html", "ch7_conclusions.html"]
APPENDICES = []

UNITS = (r"(?:mm|cm|µm|nm|°C|K/s|K/h|MPa|GPa|HV|HB|HRC|wt%|kJ|J/kg/K|J|W/m/K|W m|K|h|min|s|grades|bars|"
         r"points|steels|papers|values|heats|members|chains|draws|rows|records|trajectories)")
NB_UNIT = re.compile(r"(?<=\d)[ \t\n]+(?=" + UNITS + r"(?![A-Za-z]))")
KIND_WORD = {"fig": "Figure", "tab": "Table", "eq": "Equation", "sec": "Section", "ch": "Chapter", "app": "Appendix"}


def attrs(s: str) -> dict:
    return dict(re.findall(r'([\w-]+)="([^"]*)"', s or ""))


def roman(n: int) -> str:
    out = ""
    for v, r in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"),
                 (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out, n = out + r, n - v
    return out


def compress(nums: list[int]) -> str:
    nums, out, i = sorted(set(nums)), [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(x) for x in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)


def math_to_mathml(s: str) -> str:
    return re.sub(r"\\\((.+?)\\\)", lambda m: tex_to_mathml(m.group(1).strip()), s, flags=re.S)


def nbsp_units(s: str) -> str:
    parts = re.split(r"(<math.*?</math>|<svg.*?</svg>|<pre.*?</pre>|<[^>]+>)", s, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = NB_UNIT.sub(" ", parts[i])
    return "".join(parts)


def img_tag(src: str, width: str) -> str:
    if src.endswith(".svg"):
        svg = (HERE / src).read_text()
        return f'<div class="svgwrap" style="width:{width}">{svg}</div>'
    path = (ROOT / src) if src.startswith("figures/") else (HERE / src)
    if not path.exists():
        raise FileNotFoundError(path)
    return f'<img src="{path.as_uri()}" style="width:{width}" alt="">'


class Doc:
    def __init__(self):
        self.labels: dict[str, tuple[str, str]] = {}
        self.toc: list[dict] = []
        self.figs: list[dict] = []
        self.tabs: list[dict] = []
        self.markers: dict[str, str] = {}
        self.cite_order: dict[str, int] = {}

    def mark(self, key: str) -> str:
        tok = f"QQK{len(self.markers) + 1:04d}QQ"
        self.markers[tok] = key
        return f'<span class="mk">@@{tok}@@</span>'

    def label(self, ident: str | None, num: str):
        if not ident:
            return
        kind = ident.split(":")[0]
        if kind not in KIND_WORD:
            raise ValueError(f"unknown label kind: {ident}")
        if ident in self.labels:
            raise ValueError(f"duplicate label: {ident}")
        self.labels[ident] = (kind, num)


PAT = re.compile(r"<h2(?P<h2a>[^>]*)>(?P<h2t>.*?)</h2>"
                 r"|<fig(?P<figa>\s[^>]*)>(?P<figc>.*?)</fig>"
                 r"|<tab(?P<taba>\s[^>]*)>(?P<tabc>.*?)</tab>"
                 r"|<eq(?P<eqa>(?:\s[^>]*)?)>(?P<eqc>.*?)</eq>", re.S)


def process_part(text: str, prefix: str, label: str, doc: Doc) -> str:
    m = re.match(r"\s*<h1([^>]*)>(.*?)</h1>", text, re.S)
    if not m:
        raise ValueError(f"part {prefix} has no <h1>")
    a, title = attrs(m.group(1)), m.group(2).strip()
    doc.label(a.get("id"), prefix)
    key = f"part:{prefix}"
    doc.toc.append({"level": 1, "label": label, "title": title, "key": key})
    head = (f'<div class="chapter-head">{doc.mark(key)}<div class="label">{label}</div>'
            f'<div class="title">{title}</div></div>')
    count = {"h2": 0, "fig": 0, "tab": 0, "eq": 0}

    def rep(mm: re.Match) -> str:
        if mm.group("h2t") is not None:
            count["h2"] += 1
            num, t = f"{prefix}.{count['h2']}", mm.group("h2t").strip()
            doc.label(attrs(mm.group("h2a")).get("id"), num)
            k = f"sec:{num}"
            doc.toc.append({"level": 2, "num": num, "title": t, "key": k})
            return f"<h2>{doc.mark(k)}<span class=\"hn\">{num}</span>&ensp;{t}</h2>"
        if mm.group("figc") is not None:
            count["fig"] += 1
            num, fa = f"{prefix}.{count['fig']}", attrs(mm.group("figa"))
            doc.label(fa.get("id"), num)
            k = f"fig:{num}"
            doc.figs.append({"num": num, "short": fa["short"], "key": k})
            cap = mm.group("figc").strip()
            return (f'<figure class="{fa.get("class", "")}">{doc.mark(k)}{img_tag(fa["src"], fa.get("w", "100%"))}'
                    f'<figcaption><span class="fl">Figure {num}:</span> {cap}</figcaption></figure>')
        if mm.group("tabc") is not None:
            count["tab"] += 1
            num, ta = f"{prefix}.{count['tab']}", attrs(mm.group("taba"))
            doc.label(ta.get("id"), num)
            k = f"tab:{num}"
            doc.tabs.append({"num": num, "short": ta["short"], "key": k})
            body = mm.group("tabc")
            cap = re.search(r"<cap>(.*?)</cap>", body, re.S).group(1).strip()
            table = re.search(r"<table.*?</table>", body, re.S).group(0)
            note = re.search(r"<note>(.*?)</note>", body, re.S)
            note_html = f'<div class="tnote">{note.group(1).strip()}</div>' if note else ""
            return (f'<div class="tab {ta.get("class", "")}">{doc.mark(k)}<div class="cap"><span class="fl">Table {num}:'
                    f"</span> {cap}</div>{table}{note_html}</div>")
        count["eq"] += 1
        num, ea = f"{prefix}.{count['eq']}", attrs(mm.group("eqa"))
        doc.label(ea.get("id"), num)
        lines = [ln.strip() for ln in mm.group("eqc").split("\\\\") if ln.strip()]
        mml = "".join(f'<div class="ln">{tex_to_mathml(ln, display="block")}</div>' for ln in lines)
        multi = " multi" if len(lines) > 1 else ""
        return f'<div class="eq{multi}"><div class="m">{mml}</div><div class="n">({num})</div></div>'

    return head + PAT.sub(rep, text[m.end():])


def resolve(s: str, doc: Doc) -> str:
    def ref(m: re.Match) -> str:
        to, bare = m.group(1), m.group(2) is not None
        if to not in doc.labels:
            raise KeyError(f"unresolved reference: {to}")
        kind, num = doc.labels[to]
        txt = f"({num})" if kind == "eq" else num
        return txt if bare else f"{KIND_WORD[kind]} {txt}"

    def cite(m: re.Match) -> str:
        nums = []
        for k in (x.strip() for x in m.group(1).split(",")):
            if k not in REFS:
                raise KeyError(f"unknown reference key: {k}")
            doc.cite_order.setdefault(k, len(doc.cite_order) + 1)
            nums.append(doc.cite_order[k])
        return f"[{compress(nums)}]"

    s = re.sub(r'<ref to="([^"]+)"(\s+n)?\s*/>', ref, s)
    return re.sub(r'<c k="([^"]+)"\s*/>', cite, s)


def page_html(body: str, css: str, extra_css: str = "") -> str:
    return (f'<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><title>{TITLE}</title>'
            f"<style>{css}\n{extra_css}</style></head><body>{body}</body></html>")


def with_markers(html_text: str, show: bool) -> str:
    return re.sub(r"@@(QQK\d{4}QQ)@@", (lambda m: m.group(1)) if show else (lambda m: ""), html_text)


def render(browser, html_text: str, out: pathlib.Path):
    src = out.with_suffix(".html")
    src.write_text(html_text)
    page = browser.new_page()
    page.goto(src.as_uri(), wait_until="load")
    page.evaluate("document.fonts.ready")
    page.pdf(path=str(out), prefer_css_page_size=True, print_background=True)
    page.close()


def marker_pages(pdf: pathlib.Path, doc: Doc) -> dict[str, int]:
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    pages, found = text.split("\f"), {}
    for i, t in enumerate(pages):
        for tok in re.findall(r"QQK\d{4}QQ", t.replace(" ", "")):
            found.setdefault(doc.markers[tok], i)
    missing = [k for k in doc.markers.values() if k not in found]
    if missing:
        raise RuntimeError(f"markers not found in {pdf.name}: {missing[:5]}")
    return found


def first_lines(pdf: pathlib.Path) -> list[str]:
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    # the same characters on every page, whatever order pdftotext reads them in
    return ["".join(sorted(re.sub(r"QQK\d{4}QQ", "", re.sub(r"\s+", "", t)))) for t in text.split("\f")]


def references_html(doc: Doc) -> str:
    items = sorted(doc.cite_order.items(), key=lambda kv: kv[1])
    rows = "".join(f'<div class="ref"><span class="rn">[{n}]</span><span class="rt">{REFS[k]}</span></div>'
                   for k, n in items)
    key = "part:refs"
    doc.toc.append({"level": 1, "label": "", "title": "References", "key": key})
    return (f'<div class="chapter-head refs-head">{doc.mark(key)}<div class="title">References</div></div>'
            f'<div class="refs">{rows}</div>')


def toc_line(cls: str, title: str, page: str, num: str = "") -> str:
    n = f'<span class="tn">{num}</span>' if num else ""
    return (f'<div class="toc {cls}">{n}<span class="tt">{title}</span><span class="ld"></span>'
            f'<span class="pg">{page}</span></div>')


def build_front(doc: Doc, body_pages: dict[str, int], front_pages: dict[str, int] | None) -> str:
    src = (HERE / "front.html").read_text()

    def fp(key):  # roman page of a front-matter section
        return roman(front_pages[key] + 1) if front_pages else "x"

    def bp(key):
        return str(body_pages[key] + 1)

    lines = []
    for key, title in (("ack", "Acknowledgement"),
                       ("abstract", "Abstract")):
        lines.append(toc_line("l0", title, fp(key)))
    for e in doc.toc:
        if e["level"] == 1:
            t = f"{e['label']}&ensp;{e['title']}" if e["label"] else e["title"]
            lines.append(toc_line("l1", t, bp(e["key"])))
        else:
            lines.append(toc_line("l2", e["title"], bp(e["key"]), e["num"]))
    contents = (f'<section class="fm" data-key="contents"><h1 class="fm-title">Contents</h1>'
                f'{"".join(lines)}</section>')
    lof = "".join(toc_line("lf", f["short"], bp(f["key"]), f"Figure {f['num']}") for f in doc.figs)
    lot = "".join(toc_line("lf", t["short"], bp(t["key"]), f"Table {t['num']}") for t in doc.tabs)
    src = src.replace("<!--CONTENTS-->", contents)
    src = src.replace("<!--LOF-->", f'<section class="fm" data-key="lof"><h1 class="fm-title">List of Figures</h1>{lof}</section>')
    src = src.replace("<!--LOT-->", f'<section class="fm" data-key="lot"><h1 class="fm-title">List of Tables</h1>{lot}</section>')
    src = src.replace("@@TITLE@@", TITLE).replace("@@SUBTITLE@@", SUBTITLE)
    return src


def front_marked(src: str, doc: Doc) -> str:
    return re.sub(r'<section class="fm" data-key="(\w+)">',
                  lambda m: f'<section class="fm" data-key="{m.group(1)}">{doc.mark("fm:" + m.group(1))}', src)


def main():
    BUILD.mkdir(exist_ok=True)
    css = (HERE / "style.css").read_text()
    doc = Doc()

    parts = []
    for i, name in enumerate(CHAPTERS, start=1):
        parts.append(process_part((HERE / "chapters" / name).read_text(), str(i), f"Chapter {i}", doc))
    appendix_parts = []
    for letter, name in zip("AB", APPENDICES):
        appendix_parts.append(process_part((HERE / "chapters" / name).read_text(), letter, f"Appendix {letter}", doc))
    parts = [resolve(p, doc) for p in parts]
    appendix_parts = [resolve(p, doc) for p in appendix_parts]
    body = "".join(parts) + references_html(doc) + "".join(appendix_parts)
    body = nbsp_units(math_to_mathml(body))
    # the references go before the appendices in the contents
    refs_entry = next(e for e in doc.toc if e["key"] == "part:refs")
    doc.toc.remove(refs_entry)
    first_app = next((i for i, e in enumerate(doc.toc) if e["key"] == "part:A"), len(doc.toc))
    doc.toc.insert(first_app, refs_entry)

    body_css = ('@page { @bottom-center { content: counter(page); font-family: "Times New Roman", serif; '
                "font-size: 12pt; } }")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        render(browser, page_html(with_markers(body, True), css, body_css), BUILD / "body_marked.pdf")
        body_pages = marker_pages(BUILD / "body_marked.pdf", doc)
        render(browser, page_html(with_markers(body, False), css, body_css), BUILD / "body.pdf")
        a, b = first_lines(BUILD / "body_marked.pdf"), first_lines(BUILD / "body.pdf")
        if a != b:
            bad = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) if len(a) == len(b) else "count"
            raise RuntimeError(f"markers changed the body layout (page {bad})")

        front_css = ('@page { @bottom-center { content: counter(page, lower-roman); font-family: "Times New Roman", '
                     "serif; font-size: 12pt; } } @page cover { @bottom-center { content: none; } }")
        fdoc_marks = Doc()
        src = front_marked(build_front(doc, body_pages, None), fdoc_marks)
        render(browser, page_html(nbsp_units(with_markers(src, True)), css, front_css), BUILD / "front_marked.pdf")
        fpages = marker_pages(BUILD / "front_marked.pdf", fdoc_marks)
        front_pages = {k.split(":", 1)[1]: v for k, v in fpages.items()}
        n_front_first = len(PdfReader(BUILD / "front_marked.pdf").pages)
        src = front_marked(build_front(doc, body_pages, front_pages), fdoc_marks)
        render(browser, page_html(nbsp_units(with_markers(src, False)), css, front_css), BUILD / "front.pdf")
        if len(PdfReader(BUILD / "front.pdf").pages) != n_front_first:
            raise RuntimeError("the front matter changed length when its page numbers were filled in")
        browser.close()

    front, main_pdf = PdfReader(BUILD / "front.pdf"), PdfReader(BUILD / "body.pdf")
    w = PdfWriter()
    for pg in front.pages:
        w.add_page(pg)
    for pg in main_pdf.pages:
        w.add_page(pg)
    nf, nb = len(front.pages), len(main_pdf.pages)
    w.set_page_label(0, nf - 1, style="/r", start=1)
    w.set_page_label(nf, nf + nb - 1, style="/D", start=1)
    for key, title in (("ack", "Acknowledgement"), ("abstract", "Abstract"),
                       ("contents", "Contents")):
        w.add_outline_item(title=title, page_number=front_pages[key])
    parent = None
    for e in doc.toc:
        page = nf + body_pages[e["key"]]
        if e["level"] == 1:
            t = f"{e['label']}  {e['title']}" if e["label"] else e["title"]
            parent = w.add_outline_item(title=re.sub("<[^>]+>", "", t), page_number=page)
        else:
            w.add_outline_item(title=f"{e['num']}  {re.sub('<[^>]+>', '', e['title'])}", page_number=page, parent=parent)
    w.add_metadata({"/Title": f"{TITLE}: {SUBTITLE}", "/Author": AUTHORS,
                    "/Subject": "B.Tech Project Report, Department of Metallurgical and Materials Engineering, "
                                "Indian Institute of Technology Roorkee",
                    "/Keywords": "heat treatment, annealing, normalising, hardening, tempering, stress relieving, "
                                 "steel, transformation kinetics, Gaussian process, uncertainty"})
    w.page_mode = "/UseOutlines"
    out = HERE / "report.pdf"
    with open(out, "wb") as fh:
        w.write(fh)
    print(f"front {nf} pages, body {nb} pages, {len(doc.figs)} figures, {len(doc.tabs)} tables, "
          f"{len(doc.cite_order)} references -> {out}")


if __name__ == "__main__":
    main()
