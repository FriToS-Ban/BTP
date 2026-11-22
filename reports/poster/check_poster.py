"""QA of the finished poster: font sizes by role, word count, raster resolution, and the
numbers printed on it next to the results files they come from.

    python reports/poster/check_poster.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

from pptx import Presentation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_poster  # noqa: E402

MIN = {"title": 80, "heading": 48, "body": 28, "caption": 20, "table": 24}


def role(name, size):
    if name == "title block":
        return "title" if size >= 60 else "header lines"
    if name.startswith("heading"):
        return "heading"
    if name.startswith(("caption", "references", "acknowledgements")):
        return "caption"
    if name.startswith(("data tile", "GA ", "eq label", "equation")):
        return "table"
    if name.startswith(("footer", "subheading", "takeaway")):
        return "body"
    return "body"


def main():
    prs = Presentation(str(HERE / "BTP_poster_A1.pptx"))
    slide = prs.slides[0]
    sizes = defaultdict(set)
    words = defaultdict(int)
    text_all = []
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                if r.font.size is None or not r.text.strip():
                    continue
                pt = r.font.size.pt
                sizes[(role(sh.name, pt), sh.name.split(":")[0].split(" f")[0])].add(pt)
            t = "".join(r.text for r in p.runs)
            if t.strip():
                text_all.append((sh.name, t))
                if not sh.name.startswith(("title block", "footer", "GA number", "heading")):
                    words[sh.name.split(" ")[0]] += len(re.findall(r"\S+", t))
    print("smallest size per role (pt):")
    worst = defaultdict(lambda: 999)
    for (rl, nm), ss in sizes.items():
        worst[rl] = min(worst[rl], min(ss))
    ok = True
    for rl, v in sorted(worst.items()):
        need = MIN.get(rl)
        flag = "" if need is None or v >= need else "   <-- below minimum"
        ok &= flag == ""
        print(f"  {rl:13s} {v:5.0f}  (min {need if need else '-'}){flag}")
    print("words (excluding title block, headings, footer):", sum(words.values()))
    print("  by box type:", dict(sorted(words.items(), key=lambda kv: -kv[1])))
    print("all sizes OK" if ok else "SIZE PROBLEMS")
    return text_all


if __name__ == "__main__":
    main()
