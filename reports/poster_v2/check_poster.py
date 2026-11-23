"""QA of the finished poster: the PDF itself (size, fonts, text), type sizes by role, word count, and
every number printed on it next to the results file it comes from.

    python reports/poster_v2/check_poster.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from pptx import Presentation
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_poster as bp  # noqa: E402

MIN = {"title": 40, "heading": 28, "body": 20, "takeaway": 17, "caption": 16, "small": 15}   # reference-poster scale


def role(name):
    if name == "title block":
        return "title block"
    if name.startswith("heading"):
        return "heading"
    if name.startswith("takeaway"):
        return "takeaway"
    if name.startswith("caption"):
        return "caption"
    if name.startswith(("intro", "methods", "conclusion", "data", "subheading", "footer")):
        return "body"
    return "small"      # tiles, graphical abstract, equations, references, acknowledgements


def check_pdf(path, N):
    ok = True
    r = PdfReader(str(path))
    page = r.pages[0]
    w, h = float(page.mediabox.width) * 25.4 / 72, float(page.mediabox.height) * 25.4 / 72
    print(f"PDF: {len(r.pages)} page, {w:.1f} x {h:.1f} mm, {path.stat().st_size / 1e6:.2f} MB")
    ok &= len(r.pages) == 1 and abs(w - bp.PAGE_W) < 0.5 and abs(h - bp.PAGE_H) < 0.5
    fonts, missing = set(), []
    for f in (page["/Resources"].get("/Font") or {}).values():
        f = f.get_object()
        name = str(f.get("/BaseFont", "Type3")).lstrip("/").split("+")[-1]
        fonts.add(name)
        if f.get("/Subtype") == "/Type3":
            continue
        d = f.get("/FontDescriptor") or (f["/DescendantFonts"][0].get_object().get("/FontDescriptor")
                                         if "/DescendantFonts" in f else None)
        d = d.get_object() if d is not None else {}
        if not any(k in d for k in ("/FontFile", "/FontFile2", "/FontFile3")):
            missing.append(name)
    print("  fonts:", ", ".join(sorted(fonts)), "| not embedded:", missing or "none")
    ok &= not missing
    text = re.sub(r"\s+", " ", page.extract_text()).replace(" ", " ")
    want = [bp.TITLE] + [f"Fig {k}:" for k in range(1, 16)] + [
        f"{N['an_uts']:.0f} MPa", f"{N['qt_uts']:.0f} MPa", f"{N['jom_cal']:.1f} HRC", f"{100 * N['an_cov']:.1f} %",
        f"{N['cls_family']:.2f}", f"{N['jom_pub']:.1f}", f"+{N['gain_high']:.0f} MPa", f"{N['bs_pred']:.0f} vs",
        f"{N['di_ratio']:.1f}×", f"{N['cct_pts']}", f"{N['qt_bars']}", f"{N['jom_pts']}", f"{N['lit_rows']}",
        f"{N['fc4140']:.0f} MPa", f"{N['ac4140']:.0f} MPa", f"{N['ft_min']:.0f} to {N['ft_max']:.0f} MPa"]
    lost = [s for s in want if s not in text]
    print(f"  text: {len(text.split())} words extracted; expected strings missing: {lost or 'none'}")
    ok &= not lost
    return ok


def check_layout():
    lay = json.loads((bp.CACHE / "layout.json").read_text(encoding="utf-8"))
    smallest = defaultdict(lambda: 999.0)
    words = 0
    for b in lay["boxes"]:
        for p in b["pars"]:
            for r in p["runs"]:
                if r["t"].strip() and not r["base"]:
                    smallest[role(b["name"])] = min(smallest[role(b["name"])], r["size"])
            if not b["name"].startswith(("title", "heading", "footer", "GA number")):
                words += len(re.findall(r"\S+", "".join(r["t"] for r in p["runs"])))
    ok = True
    print("smallest type size per role (pt):")
    for rl, v in sorted(smallest.items()):
        need = MIN.get(rl) if rl != "title block" else None
        flag = "" if need is None or v >= need - 0.01 else "   <-- below the reference scale"
        ok &= flag == ""
        print(f"  {rl:12s} {v:5.1f}  (reference {need if need else '-'}){flag}")
    title = next(b for b in lay["boxes"] if b["name"] == "title block")["pars"][0]["runs"][0]["size"]
    print(f"  title        {title:5.1f}  (reference {MIN['title']})")
    ok &= title >= MIN["title"]
    print(f"words outside title, headings and footer: {words}; figures: {len(lay['pics'])}")
    return ok


def numbers_table(N):
    rows = [("UTS error, annealed + normalised, unseen grades (MPa)", N["an_uts"], "s04_model_benchmark.csv"),
            ("  textbook laws alone (MPa)", N["an_uts_txt"], "s04_model_benchmark.csv"),
            ("UTS error, quenched + tempered, unseen grades (MPa)", N["qt_uts"], "s09_qt_benchmark.csv"),
            ("  textbook physics alone (MPa)", N["qt_uts_txt"], "s09_qt_benchmark.csv"),
            ("90 % interval coverage, annealed + normalised", N["an_cov"], "s04_conformal.csv"),
            ("90 % interval coverage, quenched + tempered", N["qt_cov"], "s09_qt_benchmark.csv"),
            ("Jominy band error, published constants (HRC)", N["jom_pub"], "s08_kinetic_calibration_cv_metrics.csv"),
            ("Jominy band error, recalibrated, unseen grades (HRC)", N["jom_cal"], "s08_kinetic_calibration_cv_metrics.csv"),
            ("  H-steels / points in that test", f"{N['jom_grades']} / {N['jom_pts']}", "s08_kinetic_calibration_cv.csv"),
            ("steel-family accuracy on real heats", N["cls_family"], "s02_classifier_cv.csv"),
            ("  real heats", N["cls_heats"], "s02_external_tests.csv"),
            ("ferrite / pearlite / martensite start MAE (C)", f"{N['fs_mae']:.1f} / {N['ps_mae']:.1f} / {N['ms_mae']:.1f}",
             "s03_kinetics_summary.csv"),
            ("CCT atlas: hardness RMSE (HV), F+P fraction MAE", f"{N['cct_hv']:.1f}, {N['cct_fp']:.3f}",
             "s03_cct_atlas_summary.csv"),
            ("air-cooling gain, low / medium / high C (MPa)",
             f"{N['gain_low']:.1f} / {N['gain_med']:.1f} / {N['gain_high']:.1f}", "s05_category_summary.csv"),
            ("bainite + martensite in air, Cr-Mo / Ni-Cr-Mo", f"{N['bm_crmo']:.3f} / {N['bm_nicrmo']:.3f}",
             "s05_category_summary.csv"),
            ("bar-size effect: sign agreement, predicted vs measured (MPa)",
             f"{N['bs_sign']:.2f} of {N['bs_n']}, {N['bs_pred']:.1f} vs {N['bs_meas']:.1f}", "s06_bar_sizes_metrics.csv"),
            ("five treatments of 4140, lowest and highest UTS (MPa)", f"{N['ft_min']:.1f}, {N['ft_max']:.1f}",
             "s09_five_treatments.csv"),
            ("4140 handbook UTS annealed / normalised (MPa)", f"{N['fc4140']:.0f} / {N['ac4140']:.0f}", "s01_route_pairs.csv"),
            ("4140 quenched + tempered bars, measured UTS range (MPa)", f"{N['qt4140_lo']:.0f} - {N['qt4140_hi']:.0f}",
             "s09_qt_oof.csv"),
            ("25 mm bar in air vs furnace, cooling-rate ratio", N["air_vs_furnace"], "s03_cooling_paths.csv"),
            ("model D_I / ASTM A255 D_I (median)", N["di_ratio"], "s09_hardenability_check.csv"),
            ("mostly bainite + martensite in air: D_I < 40 mm, D_I >= 100 mm",
             f"{N['di_lo'][0]} of {N['di_lo'][1]}, {N['di_hi'][0]} of {N['di_hi'][1]}", "s08_ideal_diameter_catalogue.csv"),
            ("Spearman, D_I vs UTS gain on air cooling", N["di_rho"], "s08_ideal_diameter_catalogue.csv"),
            ("fully tempered hardness per % C (HV); slowest softening with",
             f"{N['temper_sC']:.0f}; {N['temper_slowest']}", "s09_tempering_constants.csv"),
            ("assumptions in the register", N["assumptions"], "s10_assumptions.csv"),
            ("handbook = 25 mm bars: grades within 3 MPa", f"{N['hm_within']} of {N['hm_grades']}",
             "s10_handbook_section_match.csv")]
    print("numbers on the poster, from results/:")
    for lab, v, src in rows:
        v = f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"  {lab:58s} {v:>22s}   {src}")


def check_pptx(path):
    prs = Presentation(str(path))
    sl = prs.slides[0]
    w, h = prs.slide_width / 36000, prs.slide_height / 36000
    pics = [s for s in sl.shapes if s.shape_type == 13]
    off = [s.name for s in sl.shapes if s.left < -40000 or s.top < -40000 or s.left + s.width > prs.slide_width + 40000
           or s.top + s.height > prs.slide_height + 40000]
    print(f"PPTX: {w:.1f} x {h:.1f} mm, {len(sl.shapes)} shapes, {len(pics)} pictures, off page: {off or 'none'}")
    return not off and abs(w - bp.PAGE_W) < 0.5


def main():
    N = bp.numbers()
    ok = check_pdf(HERE / f"{bp.OUT_NAME}.pdf", N)
    ok &= check_layout()
    if (HERE / f"{bp.OUT_NAME}.pptx").exists():
        ok &= check_pptx(HERE / f"{bp.OUT_NAME}.pptx")
    numbers_table(N)
    print("ALL CHECKS PASSED" if ok else "PROBLEMS FOUND")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
