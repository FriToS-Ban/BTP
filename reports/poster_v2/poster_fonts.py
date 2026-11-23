"""Find the body font of the poster on this machine.

The reference poster is set in Calibri. Calibri ships with Windows and with Microsoft Office; where it is
missing we use Carlito, the open (SIL OFL) font with the same metrics, so line breaks are the same either way.
macOS carries Carlito as a collection (.ttc) for exactly this purpose; its four faces are written to a cache
folder as plain .ttf files so that both matplotlib and the browser can load them.
"""

from pathlib import Path

FACES = ("regular", "bold", "italic", "bolditalic")

CALIBRI = [
    {"regular": "C:/Windows/Fonts/calibri.ttf", "bold": "C:/Windows/Fonts/calibrib.ttf",
     "italic": "C:/Windows/Fonts/calibrii.ttf", "bolditalic": "C:/Windows/Fonts/calibriz.ttf"},
    {"regular": "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibri.ttf",
     "bold": "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibrib.ttf",
     "italic": "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibrii.ttf",
     "bolditalic": "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Calibriz.ttf"},
]
CARLITO_TTF = [
    {"regular": f"{d}/Carlito-Regular.ttf", "bold": f"{d}/Carlito-Bold.ttf", "italic": f"{d}/Carlito-Italic.ttf",
     "bolditalic": f"{d}/Carlito-BoldItalic.ttf"}
    for d in ("/usr/share/fonts/truetype/crosextra", "/usr/share/fonts/carlito", str(Path.home() / "Library/Fonts"),
              "/Library/Fonts")
]
CARLITO_TTC = ("/System/Library/PrivateFrameworks/FontServices.framework/Versions/A/Resources/Fonts/"
               "ApplicationSupport/Carlito.ttc")


def _all_exist(d):
    return all(Path(p).exists() for p in d.values())


def sans_fonts(cache):
    """Return {'family': 'Calibri' or 'Carlito', 'files': {face: Path}} for the four faces."""
    for d in CALIBRI:
        if _all_exist(d):
            return {"family": "Calibri", "files": {k: Path(v) for k, v in d.items()}}
    for d in CARLITO_TTF:
        if _all_exist(d):
            return {"family": "Carlito", "files": {k: Path(v) for k, v in d.items()}}
    cache = Path(cache)
    out = {k: cache / f"Carlito-{k}.ttf" for k in FACES}
    if not _all_exist(out):
        if not Path(CARLITO_TTC).exists():
            raise FileNotFoundError("Neither Calibri nor Carlito was found; install Carlito (SIL OFL) and re-run.")
        import logging

        from fontTools.ttLib import TTCollection
        logging.getLogger("fontTools").setLevel(logging.ERROR)
        cache.mkdir(parents=True, exist_ok=True)
        key = {"Regular": "regular", "Bold": "bold", "Italic": "italic", "Bold Italic": "bolditalic"}
        for f in TTCollection(CARLITO_TTC).fonts:
            f.save(str(out[key[f["name"].getDebugName(2)]]))
    return {"family": "Carlito", "files": out}
