"""Assemble a static site into site/ for GitHub Pages.

Videos use preload="none" with a poster, so nothing downloads until someone
presses play. GIFs are 5x larger and would autoplay on load.

    python src/build_site.py
"""
from __future__ import annotations
import os, shutil, subprocess, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from make_gif import _ffmpeg

SITE = C.ROOT / "site"
# (stem, anchor id, heading, caption) -- ids give each panel a shareable link,
# which is what figure numbers would be for on a page with no body text
ANIM = [("aerosol_optical_depth", "aod", "Aerosol optical depth",
         "Column smoke loading, with VIIRS thermal anomalies."),
        ("carbon_monoxide", "co", "Carbon monoxide",
         "Surface CO at 10 m — the conservative tracer. Persists after aerosol is scavenged."),
]
FIGS = [("cebu_forecast.png", "forecast", "PM2.5 and PM10 · 24 h outlook",
         "Corrected CAMS against the DENR-EMB station, with a 50% predictive band. "
         "The shaded floor is the WHO 2021 24-hour guideline.")]

CSS = """*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
color:#1c1917;background:#fafaf9}
.wrap{max-width:1000px;margin:0 auto;padding:0 20px 64px}
header{padding:48px 0 8px}
h1{font-size:1.9rem;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:#78716c;margin:0 0 4px}
.stamp{color:#a8a29e;font-size:.85rem}
section{margin:44px 0 0}
h2{font-size:1.15rem;margin:0 0 4px;letter-spacing:-.01em}
h2 a{color:inherit;text-decoration:none}
h2 a:hover{text-decoration:underline}
p.cap{color:#78716c;margin:0 0 14px;font-size:.92rem}
video,img.fig{width:100%;height:auto;border:1px solid #e7e5e4;border-radius:8px;background:#fff}
.note{background:#fff;border:1px solid #e7e5e4;border-left:3px solid #b45309;
border-radius:6px;padding:14px 16px;margin:28px 0;font-size:.92rem}
.note b{color:#b45309}
footer{margin-top:56px;padding-top:20px;border-top:1px solid #e7e5e4;
color:#a8a29e;font-size:.78rem;line-height:1.7}
a{color:#0e7490}
@media(prefers-color-scheme:dark){
body{background:#0c0a09;color:#e7e5e4}
video,img.fig,.note{background:#1c1917;border-color:#292524}
h1,h2{color:#fafaf9}.sub,p.cap{color:#a8a29e}
footer{border-color:#292524}}
"""


def poster(stem):
    exe = _ffmpeg()
    src, out = C.FIGS / f"{stem}.mp4", SITE / f"{stem}.jpg"
    if not exe or not src.exists():
        return None
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", "select=eq(n\\,0)", "-vframes", "1", "-q:v", "4", str(out)],
                   check=False)
    return out if out.exists() else None


def main() -> None:
    # rebuild from scratch: publish_site.sh adds the whole directory, so a
    # figure dropped from FIGS would otherwise stay on the site forever
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    stamp = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=C.TZ_OFFSET_H)
    body = []

    for stem, anchor, title, cap in ANIM:
        mp4 = C.FIGS / f"{stem}.mp4"
        if not mp4.exists():
            print(f"  skip {stem}: no mp4", file=sys.stderr); continue
        shutil.copy2(mp4, SITE / mp4.name)
        gif = C.FIGS / f"{stem}.gif"
        extra = ""
        if gif.exists():
            shutil.copy2(gif, SITE / gif.name)
            extra = f' &middot; <a href="{gif.name}" download>GIF, {gif.stat().st_size/1e6:.1f} MB</a>'
        p = poster(stem)
        pa = f' poster="{p.name}"' if p else ""
        body.append(f"""<section id="{anchor}"><h2><a href="#{anchor}">{title}</a></h2>
<p class="cap">{cap}{extra}</p>
<video controls loop muted playsinline preload="none"{pa}>
<source src="{mp4.name}" type="video/mp4"></video></section>""")

    for fn, anchor, title, cap in FIGS:
        src = C.FIGS / fn
        if not src.exists():
            continue
        shutil.copy2(src, SITE / fn)
        body.append(f"""<section id="{anchor}"><h2><a href="#{anchor}">{title}</a></h2>
<p class="cap">{cap}</p>
<img class="fig" src="{fn}" alt="{title}" loading="lazy"></section>""")

    for md in ("README.md",):
        if (C.ROOT / md).exists():
            shutil.copy2(C.ROOT / md, SITE / md)

    (SITE / "style.css").write_text(CSS)
    (SITE / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cebu haze source tracking</title>
<link rel="stylesheet" href="style.css"></head><body><div class="wrap">
<header><h1>Cebu haze source tracking</h1>
<p class="sub">Tracing particulate episodes over Cebu City to their source, from free satellite and model data.</p>
<p class="stamp">Updated {stamp:%a %d %b %Y %H:%M} Philippine time</p></header>

<div class="note"><b>These are model fields, not measurements.</b> CAMS ran
2.7&times; low for PM<sub>2.5</sub> against the DENR-EMB station in Cebu, and its
diurnal cycle is close to anti-phased. Corrected products on this page carry a
predictive band; read it, not just the central line.</div>

{"".join(body)}

<footer>
Contains modified Copernicus Atmosphere Monitoring Service information {stamp:%Y} &middot;
DWD ICON winds &middot; VIIRS active fire / thermal anomalies: NASA LANCE/FIRMS &middot;
imagery: NASA Worldview/GIBS &middot; served via Open-Meteo (CC BY 4.0) &middot;
coastlines: Natural Earth<br>
Neither the European Commission nor ECMWF is responsible for any use that may be made of the information it contains.<br>
Juan Carlos Graciosa, Xavier Bacalla, Junelie Velonta, Vhan Sabellano
&middot;
<a href="https://github.com/jcgraciosa/cebu-pollution-tracking">source</a>
</footer></div></body></html>""")

    tot = sum(f.stat().st_size for f in SITE.iterdir() if f.is_file())
    print(f"site/ built: {len(list(SITE.iterdir()))} files, {tot/1e6:.1f} MB total")
    mp4s = sum(f.stat().st_size for f in SITE.glob("*.mp4"))
    print(f"  mp4 payload if all played: {mp4s/1e6:.1f} MB; initial load is HTML+posters only")


if __name__ == "__main__":
    main()
