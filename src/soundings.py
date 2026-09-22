"""Mactan radiosondes (WMO 98646), cached locally.

The only direct observation of the vertical structure over Cebu. Two per day,
~115 levels, so it resolves both the shallow surface inversion and the trade
inversion that every gridded product we have smooths or misses.

Caveat for any comparison: Mactan is a WMO station, so GDAS very likely
assimilates it. Agreement is a consistency check, not independent validation.

    python src/soundings.py 2026-09-13 2026-09-21
"""
from __future__ import annotations
import os, re, sys, html, subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C

CACHE = C.DATA / "soundings"
URL = ("https://weather.uwyo.edu/wsgi/sounding?datetime={d}%20{h}:00:00"
       "&id=98646&src=UNKNOWN&type=TEXT:LIST")


def fetch(day: str, hour: int) -> pd.DataFrame | None:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{day}_{hour:02d}Z.txt"
    if not f.exists():
        # curl, not requests: the server rejects the re-encoded URL requests sends
        r = subprocess.run(["curl", "-sS", "--http1.1", "-m", "60",
                            URL.format(d=day, h=f"{hour:02d}")],
                           capture_output=True, text=True)
        pre = re.findall(r"<PRE>(.*?)</PRE>", r.stdout, re.S | re.I)
        f.write_text(html.unescape(pre[0]) if pre else "")
    txt = f.read_text()
    rows = []
    for line in txt.splitlines():
        g = line.split()
        if len(g) >= 11 and re.match(r"^\d", g[0]):
            try:
                rows.append([float(x) for x in g[:11]])
            except ValueError:
                pass
    if len(rows) < 10:
        return None
    a = np.array(rows)
    return pd.DataFrame(dict(pres=a[:, 0], z=a[:, 1], temp=a[:, 2], dwpt=a[:, 3],
                             relh=a[:, 4], drct=a[:, 6], sped=a[:, 7], theta=a[:, 8]))


def inversion(z, theta, zmin=200.0, zmax=4000.0):
    """Height and strength of the strongest stable layer in a window.

    Strength is dtheta/dz in K/km, which is what distinguishes a true capping
    inversion (several K/km) from ordinary free-tropospheric stability.
    """
    m = (z >= zmin) & (z <= zmax) & np.isfinite(theta)
    if m.sum() < 4:
        return np.nan, np.nan
    zz, th = z[m], theta[m]
    g = np.gradient(th, zz) * 1000.0
    k = int(np.argmax(g))
    return float(zz[k]), float(g[k])


def series(days, hours=(0, 12)) -> pd.DataFrame:
    out = []
    for d in days:
        for h in hours:
            s = fetch(d, h)
            if s is None:
                out.append(dict(time=pd.Timestamp(f"{d} {h:02d}:00"), zi=np.nan,
                                strength=np.nan, n=0))
                continue
            zi, st = inversion(s.z.values, s.theta.values)
            out.append(dict(time=pd.Timestamp(f"{d} {h:02d}:00"), zi=zi,
                            strength=st, n=len(s)))
    return pd.DataFrame(out).set_index("time")


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a, b)]
    s = series(days)
    print(s.to_string(float_format=lambda x: f"{x:.0f}"))
