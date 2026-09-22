"""Inversion height over the Visayas, from GFS model levels, against Mactan.

The point of the ARL data: 40 levels with 46-270 m spacing through the lowest
2 km, against the six levels and 1.6 km gap in the forecast APIs. That is the
difference between resolving the trade inversion and stepping over it.

    python src/arl_cube.py && python src/plot_inversion.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from soundings import fetch, inversion, series

EPI = pd.Timestamp("2026-09-18 10:00")
DPI = 200


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")


def main() -> None:
    d = np.load(C.DATA / "wind3d_arl.npz", allow_pickle=True)
    t = pd.to_datetime(d["time"])
    lat, lon, z = d["lat"], d["lon"], d["z"]
    T, P = d["temp"], d["pres"]
    theta = T * (1000.0 / P) ** 0.2854
    zf = d["zfull"]

    i = int(np.argmin(abs(lat - C.CEBU["lat"])))
    j = int(np.argmin(abs(lon - C.CEBU["lon"])))
    th_c = np.nanmean(theta[:, :, i - 1:i + 2, j - 1:j + 2], axis=(2, 3))
    z_c = np.nanmean(zf[:, :, i - 1:i + 2, j - 1:j + 2], axis=(2, 3))

    # inversion from the model column, same definition used on the soundings
    zi, st = [], []
    for k in range(len(t)):
        a, b = inversion(z_c[k], th_c[k])
        zi.append(a); st.append(b)
    zi, st = np.array(zi), np.array(st)
    tp = t + pd.Timedelta(hours=C.TZ_OFFSET_H)

    days = [x.strftime("%Y-%m-%d") for x in pd.date_range("2026-09-13", "2026-09-21")]
    snd = series(days).dropna()
    snd.index = snd.index + pd.Timedelta(hours=C.TZ_OFFSET_H)

    fig = plt.figure(figsize=(13.2, 12.4), dpi=DPI)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0], hspace=0.42,
                          wspace=0.32, left=0.07, right=0.94, top=0.90, bottom=0.115)

    # --- 1. stability section with the inversion tracked --------------------
    ax = fig.add_subplot(gs[0, :])
    grad = np.gradient(th_c, axis=1) / np.gradient(z_c, axis=1) * 1000.0
    m = z.mean() * 0 + 1  # placeholder
    zk = z / 1000.0
    lim = np.nanpercentile(grad[:, zk < 5], 98)
    pm = ax.pcolormesh(mdates.date2num(tp), zk, grad.T, cmap="magma_r",
                       vmin=0, vmax=lim, shading="nearest")
    ax.plot(tp, zi / 1000.0, "o-", ms=3, lw=1.3, color="#22d3ee",
            label="model inversion")
    ax.plot(snd.index, snd.zi / 1000.0, "s", ms=6, color="#f472b6",
            mec="white", mew=0.8, label="Mactan sounding")
    ax.axvline(mdates.date2num(EPI), color="#111827", lw=1.8)
    ax.set_ylim(0, 5); ax.set_ylabel("height (km)", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.026)
    cb.set_label("d$\\theta$/dz (K km$^{-1}$)  ·  dark = stable",
                 fontsize=8, color=C.INK)
    cb.ax.tick_params(labelsize=7, colors=C.INK_MUTED)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")
    _style(ax)
    ax.set_title("Stability over Cebu, 40 model levels. GFS spreads the "
                 "inversion over 1–2.5 km, so the tracked height wanders",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    # --- 2. profiles, model against sounding --------------------------------
    for col, (day, hh) in enumerate((("2026-09-15", 0), ("2026-09-19", 0),
                                     ("2026-09-21", 0))):
        ax = fig.add_subplot(gs[1, col])
        s = fetch(day, hh)
        tt = pd.Timestamp(f"{day} {hh:02d}:00")
        k = int(np.argmin(abs(t - tt)))
        ax.plot(th_c[k], z_c[k] / 1000.0, "-o", ms=2.5, lw=1.5, color="#0e7490",
                label="GFS (40 lv)")
        if s is not None:
            ax.plot(s.theta, s.z / 1000.0, lw=1.4, color="#b45309",
                    label="Mactan sonde")
            a, b = inversion(s.z.values, s.theta.values)
            ax.axhline(a / 1000.0, color="#b45309", lw=0.9, ls=":")
        ax.axhline(zi[k] / 1000.0, color="#0e7490", lw=0.9, ls=":")
        ax.set_ylim(0, 5); ax.set_xlim(298, 330)
        ax.set_xlabel("$\\theta$ (K)", fontsize=8.5, color=C.INK)
        if col == 0:
            ax.set_ylabel("height (km)", fontsize=9, color=C.INK)
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9,
                      facecolor="white", edgecolor="#e7e5e4")
        _style(ax)
        ax.set_title(f"{tt + pd.Timedelta(hours=8):%d %b %H:%M} PHT",
                     fontsize=9, color=C.INK_MUTED, loc="left", pad=4)

    # --- 3. map of inversion height, and the model-sonde comparison ---------
    ax = fig.add_subplot(gs[2, :2])
    kk = int(np.argmin(abs(t - pd.Timestamp("2026-09-19 00:00"))))
    gmap = np.gradient(theta[kk], axis=0) / np.gradient(zf[kk], axis=0) * 1000.0
    win = (zf[kk] > 200) & (zf[kk] < 4000)
    gm = np.where(win, gmap, -np.inf)
    zmap = np.take_along_axis(zf[kk], np.argmax(gm, axis=0)[None], axis=0)[0]
    pm2 = ax.pcolormesh(lon, lat, zmap / 1000.0, cmap="viridis", shading="nearest",
                        vmin=1.0, vmax=4.0)
    try:
        import cartopy.io.shapereader as shp
        from shapely.geometry import box as sbox
        clip = sbox(lon.min(), lat.min(), lon.max(), lat.max())
        for g in shp.Reader(shp.natural_earth(resolution="50m", category="physical",
                                              name="coastline")).geometries():
            q = g.intersection(clip)
            for part in (q.geoms if hasattr(q, "geoms") else [q]):
                xy = np.asarray(part.coords)
                if len(xy) > 1:
                    ax.plot(xy[:, 0], xy[:, 1], color="w", lw=0.6)
    except Exception:
        pass
    ax.plot(C.CEBU["lon"], C.CEBU["lat"], "*", ms=13, color="w", mec="#111827",
            mew=0.8)
    cb2 = fig.colorbar(pm2, ax=ax, pad=0.012, fraction=0.03)
    cb2.set_label("inversion height (km)", fontsize=8, color=C.INK)
    cb2.ax.tick_params(labelsize=7, colors=C.INK_MUTED)
    ax.set_xlabel("longitude", fontsize=8.5, color=C.INK)
    ax.set_ylabel("latitude", fontsize=8.5, color=C.INK)
    _style(ax)
    ax.set_title(f"Spatial variation, {tp[kk]:%d %b %H:%M} PHT — the lid is not "
                 f"flat", fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)

    ax = fig.add_subplot(gs[2, 2])
    # four definitions, each applied identically to model and sonde. None of
    # them verifies -- which is the result, not a detail of the plotting.
    from soundings import fetch as _f
    def metrics(zz, tt, rr):
        o = np.argsort(zz); zz, tt, rr = zz[o], tt[o], rr[o]
        m = (zz >= 0) & (zz <= 5000) & np.isfinite(tt)
        zz, tt, rr = zz[m], tt[m], rr[m]
        g = np.gradient(tt, zz) * 1000
        out = {}
        k = np.where(tt > tt[0] + 0.5)[0]; out["mixed layer"] = zz[k[0]] if len(k) else np.nan
        k = np.where((g > 4.0) & (zz > 300))[0]
        out["stable base"] = zz[k[0]] if len(k) else np.nan
        k = (zz > 300) & np.isfinite(rr)
        out["RH maximum"] = zz[k][np.argmax(rr[k])] if k.sum() else np.nan
        k = (zz > 200) & (zz < 4000)
        out["max dθ/dz"] = zz[k][np.argmax(g[k])] if k.sum() else np.nan
        return out
    rows = []
    for day in pd.date_range("2026-09-13", "2026-09-21"):
        for hh in (0, 12):
            sd = _f(day.strftime("%Y-%m-%d"), hh)
            if sd is None:
                continue
            tt0 = pd.Timestamp(f"{day:%Y-%m-%d} {hh:02d}:00")
            kk2 = int(np.argmin(abs(t - tt0)))
            if abs((t[kk2] - tt0).total_seconds()) > 5400:
                continue
            M = metrics(z_c[kk2], th_c[kk2],
                        np.nanmean(d["relh"][kk2, :, i-1:i+2, j-1:j+2], axis=(1, 2)))
            S = metrics(sd.z.values, sd.theta.values, sd.relh.values)
            rows.append((M, S))
    names = ["mixed layer", "stable base", "RH maximum", "max dθ/dz"]
    rr_, rms_ = [], []
    for nm in names:
        a = np.array([s2[nm] for _, s2 in rows]); b = np.array([m2[nm] for m2, _ in rows])
        ok = np.isfinite(a) & np.isfinite(b)
        rr_.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan)
        rms_.append(np.sqrt(np.mean((b[ok] - a[ok]) ** 2)) / 1000 if ok.sum() > 2 else np.nan)
    y = np.arange(len(names))
    ax.barh(y, rr_, color=["#b91c1c" if v < 0.3 else "#047857" for v in rr_], height=0.55)
    ax.axvline(0, color="#d6d3d1", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("correlation, model vs sonde", fontsize=8.5, color=C.INK)
    for k2, (v, e) in enumerate(zip(rr_, rms_)):
        ax.annotate(f"RMS {e:.1f} km", (0.02 if v < 0 else -0.02, k2),
                    ha="left" if v < 0 else "right", va="center", fontsize=7,
                    color=C.INK_MUTED)
    _style(ax)
    ax.set_title(f"None of four definitions verifies (n={len(rows)})",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)

    fig.text(0.07, 0.968, "The inversion over Cebu: does GFS see it?", fontsize=14, color=C.INK,
             weight="bold", va="top")
    fig.text(0.07, 0.938,
             f"NOAA GFS 0.25° ARL, 40 model levels (46–270 m through the lowest "
             f"2 km) · 3-hourly · {C.WATERMARK.replace('Made by: ', '')}",
             fontsize=9, color=C.INK_MUTED, va="top")
    notes = ["VERDICT: the bulk stratification agrees (middle row), but GFS does not reproduce the OBSERVED inversion height profile-by-profile under any of four definitions.",
             "It spreads the trade inversion across 1-2.5 km where the sonde has a sharp step, so a height picked from the model is not well posed.",
             "Mactan is a WMO station that GDAS almost certainly assimilates, so agreement is a consistency check, not independent validation.",
             "ARL vertical velocity is dp/dt in hPa/s, negative upward; converted here to m/s via w = -omega/(rho g).",
             "Soundings are 00Z and 12Z only, and 18 and 20 Sep 00Z were not published."]
    for n, line in enumerate(notes + C.attribution(coastlines=False, trajectory=False, cams=False, gfs=True)):
        fig.text(0.07, 0.092 - n * 0.0108, line, fontsize=6.4,
                 color=C.INK_MUTED, va="top")

    out = C.FIGS / "inversion.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")
    print(f"  model inversion: median {np.nanmedian(zi):.0f} m, "
          f"strength {np.nanmedian(st):.1f} K/km")
    print(f"  sonde inversion: median {snd.zi.median():.0f} m, "
          f"strength {snd.strength.median():.1f} K/km")


if __name__ == "__main__":
    main()
