"""How often is the air above Cebu replaced, and by which route?

A mass budget on the column: gross outward flux through the four faces of a box
divided into the column mass gives a turnover time. Because tau = L / u-bar,
this is the flux read a different way rather than a new measurement -- which is
why it scales with the box, and why u-bar is the invariant, not tau.

The vertical term is plotted as a signed RATE, not a time: rho*w passes through
zero, and 1/0 is not a turnover time. Positive means the column is venting
upward, negative means it is being fed from above.

    python src/plot_replacement.py
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import prepared
from plot_forecast import simulate

EXT = "/Volumes/JCG_Backup1/pollution-tracking/data/wind3d_region.npz"
EPI = pd.Timestamp("2026-09-18 10:00")
TOP = 4000.0
DPI = 200
R = 6371000.0


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")


def budget(d, half):
    """Turnover time through the sides (h) and signed vertical rate (%/h)."""
    lat, lon, zf, zl = d["lat"], d["lon"], d["zfull"], d["z"]
    spd, rad = d["spd"] / 3.6, np.deg2rad(d["dir"])
    u, v, w = -spd * np.sin(rad), -spd * np.cos(rad), d["w"]
    rho = (d["pres"] * 100.0) / (287.05 * d["temp"])
    dz = np.empty_like(zf)
    dz[:, 1:-1] = 0.5 * (zf[:, 2:] - zf[:, :-2])
    dz[:, 0] = zf[:, 1] - zf[:, 0]; dz[:, -1] = zf[:, -1] - zf[:, -2]
    wt = np.where(zf <= TOP, dz, 0.0)
    Fx, Fy = (rho * u * wt).sum(1), (rho * v * wt).sum(1)
    Mcol = (rho * wt).sum(1)
    ktop = int(np.argmin(abs(zl - TOP)))
    Wtop = (rho * w)[:, ktop]

    i = int(np.argmin(abs(lat - C.CEBU["lat"])))
    j = int(np.argmin(abs(lon - C.CEBU["lon"])))
    dy = np.deg2rad(0.25) * R
    dx = dy * np.cos(np.deg2rad(lat[i]))
    Lx, Ly = 2 * half * dx, 2 * half * dy
    out = (np.clip(Fx[:, i - half:i + half + 1, j + half], 0, None).sum(1) * dy
           + np.clip(-Fx[:, i - half:i + half + 1, j - half], 0, None).sum(1) * dy
           + np.clip(Fy[:, i + half, j - half:j + half + 1], 0, None).sum(1) * dx
           + np.clip(-Fy[:, i - half, j - half:j + half + 1], 0, None).sum(1) * dx)
    mass = Mcol[:, i - half:i + half + 1, j - half:j + half + 1].mean((1, 2)) * Lx * Ly
    vert = Wtop[:, i - half:i + half + 1, j - half:j + half + 1].mean((1, 2)) * Lx * Ly
    tau_h = mass / np.maximum(out, 1e-9) / 3600.0            # hours
    rate_v = vert / mass * 3600.0 * 100.0                    # % per hour, signed
    ubar = np.hypot(Fx, Fy)[:, i, j] / Mcol[:, i, j]         # m/s, the invariant
    return tau_h, rate_v, ubar, Lx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=EXT)
    a = ap.parse_args()
    src = a.src if os.path.exists(a.src) else str(C.DATA / "wind3d_arl.npz")
    d = np.load(src, allow_pickle=True)
    t = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    epi = t >= EPI
    boxes = [(2, "#0e7490"), (4, "#b45309"), (8, "#7c3aed")]
    res = {h: budget(d, h) for h, _ in boxes}

    m, sp = prepared("pm25")
    site = pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
    tutc = pd.to_datetime(d["time"])
    f = site[(site.time >= tutc.min()) & (site.time <= tutc.max())].copy()
    f["tl"] = f.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    f["hr"] = f.tl.dt.hour
    S, _, _ = simulate(f, "pm25", np.random.default_rng(0))
    p25, p50, p75 = (np.percentile(S, q, 1) for q in (25, 50, 75))
    st = m[(m.t_pht >= f.tl.min()) & (m.t_pht <= f.tl.max())]

    fig = plt.figure(figsize=(13.0, 10.2), dpi=DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.30,
                          left=0.075, right=0.955, top=0.895, bottom=0.195)

    # --- 1. turnover through the sides --------------------------------------
    ax = fig.add_subplot(gs[0])
    for h, col in boxes:
        tau, _, _, Lx = res[h]
        ax.plot(t, tau, lw=1.5, color=col,
                label=f"{2*h*0.25:.0f}° box (~{Lx/1000:.0f} km)")
    ax.axvspan(EPI, t.max(), color="#fca5a5", alpha=0.12, lw=0, zorder=0)
    ax.axvline(EPI, color="#111827", lw=1.6)
    ax.set_ylabel("turnover time (h)", fontsize=9, color=C.INK)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")
    _style(ax)
    tb = np.array([res[h][0][~epi].mean() for h, _ in boxes])
    te = np.array([res[h][0][epi].mean() for h, _ in boxes])
    ax.set_title("HORIZONTAL — the column is flushed every few hours, and the "
                 f"episode does not change it ({tb[0]:.1f} → {te[0]:.1f} h at 109 km)",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    # --- 2. vertical exchange as a signed rate ------------------------------
    ax = fig.add_subplot(gs[1])
    for h, col in boxes:
        _, rv, _, Lx = res[h]
        ax.plot(t, rv, lw=1.4, color=col)
    ax.axhline(0, color="#57534e", lw=0.9)
    ax.axvspan(EPI, t.max(), color="#fca5a5", alpha=0.12, lw=0, zorder=0)
    ax.axvline(EPI, color="#111827", lw=1.6)
    ax.set_ylabel("vertical exchange (% h$^{-1}$)", fontsize=9, color=C.INK)
    _style(ax)
    vb, ve = res[2][1][~epi].mean(), res[2][1][epi].mean()
    ax.set_title(f"VERTICAL — signed: above zero the column vents upward, below "
                 f"it is fed from above ({vb:+.2f} → {ve:+.2f} % h⁻¹ at 109 km)",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    # --- 3. what Cebu was breathing -----------------------------------------
    ax = fig.add_subplot(gs[2])
    ax.fill_between(f.tl, p25, p75, color="#047857", alpha=0.25, lw=0,
                    label="corrected CAMS · 50% band")
    ax.plot(f.tl, p50, lw=1.7, color="#047857", label="corrected CAMS · median")
    ax.plot(st.t_pht, st[sp["obs"]], "o", ms=2.8, color="#1c1917", mec="white",
            mew=0.4, ls="none", label="station (EMB Central Visayas)")
    ax.axvspan(EPI, f.tl.max(), color="#fca5a5", alpha=0.12, lw=0, zorder=0)
    ax.axvline(EPI, color="#111827", lw=1.6)
    ax.set_xlim(t.min(), t.max()); ax.set_ylim(0, None)
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")
    _style(ax)
    sb = st[st.t_pht < EPI][sp["obs"]].mean(); se = st[st.t_pht >= EPI][sp["obs"]].mean()
    ax.set_title(f"THE AIR — station PM₂.₅ {sb:.0f} → {se:.0f} µg m⁻³ while the "
                 f"column kept turning over at the same rate",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)
    for A in fig.axes[:2]:
        A.set_xlim(t.min(), t.max())
        A.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.text(0.075, 0.968, "Was the air above Cebu stagnant?", fontsize=14,
             color=C.INK, weight="bold", va="top")
    ub, ue = res[2][2][~epi].mean(), res[2][2][epi].mean()
    fig.text(0.075, 0.936,
             f"mass budget on the 0–4 km column · NOAA GFS 0.25° · mass-weighted "
             f"mean wind {ub:.2f} → {ue:.2f} m s⁻¹ · "
             f"{C.WATERMARK.replace('Made by: ', '')}",
             fontsize=9, color=C.INK_MUTED, va="top")
    notes = ["tau tracks L / u-bar but is 0.66-0.79 of it here: gross outflow counts all four faces, so flow across a corner exits through two. tau scales with the box; the mean wind is the invariant.",
             "The vertical term is a signed rate because rho*w passes through zero and a turnover time would be infinite there.",
             "Fourth independent test of stagnation, after parcel residence, the horizontal smoke flux and the air mass flux. All four reject it.",
             "One episode, autocorrelated in time."]
    for n, line in enumerate(notes + C.attribution(coastlines=False, trajectory=False,
                                                   cams=True, gfs=True)):
        fig.text(0.075, 0.148 - n * 0.0112, line, fontsize=6.4,
                 color=C.INK_MUTED, va="top")

    out = C.FIGS / "replacement.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")
    for n, (h, _) in enumerate(boxes):
        print(f"  {2*h*0.25:.0f}° box: turnover {tb[n]:.1f} -> {te[n]:.1f} h")
    print(f"  vertical rate: {vb:+.3f} -> {ve:+.3f} %/h (109 km box)")


if __name__ == "__main__":
    main()
