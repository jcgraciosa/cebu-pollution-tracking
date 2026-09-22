"""Did the air linger over the Visayas during the episode? Five panels.

Shows the null result and the one suggestive signal side by side, with the
effective sample size on the scatters so the correlation cannot be read as
stronger than eight days of autocorrelated data can support.

    python src/residence.py && python src/plot_residence.py
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
from compare_ground import prepared, hour_factors

EPI = pd.Timestamp("2026-09-18 10:00")      # PHT; when the station PM took off
RES, PM, TOP, SIDE = "#0e7490", "#b45309", "#7c3aed", "#78716c"
DPI = 200


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.20, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")


def eff_n(x, y):
    """Effective sample size under AR(1), and the p value that follows."""
    from scipy import stats
    a = lambda v: float(np.corrcoef(v[:-1] - v.mean(), v[1:] - v.mean())[0, 1])
    rx, ry = a(x), a(y)
    n = len(x) * (1 - rx * ry) / (1 + rx * ry)
    r = float(np.corrcoef(x, y)[0, 1])
    t = r * np.sqrt(max(n - 2, 1) / max(1 - r ** 2, 1e-9))
    return r, n, 2 * (1 - stats.t.cdf(abs(t), max(n - 2, 1)))


def scatter(ax, x, y, xlab, ylab, colour):
    r, n, p = eff_n(x, y)
    ax.scatter(x, y, s=16, c=colour, alpha=0.55, linewidths=0)
    b, a0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 40)
    ax.plot(xs, a0 + b * xs, color=C.INK, lw=1.4)
    ax.annotate(f"r = {r:+.2f}\nn = {len(x)}, effective n = {n:.1f}\np = {p:.2f}",
                (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=7.5,
                color=C.INK, bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                       edgecolor="#d6d3d1", alpha=0.92))
    ax.set_xlabel(xlab, fontsize=8.5, color=C.INK)
    ax.set_ylabel(ylab, fontsize=8.5, color=C.INK)
    _style(ax)


def main() -> None:
    r = pd.read_csv(C.DATA / "residence.csv", parse_dates=["time"])
    r["t"] = r.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    m, sp = prepared("pm25")
    # Once the wind record runs out, only fast-exiting parcels have a finite
    # residence, so the median of the survivors collapses. That is censoring,
    # not lingering, and it lands exactly on the episode -- mask it everywhere.
    clean = r.censored < 0.02
    r.loc[~clean, ["residence_h", "residence_p25", "residence_p75"]] = np.nan
    j = r[clean].merge(m[["time", "PM 2.5"]], on="time",
                       how="inner").dropna(subset=["PM 2.5"])

    d = np.load(C.DATA / "wind3d.npz", allow_pickle=True)
    zl = np.nanmean(d["z"], axis=(0, 2, 3)); o = np.argsort(zl)
    zl, w = zl[o], d["w"][:, o]
    tw = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    keep = np.isfinite(w).all(axis=(1, 2, 3))
    tw, w = tw[keep], w[keep]
    e = tw >= EPI

    fig = plt.figure(figsize=(12.0, 11.2), dpi=DPI)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.42,
                          wspace=0.26, left=0.075, right=0.94, top=0.895, bottom=0.155)

    # --- 1. residence time, with the station on a twin axis -----------------
    ax = fig.add_subplot(gs[0, :])
    ax.axvspan(EPI, r.t.max(), color="#fca5a5", alpha=0.16, lw=0, zorder=0)
    ax.fill_between(r.t, r.residence_p25, r.residence_p75, color=RES, alpha=0.20, lw=0)
    ax.plot(r.t, r.residence_h, lw=1.8, color=RES, label="median residence")
    ax.plot(r.t, r.flush_h, lw=1.2, ls="--", color="#94a3b8", label="flush time L/V")
    ax.set_ylabel("hours inside the box", fontsize=9, color=RES)
    ax.set_ylim(0, None)
    a2 = ax.twinx()
    a2.plot(m[m.t_pht >= r.t.min()].t_pht, m[m.t_pht >= r.t.min()]["PM 2.5"],
            lw=1.4, color=PM, alpha=0.85)
    a2.set_ylabel("station PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9, color=PM)
    a2.tick_params(labelsize=8, colors=PM)
    for s in ("top",):
        a2.spines[s].set_visible(False)
    _style(ax)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.annotate("episode", (EPI + (r.t.max() - EPI) / 2, 0.94), fontsize=8.5,
                xycoords=("data", "axes fraction"), ha="center", color="#b91c1c",
                style="italic")
    t_cens = r.loc[~clean, "t"]
    if len(t_cens):
        ax.axvspan(t_cens.min(), r.t.max(), color="#d6d3d1", alpha=0.45, lw=0, zorder=0)
        ax.annotate("wind record ends —\nresidence censored", (t_cens.min(), 0.55),
                    xycoords=("data", "axes fraction"), fontsize=7, color=C.INK_MUTED,
                    ha="right", va="center")
    ax.set_title("Residence unchanged: 20.0 h before the episode, 19.0 h during",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    # --- 2. which face did parcels leave by? --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.stackplot(r.t, r.frac_side, r.frac_top, r.censored,
                 colors=[SIDE, TOP, "#e7e5e4"], alpha=0.85,
                 labels=["sides", "lid", "censored"])
    ax.axvline(EPI, color="#b91c1c", lw=1.3)
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of parcels", fontsize=8.5, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7, loc="lower left", ncol=3, framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")
    _style(ax)
    ax.set_title("Export through the lid stops: 19% → 0%", fontsize=9.5,
                 color=C.INK_MUTED, loc="left", pad=5)

    # --- 3. the vertical velocity profile behind it -------------------------
    ax = fig.add_subplot(gs[1, 1])
    for msk, lab, col in ((~e, "background 13–18 Sep", "#0e7490"),
                          (e, "episode 18–20 Sep", "#b91c1c")):
        prof = np.nanmean(w[msk], axis=(0, 2, 3))
        ax.plot(prof * 100, zl, "-o", ms=4, lw=1.8, color=col, label=lab)
    ax.axvline(0, color="#d6d3d1", lw=0.8)
    ax.axhline(C.WIND3D_LID, color="#94a3b8", lw=1.0, ls=":")
    ax.annotate("lid 2.5 km", (0.98, C.WIND3D_LID), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7, color="#94a3b8")
    ax.set_xlabel("mean vertical velocity (cm s$^{-1}$)", fontsize=8.5, color=C.INK)
    ax.set_ylabel("height (m)", fontsize=8.5, color=C.INK)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")
    _style(ax)
    ax.set_title("Ascent suppressed through the whole column", fontsize=9.5,
                 color=C.INK_MUTED, loc="left", pad=5)

    # --- 4 & 5. the two scatters: the signal, and the null ------------------
    scatter(fig.add_subplot(gs[2, 0]), j.frac_top.values, j["PM 2.5"].values,
            "fraction exiting through the lid", "station PM$_{2.5}$ (µg m$^{-3}$)", TOP)
    scatter(fig.add_subplot(gs[2, 1]), j.residence_h.values, j["PM 2.5"].values,
            "median residence (h)", "station PM$_{2.5}$ (µg m$^{-3}$)", RES)

    fig.text(0.075, 0.975, "Does the air linger over the Visayas?", fontsize=14,
             color=C.INK, weight="bold", va="top")
    fig.text(0.075, 0.945,
             f"3D parcels on {C.WIND3D_MODEL_LABEL} winds · box "
             f"{C.WIND3D_BOX['south']:.0f}–{C.WIND3D_BOX['north']:.0f}°N, "
             f"{C.WIND3D_BOX['west']:.0f}–{C.WIND3D_BOX['east']:.0f}°E × 0–2.5 km · "
             f"324 parcels every 2 h · {C.WATERMARK.replace('Made by: ', '')}",
             fontsize=9, color=C.INK_MUTED, va="top")
    notes = ["Residence is compared only over seed hours with <2% censoring; beyond those the wind record ends and the median collapses for a reason that is not physical.",
             "Bottom left is the only suggestive result and it is NOT significant: "
             "the lid fraction has AR(1) = 0.99, so 90 samples carry an effective n of 5.6.",
             "One episode, and Open-Meteo serves pressure levels for 8 days only. "
             "Treat as a hypothesis for a multi-episode HYSPLIT study, not a finding.",
             "Resolved motion only — no convection at 0.25°, no turbulent dispersion, "
             "no wet deposition. Residence of air, not of pollution."]
    for n, line in enumerate(notes + C.attribution(coastlines=False, trajectory=False)):
        fig.text(0.075, 0.112 - n * 0.0118, line, fontsize=6.4,
                 color=C.INK_MUTED, va="top")

    out = C.FIGS / "residence.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
