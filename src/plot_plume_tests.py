"""Was the plume advected high and then brought down over Cebu? Four tests.

  1   onset timing -- an elevated plume mixing down puts column AOD ahead of
                      surface PM by hours
  2   column vs surface -- and leaves the AOD/PM ratio spiking, then collapsing
  3a  vertical velocity over Cebu -- descent would have to deliver it
  3b  the two horizontal components, same section, as signed colour rather than
      barbs: east-west and north-south read off a diverging scale with no
      convention to remember
  4   the flux -- concentration x speed, which is where the answer actually is

    python src/plot_plume_tests.py
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
from compare_ground import prepared

EPI = pd.Timestamp("2026-09-18 10:00")
DPI = 200
AOD_C, PM_C, CAMS_C, CO_C = "#b45309", "#1c1917", "#047857", "#7c3aed"


def _style(ax):
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.20, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d3d1")


def onset(v, t, base_end="2026-09-17", k=2.0):
    base = np.nanmedian(v[t < base_end])
    hit = np.where((v > k * base) & (t >= base_end))[0]
    return (t.iloc[hit[0]] if len(hit) else None), base


def main() -> None:
    s = pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
    s["t"] = s.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    m, sp = prepared("pm25")
    j = (s.merge(m[["time", "PM 2.5"]], on="time", how="inner")
           .dropna(subset=["PM 2.5"]))
    j = j[j.t >= "2026-09-12"].reset_index(drop=True)

    d = np.load(C.DATA / "wind3d.npz", allow_pickle=True)
    zl = (np.nanmean(d["z"], axis=(0, 2, 3)) if d["z"].ndim == 4
          else np.asarray(d["z"], float)); o = np.argsort(zl)
    zl, w = zl[o], d["w"][:, o]
    spd = d["spd"][:, o] / 3.6
    rad = np.deg2rad(d["dir"][:, o])
    uh, vh = -spd * np.sin(rad), -spd * np.cos(rad)
    tw = pd.to_datetime(d["time"]) + pd.Timedelta(hours=C.TZ_OFFSET_H)
    keep = np.isfinite(w).all(axis=(1, 2, 3))
    tw, w = tw[keep], w[keep]
    i = int(np.argmin(abs(d["lat"] - C.CEBU["lat"])))
    k = int(np.argmin(abs(d["lon"] - C.CEBU["lon"])))
    sl = (slice(None), slice(None), slice(i - 1, i + 2), slice(k - 1, k + 2))
    wc = np.nanmean(w[sl], axis=(2, 3)) * 100                       # cm/s
    uc = np.nanmean(uh[sl], axis=(2, 3))[keep]                      # m/s
    vc = np.nanmean(vh[sl], axis=(2, 3))[keep]
    sp = np.hypot(uc, vc)
    epi_w = tw >= EPI

    fig = plt.figure(figsize=(13.2, 16.4), dpi=DPI)
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1.25, 1.1, 0.85], hspace=0.52,
                          wspace=0.28, left=0.07, right=0.93, top=0.922, bottom=0.095)

    # --- TEST 1a: normalised onsets -----------------------------------------
    ax = fig.add_subplot(gs[0, :])
    series = [("column AOD", j.aerosol_optical_depth, AOD_C),
              ("CAMS surface PM2.5", j.pm2_5, CAMS_C),
              ("station PM2.5", j["PM 2.5"], PM_C),
              ("CAMS surface CO", j.carbon_monoxide, CO_C)]
    for name, v, col in series:
        o_t, base = onset(v.values, j.t)
        ax.plot(j.t, v / base, lw=1.6, color=col, label=name)
        if o_t is not None:
            ax.plot([o_t], [2.0], "v", ms=8, color=col, mec="white", mew=0.8, zorder=6)
    ax.axhline(2.0, color="#d6d3d1", lw=0.8, ls="--")
    ax.axvspan(EPI, j.t.max(), color="#fca5a5", alpha=0.14, lw=0, zorder=0)
    ax.set_yscale("log"); ax.set_ylabel("multiple of pre-episode median",
                                        fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, ncol=4, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")
    _style(ax)
    ax.set_title("TEST 1 — onsets. Markers show the 2x-baseline crossing. Surface "
                 "moves with the column, not after it", fontsize=9.5,
                 color=C.INK_MUTED, loc="left", pad=6)

    # --- TEST 1b: lagged correlation ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    a, p = j.aerosol_optical_depth.values, j["PM 2.5"].values
    lags = np.arange(-12, 25)
    rr = [np.corrcoef(a[:len(a) - l], p[l:])[0, 1] if l >= 0
          else np.corrcoef(a[-l:], p[:len(p) + l])[0, 1] for l in lags]
    ax.plot(lags, rr, lw=1.8, color=AOD_C)
    best = lags[int(np.argmax(rr))]
    ax.axvline(best, color=C.FIRE, lw=1.4)
    ax.annotate(f"peak at {best:+d} h", (best, max(rr)), xytext=(6, -14),
                textcoords="offset points", fontsize=8, color=C.FIRE)
    ax.axvline(0, color="#d6d3d1", lw=0.8)
    ax.set_xlabel("hours by which column AOD leads station PM$_{2.5}$",
                  fontsize=8.5, color=C.INK)
    ax.set_ylabel("correlation", fontsize=8.5, color=C.INK)
    _style(ax)
    ax.set_title("No lead. An elevated plume mixing down would peak "
                 "well to the right", fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)

    # --- TEST 2: column vs surface ------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ratio = j.aerosol_optical_depth / j["PM 2.5"]
    ax.plot(j.t, ratio, lw=1.4, color="#0e7490")
    ax.plot(j.t, ratio.rolling(24, center=True, min_periods=12).mean(), lw=2.4,
            color="#134e4a", label="24 h mean")
    ax.axvspan(EPI, j.t.max(), color="#fca5a5", alpha=0.14, lw=0, zorder=0)
    for lab, a0, b0 in (("14–17 Sep", "2026-09-14", "2026-09-18"),
                        ("19–20 Sep", "2026-09-19", "2026-09-21")):
        msk = (j.t >= a0) & (j.t < b0)
        ax.annotate(f"{lab}: {ratio[msk].median():.4f}",
                    (j.t[msk].mean(), ratio[msk].median()), fontsize=7.5,
                    color=C.INK, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#d6d3d1", alpha=0.9))
    ax.set_ylabel("column AOD ÷ surface PM$_{2.5}$", fontsize=8.5, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")
    _style(ax)
    ax.set_title("TEST 2 — a plume held aloft would spike this ratio, "
                 "then collapse", fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)

    # --- TEST 3: vertical velocity only -------------------------------------
    # This panel answers one question -- did anything come down -- so it carries
    # one field. Neither barbs nor contours belong here. Barbs encode direction
    # in MAP space while this axis is height, and contours would draw smooth
    # structure through only SIX samples in the vertical, the widest gap 1.65 km.
    # Blocks are honest about that: one block per level, nothing between.
    ax = fig.add_subplot(gs[2, :])
    tn = mdates.date2num(tw)
    lim = np.nanpercentile(abs(wc), 98)
    pm = ax.pcolormesh(tn, zl / 1000, wc.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       shading="nearest")
    ax.set_yticks(zl / 1000)
    ax.set_yticklabels([f"{z/1000:.1f}" for z in zl])
    for z in zl / 1000:                    # where the model actually has data
        ax.plot(-0.004, z, "_", ms=6, color="#111827", clip_on=False,
                transform=ax.get_yaxis_transform())
    ax.axvline(mdates.date2num(EPI), color="#111827", lw=1.8)
    ax.axhline(C.WIND3D_LID / 1000, color="#111827", lw=0.9, ls=":")
    ax.annotate("lid 2.5 km", (0.004, C.WIND3D_LID / 1000),
                xycoords=("axes fraction", "data"), fontsize=7, color="#111827",
                va="bottom")
    ax.annotate("episode", (mdates.date2num(EPI), 0.96),
                xycoords=("data", "axes fraction"), fontsize=8.5, color="#111827",
                ha="left", va="top", style="italic", xytext=(6, 0),
                textcoords="offset points")
    ax.set_xlim(tn[0], tn[-1])
    ax.set_ylabel("height (km)", fontsize=9, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("vertical velocity (cm s$^{-1}$)  ·  blue = sinking",
                 fontsize=8, color=C.INK)
    cb.ax.tick_params(labelsize=7, colors=C.INK_MUTED)
    _style(ax)

    b = wc[~epi_w].mean(0); e = wc[epi_w].mean(0)
    lo3 = zl <= 3000
    bs, es = np.nanmean(sp[~epi_w][:, lo3]), np.nanmean(sp[epi_w][:, lo3])
    bd = (np.degrees(np.arctan2(np.nanmean(uc[~epi_w][:, lo3]),
                                np.nanmean(vc[~epi_w][:, lo3]))) + 360) % 360
    ed = (np.degrees(np.arctan2(np.nanmean(uc[epi_w][:, lo3]),
                                np.nanmean(vc[epi_w][:, lo3]))) + 360) % 360
    jw = j.set_index("t")["PM 2.5"].reindex(tw, method="nearest")
    fb = float(np.nanmean(jw[~epi_w] * sp[~epi_w][:, lo3].mean(1)))
    fe = float(np.nanmean(jw[epi_w] * sp[epi_w][:, lo3].mean(1)))
    ax.set_title(
        f"TEST 3a — VERTICAL velocity, 83 km mean over Cebu, one block per "
        f"model level. Ascent stops but never reverses "
        f"({b[3]:+.2f} → {e[3]:+.2f} cm/s at 3.2 km), so nothing is pushed down",
        fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    # --- TEST 3b: the same section for the two horizontal components ---------
    # Signed colour, not barbs: east-west and north-south read straight off a
    # diverging scale, with no convention to remember and no direction encoded
    # in a plot axis that means height.
    for col, (comp, name, pos, neg) in enumerate(
            ((uc, "EAST–WEST (u)", "eastward", "westward"),
             (vc, "NORTH–SOUTH (v)", "northward", "southward"))):
        ax = fig.add_subplot(gs[3, col])
        cl = np.nanpercentile(abs(comp), 99)
        pmc = ax.pcolormesh(tn, zl / 1000, comp.T, cmap="PuOr_r",
                            vmin=-cl, vmax=cl, shading="nearest")
        ax.set_yticks(zl / 1000)
        ax.set_yticklabels([f"{z/1000:.1f}" for z in zl])
        ax.axvline(mdates.date2num(EPI), color="#111827", lw=1.6)
        ax.axhline(C.WIND3D_LID / 1000, color="#111827", lw=0.8, ls=":")
        ax.set_xlim(tn[0], tn[-1])
        ax.set_ylabel("height (km)", fontsize=8.5, color=C.INK)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        cbc = fig.colorbar(pmc, ax=ax, pad=0.015, fraction=0.04)
        cbc.set_label(f"m s$^{{-1}}$  ·  orange = {pos}, purple = {neg}",
                      fontsize=7.5, color=C.INK)
        cbc.ax.tick_params(labelsize=7, colors=C.INK_MUTED)
        _style(ax)
        bb = np.nanmean(comp[~epi_w][:, lo3]); ee = np.nanmean(comp[epi_w][:, lo3])
        ax.set_title(f"TEST 3b — {name}: {bb:+.1f} → {ee:+.1f} m/s below 3 km",
                     fontsize=9.5, color=C.INK_MUTED, loc="left", pad=5)

    # --- TEST 4: the flux argument, which is 1D and does not need height -----
    ax = fig.add_subplot(gs[4, :])
    spl = sp[:, lo3].mean(1)
    ax.plot(tw, spl, lw=1.6, color="#0e7490", label="wind speed below 3 km")
    ax.set_ylabel("wind speed (m s$^{-1}$)", fontsize=9, color="#0e7490")
    ax.set_ylim(0, None)
    a2 = ax.twinx()
    a2.plot(tw, jw * spl, lw=1.8, color=AOD_C, label="PM$_{2.5}$ flux")
    a2.set_ylabel("PM$_{2.5}$ flux (µg m$^{-2}$ s$^{-1}$)", fontsize=9, color=AOD_C)
    a2.tick_params(labelsize=8, colors=AOD_C)
    a2.set_ylim(0, None)
    a2.spines["top"].set_visible(False)
    ax.axvspan(EPI, tw.max(), color="#fca5a5", alpha=0.14, lw=0, zorder=0)
    ax.axvline(EPI, color="#b91c1c", lw=1.4)
    ax.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(ax)
    ax.set_title(f"TEST 4 — the flux. Wind {bs:.1f} → {es:.1f} m/s (toward "
                 f"{bd:.0f}° → {ed:.0f}° throughout) while the flux goes "
                 f"{fb:.0f} → {fe:.0f} µg m⁻² s⁻¹: the conveyor is unchanged, "
                 f"the load is not",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=6)

    fig.text(0.07, 0.978, "Was the plume brought down from aloft?", fontsize=14,
             color=C.INK, weight="bold", va="top")
    fig.text(0.07, 0.954,
             f"Three tests against one hypothesis · Cebu City · "
             f"wind column to {zl.max()/1000:.1f} km ({C.WIND3D_MODEL_LABEL}) · "
             f"{C.WATERMARK.replace('Made by: ', '')}",
             fontsize=9, color=C.INK_MUTED, va="top")
    notes = ["Verdict: no. Surface and column load together, the ratio barely moves, "
             "and the mid-troposphere goes to zero rather than sinking.",
             "A descending plume is a real mechanism — Saharan dust, pyroconvective "
             "smoke — it just does not match this event's timing.",
             "The wind column has only SIX levels (0.09, 0.78, 1.51, 3.16, 4.43, 5.89 km), so Test 3 is drawn as blocks: there is no data between them to contour.",
             "Resolved vertical motion only. Convective transport is unresolved at 0.25 deg and would not appear here."]
    for n, line in enumerate(notes + C.attribution(coastlines=False, trajectory=False)):
        fig.text(0.07, 0.068 - n * 0.0078, line, fontsize=6.4,
                 color=C.INK_MUTED, va="top")

    out = C.FIGS / "plume_tests.png"
    fig.savefig(out, facecolor=C.SURFACE); plt.close(fig)
    print(f"wrote {out}")
    print(f"  onset AOD {onset(j.aerosol_optical_depth.values, j.t)[0]}  "
          f"station PM {onset(j['PM 2.5'].values, j.t)[0]}  peak lag {best:+d} h")


if __name__ == "__main__":
    main()
