"""Per-hour multiplicative bias correction of CAMS PM2.5 against the station.

CAMS is ~2.7x low overall AND anti-phased within the day, so one global factor
cannot fix it: the factors below range 1.9 (mid-morning) to 4.5 (19:00, when
CAMS bottoms out and the station peaks).

    python src/bias_correct.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from compare_ground import load_pairs

OBS, RAW, COR = "#b45309", "#7e22ce", "#047857"

def factors(df):
    return df.groupby("hr").apply(lambda g: g["PM 2.5"].mean() / g.pm2_5.mean(),
                                  include_groups=False)


def scores(o, p):
    e = p - o
    return np.sqrt((e ** 2).mean()), e.mean(), np.corrcoef(o, p)[0, 1]


def fetch_forecast():
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=dict(
        latitude=C.CEBU["lat"], longitude=C.CEBU["lon"],
        hourly="pm2_5,aerosol_optical_depth,carbon_monoxide", forecast_days=2,
        past_days=1, timezone="UTC"), timeout=60).json()["hourly"]
    f = pd.DataFrame(r); f["time"] = pd.to_datetime(f.time)
    f["t_pht"] = f.time + pd.Timedelta(hours=C.TZ_OFFSET_H); f["hr"] = f.t_pht.dt.hour
    return f


def main() -> None:
    m = load_pairs()
    m["t_pht"] = m.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    m["hr"] = m.t_pht.dt.hour
    m = m.dropna(subset=["PM 2.5", "pm2_5"])
    tr, te, cut = split(m)
    te = te.copy()
    f_tr, f_all = factors(tr), factors(m)
    g = tr["PM 2.5"].mean() / tr.pm2_5.mean()
    te["corr_hr"] = te.pm2_5 * te.hr.map(f_tr)
    te["corr_gl"] = te.pm2_5 * g

    fig = plt.figure(figsize=(12.5, 11.4), dpi=C.FIG_DPI)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.15], hspace=0.42,
                          left=0.075, right=0.955, top=0.905, bottom=0.085)

    ax = fig.add_subplot(gs[0])
    ax.bar(f_all.index, f_all.values, color=COR, alpha=0.8, width=0.75)
    ax.axhline(m["PM 2.5"].mean() / m.pm2_5.mean(), color=C.INK, ls="--", lw=1.2,
               label=f"single global factor ×{m['PM 2.5'].mean()/m.pm2_5.mean():.2f}")
    ax.set_xticks(range(0, 24, 2)); ax.set_xlim(-0.6, 23.6)
    ax.set_xlabel("hour of day (PHT)", fontsize=9, color=C.INK)
    ax.set_ylabel("station ÷ CAMS", fontsize=9, color=C.INK)
    ax.set_title(f"Per-hour correction factors, all {len(m)} paired hours "
                 f"(range ×{f_all.min():.1f} – ×{f_all.max():.1f})",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")

    ax = fig.add_subplot(gs[1])
    ax.plot(te.t_pht, te["PM 2.5"], lw=1.8, color=OBS, label="station (truth)")
    ax.plot(te.t_pht, te.pm2_5, lw=1.3, color=RAW, label="CAMS raw")
    ax.plot(te.t_pht, te.corr_hr, lw=1.5, color=COR, label="CAMS + per-hour factors")
    r0, b0, c0 = scores(te["PM 2.5"].values, te.pm2_5.values)
    r2, b2, c2 = scores(te["PM 2.5"].values, te.corr_hr.values)
    ax.set_title(f"Hold-out test · factors fitted on {len(tr)} h before "
                 f"{cut:%d %b}, applied to {len(te)} h after · "
                 f"RMSE {r0:.1f}→{r2:.1f}, r {c0:+.2f}→{c2:+.2f}",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9,
              facecolor="white", edgecolor="#e7e5e4")

    ax = fig.add_subplot(gs[2])
    fc = fetch_forecast()
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    fc = fc[(fc.time >= now - pd.Timedelta(hours=12)) & (fc.time <= now + pd.Timedelta(hours=24))]
    fc["corr"] = fc.pm2_5 * fc.hr.map(f_all)
    ax.plot(fc.t_pht, fc.pm2_5, lw=1.5, color=RAW, label="CAMS raw forecast")
    ax.plot(fc.t_pht, fc["corr"], lw=2.2, color=COR, label="bias-corrected forecast")
    ax.axvline(now + pd.Timedelta(hours=C.TZ_OFFSET_H), color=C.FIRE, lw=1.6)
    ax.axvspan(now + pd.Timedelta(hours=C.TZ_OFFSET_H), fc.t_pht.max(),
               color="#a8a29e", alpha=0.10)
    ax.annotate("forecast", (now + pd.Timedelta(hours=C.TZ_OFFSET_H + 1), ax.get_ylim()[1]),
                fontsize=8, color=C.INK_MUTED, va="top")
    nxt = fc[fc.time > now]
    ax.set_title(f"Live 24 h forecast · raw peak {nxt.pm2_5.max():.0f} → "
                 f"corrected peak {nxt['corr'].max():.0f} µg m$^{{-3}}$",
                 fontsize=9.5, color=C.INK_MUTED, loc="left", pad=4)
    ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9, color=C.INK)
    ax.set_xlabel("Philippine time", fontsize=9, color=C.INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, facecolor="white",
              edgecolor="#e7e5e4")

    for a in fig.axes:
        a.tick_params(labelsize=8, colors=C.INK_MUTED)
        a.grid(alpha=0.22, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#d6d3d1")

    fig.text(0.075, 0.975, "Bias-correcting CAMS PM$_{2.5}$ with the Cebu station",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.075, 0.943,
             f"factors from {m.t_pht.min():%d %b}–{m.t_pht.max():%d %b %Y}; "
             f"only corrects errors that repeat, and 18 days is {len(m)//24} samples per hour",
             fontsize=9.5, color=C.INK_MUTED, va="top")
    for n, line in enumerate(C.attribution(2026)):
        fig.text(0.075, 0.052 - n * 0.0135, line, fontsize=6.2, color=C.INK_MUTED, va="top")

    out = C.FIGS / "bias_correction.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  hold-out: raw RMSE {r0:.1f} bias {b0:+.1f} r {c0:+.2f}")
    print(f"            corrected RMSE {r2:.1f} bias {b2:+.1f} r {c2:+.2f}")
    print(f"  live forecast peak: raw {nxt.pm2_5.max():.1f} -> corrected {nxt['corr'].max():.1f}")


if __name__ == "__main__":
    main()
