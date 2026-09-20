"""Pair the DENR-EMB station record against the modelled/remote-sensed fields.

The station file timestamps are LOCAL (PHT, UTC+8) -- confirmed by solar
radiation peaking at hour 11 and PM peaking at hour 19. Everything else in the
project is UTC, so the conversion happens once, here.

    python src/compare_ground.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config as C

STATION = C.DATA / "aqi-data-1.xlsx"

SPECIES = {
    "pm25": dict(obs="PM 2.5", mod="pm2_5", label="PM$_{2.5}$", colour="#b45309"),
    "pm10": dict(obs="PM 10", mod="pm10", label="PM$_{10}$", colour="#0e7490"),
}


def hour_factors(m, sp, upto=None):
    """Multiplicative station/CAMS ratio per hour of local day."""
    d = m if upto is None else m[m.t_pht < upto]
    d = d.dropna(subset=[sp["obs"], sp["mod"]])
    return d.groupby("hr").apply(
        lambda g: g[sp["obs"]].mean() / g[sp["mod"]].mean(), include_groups=False)


def split(m, frac=None):
    """Chronological train/test split by DAY fraction.

    A fixed number of days silently empties the test set once the model window
    slides past the station record, which is what happened when PAST_DAYS was
    10 against a station file ending 18 Sep.
    """
    frac = C.TRAIN_FRAC if frac is None else frac
    days = np.sort(m.t_pht.dt.floor("D").unique())
    if len(days) < 3:
        raise SystemExit(f"only {len(days)} overlapping days - extend PAST_DAYS "
                         "or refresh the station file")
    cut = days[max(1, int(round(len(days) * frac)))]
    return m[m.t_pht < cut], m[m.t_pht >= cut], pd.Timestamp(cut)


def prepared(species="pm25"):
    """Paired frame with local hour columns, ready for correction."""
    sp = SPECIES[species]
    m = load_pairs()
    m["t_pht"] = m.time + pd.Timedelta(hours=C.TZ_OFFSET_H)
    m["hr"] = m.t_pht.dt.hour
    out = m.dropna(subset=[sp["obs"], sp["mod"]]).copy()
    # the station file is uploaded by hand, so it goes stale silently and the
    # correction quietly starts describing a week that has already passed
    age = (pd.Timestamp.now(tz="UTC").tz_localize(None)
           + pd.Timedelta(hours=C.TZ_OFFSET_H)
           - out.t_pht.max()) / pd.Timedelta(days=1)
    if age > 3:
        print(f"  warning: station record ends {out.t_pht.max():%d %b %H:%M} PHT, "
              f"{age:.1f} days old", file=sys.stderr)
    return out, sp

# EPA PM2.5 breakpoints (2024 revision): (Clow, Chigh, Ilow, Ihigh)
AQI_BP = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
          (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]


def pm25_to_aqi(c):
    """EPA requires truncating the concentration to 1 dp first; without it a
    value in a breakpoint gap (35.45) matches no band."""
    if not np.isfinite(c):
        return np.nan
    c = np.floor(c * 10) / 10
    for clo, chi, ilo, ihi in AQI_BP:
        if clo <= c <= chi:
            return ilo + (ihi - ilo) * (c - clo) / (chi - clo)
    return 500.0 if c > AQI_BP[-1][1] else np.nan


def load_pairs() -> pd.DataFrame:
    df = pd.read_excel(STATION)
    df["t_local"] = pd.to_datetime(dict(year=df.YEAR, month=df.MONTH,
                                        day=df.DATE, hour=df.HOUR))
    df["time"] = df.t_local - pd.Timedelta(hours=C.TZ_OFFSET_H)
    site = pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
    m = df.merge(site, on="time", how="left").sort_values("time")
    m["aqi_hourly"] = m["PM 2.5"].map(pm25_to_aqi)
    # EPA defines the PM2.5 AQI on a 24 h mean; the hourly version is what
    # apps report, so keep both and lead with the proper one.
    m["aqi_24h"] = m["PM 2.5"].rolling(24, min_periods=18).mean().map(pm25_to_aqi)
    return m


def _panel(ax, x, y, xlabel, ylabel, colour, ylim=None):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ax.scatter(x, y, s=14, c=colour, alpha=0.45, linewidths=0, zorder=3)
    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color=C.INK, lw=1.6, zorder=4)
        r = np.corrcoef(x, y)[0, 1]
        rs = pd.Series(x).corr(pd.Series(y), method="spearman")
        ax.annotate(f"n = {len(x)}\npearson r = {r:+.2f}\nspearman = {rs:+.2f}\n"
                    f"slope = {b:.3g}",
                    (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
                    color=C.INK, bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                                           edgecolor="#d6d3d1", alpha=0.9))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=9, color=C.INK)
    ax.set_ylabel(ylabel, fontsize=9, color=C.INK)
    ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d6d3d1")


def main() -> None:
    m = load_pairs()
    good = m.dropna(subset=["PM 2.5", "aerosol_optical_depth", "carbon_monoxide"])
    g10 = m.dropna(subset=["PM 10", "aerosol_optical_depth", "carbon_monoxide"])
    a = m.dropna(subset=["aqi_24h", "aerosol_optical_depth", "carbon_monoxide"])

    def lim(v, pad=0.06):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return lo - (hi - lo) * pad, hi + (hi - lo) * pad

    # AQI dropped: it is a monotone transform of 24 h PM2.5 (spearman +1.000
    # against it here), so it duplicated row 1 rather than adding a measurement.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), dpi=C.FIG_DPI)
    rows = [
        (good, "PM 2.5", "Station PM$_{2.5}$ (µg m$^{-3}$)"),
        (g10,  "PM 10",  "Station PM$_{10}$ (µg m$^{-3}$)"),
    ]
    for r, (dfr, col, ylab) in enumerate(rows):
        yl = lim(dfr[col].values)
        _panel(axes[r, 0], dfr.aerosol_optical_depth.values, dfr[col].values,
               "CAMS AOD 550 nm (column)", ylab, "#b45309", ylim=yl)
        _panel(axes[r, 1], dfr.carbon_monoxide.values, dfr[col].values,
               "CAMS CO at 10 m (µg m$^{-3}$)", ylab, "#7e22ce", ylim=yl)

    fig.suptitle("Remotely sensed / modelled fields vs DENR-EMB station, Cebu",
                 fontsize=13.5, color=C.INK, weight="bold", x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.945,
             f"{good.time.min():%d %b} – {good.time.max():%d %b %Y} · "
             f"PM2.5 n={len(good)}, PM10 n={len(g10)} · "
             f"hour-for-hour · "
             f"station times converted PHT→UTC",
             fontsize=9.5, color=C.INK_MUTED)
    for n, line in enumerate(C.attribution(good.time.max().year)):
        fig.text(0.055, 0.055 - n * 0.0165, line, fontsize=6.2, color=C.INK_MUTED, va="top")
    fig.tight_layout(rect=[0.03, 0.055, 0.99, 0.94])
    out = C.FIGS / "ground_vs_model.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)

    print(f"{len(good)} paired hours  {good.time.min()} .. {good.time.max()} UTC")
    print(f"  station PM2.5 mean {good['PM 2.5'].mean():.1f}  "
          f"model PM2.5 mean {good.pm2_5.mean():.1f}  "
          f"-> model low by {good['PM 2.5'].mean()/good.pm2_5.mean():.1f}x")
    for col, lab in (("aerosol_optical_depth", "AOD"), ("carbon_monoxide", "CO"),
                     ("pm2_5", "model PM2.5"), ("us_aqi", "model AQI")):
        print(f"  {lab:12s} vs PM2.5 {good[col].corr(good['PM 2.5']):+.2f}"
              f" | PM10 {g10[col].corr(g10['PM 10']):+.2f}"
              f" | AQI24 {a[col].corr(a.aqi_24h):+.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
