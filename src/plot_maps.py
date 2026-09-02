"""Render one map frame per time step: field + winds + fires + back-trajectory,
over an optional satellite basemap, with a receptor strip carrying a cursor.

    python src/plot_maps.py                                  # AOD, 3-hourly
    python src/plot_maps.py --var us_aqi --basemap
    python src/plot_maps.py --var carbon_monoxide --stride 1
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from trajectory import back_trajectory

PC = ccrs.PlateCarree()


def fetch_gibs(date, bbox):
    """Daily true-colour snapshot, cached on disk. Returns a path or None."""
    tag = f"{bbox['south']}_{bbox['west']}_{bbox['north']}_{bbox['east']}"
    out = C.DATA / "gibs" / f"{C.GIBS_LAYER}_{date}_{tag}.jpg"
    if out.exists() and out.stat().st_size > 5000:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    px = C.GIBS_WIDTH_PX
    py = int(round(px * (bbox["north"] - bbox["south"]) / (bbox["east"] - bbox["west"])))
    url = C.GIBS_URL.format(layer=C.GIBS_LAYER, date=date, px=px, py=py,
                            s=bbox["south"], w=bbox["west"],
                            n=bbox["north"], e=bbox["east"])
    try:
        import requests
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        if len(r.content) < 5000:        # GIBS answers a miss with a blank tile
            return None
        out.write_bytes(r.content)
        return out
    except Exception as e:                                  # noqa: BLE001
        print(f"  GIBS fetch failed for {date}: {e!r}", file=sys.stderr)
        return None


def _coastline(ax):
    """Natural Earth 50m if cartopy can fetch it, else the bundled 110m."""
    for res in ("50m", "110m"):
        try:
            ax.add_feature(cfeature.LAND.with_scale(res), facecolor="#f5f5f4", zorder=0)
            ax.add_feature(cfeature.OCEAN.with_scale(res), facecolor="#eef2f5", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale(res),
                           edgecolor="#a8a29e", linewidth=0.5, zorder=5)
            ax.add_feature(cfeature.BORDERS.with_scale(res),
                           edgecolor="#d6d3d1", linewidth=0.4, zorder=5)
            return
        except Exception as e:                              # noqa: BLE001
            print(f"  coastline {res} unavailable ({e!r})", file=sys.stderr)


def pick_cube(aq, met, var):
    """Composition lives in the aq cube, met variables in the met cube."""
    for cube in (aq, met):
        if var in cube:
            return cube
    raise KeyError(f"{var} is in neither cube; re-run download.py")


def load_all():
    aq = dict(np.load(C.DATA / "aq_grid.npz", allow_pickle=True))
    met = dict(np.load(C.DATA / "met_grid.npz", allow_pickle=True))
    try:
        fires = pd.read_csv(C.DATA / "fires.csv", parse_dates=["when"])
        if "confidence" in fires:
            conf = fires.confidence.astype(str).str.lower()
            fires = fires[~conf.isin(["l", "low"])]
    except FileNotFoundError:
        fires = pd.DataFrame(columns=["latitude", "longitude", "frp", "when"])
    site = (pd.read_csv(C.DATA / "cebu_timeseries.csv", parse_dates=["time"])
              .sort_values("time").reset_index(drop=True))
    return aq, met, fires, site


def _scaled(values, spec):
    return values * spec["scale"] if spec.get("scale") else values


def make_frame(k, aq, met, fires, site, var, spec, bbox, draw_traj, outdir,
               fire_window_h=12, quiver_every=2, show_forecast=False,
               basemap=False):
    times = aq["time"]
    tstamp = pd.Timestamp(str(times[k]))
    src = pick_cube(aq, met, var)
    lat, lon = src["lat"], src["lon"]
    field = _scaled(src[var][k].astype("float32"), spec)
    if basemap:
        if spec.get("mask_below") is not None:
            field = np.where(field < spec["mask_below"], np.nan, field)
        if spec.get("mask_above") is not None:
            field = np.where(field > spec["mask_above"], np.nan, field)

    fig = plt.figure(figsize=(10.5, 8.6), dpi=110)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.0], hspace=0.16,
                          left=0.06, right=0.93, top=0.905, bottom=0.15)
    ax = fig.add_subplot(gs[0], projection=PC)
    extent = [bbox["west"], bbox["east"], bbox["south"], bbox["north"]]
    ax.set_extent(extent, crs=PC)

    img = fetch_gibs(f"{tstamp:%Y-%m-%d}", bbox) if basemap else None
    if img is not None:
        ax.imshow(plt.imread(img), origin="upper", transform=PC, zorder=0, extent=extent)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       edgecolor="#fafaf9", linewidth=0.45, alpha=0.65, zorder=5)
    else:
        _coastline(ax)

    if spec.get("bands"):
        cmap = ListedColormap(spec["band_colors"])
        norm = BoundaryNorm(spec["bands"], cmap.N)
    else:
        cmap = spec["cmap"]
        norm = (LogNorm(vmin=max(spec["vmin"], 1e-3), vmax=spec["vmax"])
                if spec["log"] else Normalize(vmin=spec["vmin"], vmax=spec["vmax"]))
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=cmap, norm=norm,
                         shading="nearest", transform=PC, zorder=1,
                         alpha=0.62 if img is not None else 0.88)

    mi = int(np.clip(np.searchsorted(met["time"], times[k]), 0, met["time"].size - 1))
    spd = met["wind_speed_850hPa"][mi] / 3.6
    rad = np.deg2rad(met["wind_direction_850hPa"][mi])
    u, v = -spd * np.sin(rad), -spd * np.cos(rad)
    s = quiver_every
    ax.quiver(met["lon"][::s], met["lat"][::s], u[::s, ::s], v[::s, ::s], transform=PC,
              color="#57534e", alpha=0.55, scale=320, width=0.0022, zorder=4)

    if len(fires):
        span_h = (fires.when.max() - fires.when.min()) / pd.Timedelta(hours=1)
        if span_h < 48:
            # keyless FIRMS is one 24 h file: show it all and say it is not
            # time-resolved, rather than let fires blink on and off
            w, flabel = fires, f"VIIRS fire · 24 h snapshot, not time-resolved (n={len(fires)})"
        else:
            lo_, hi_ = (tstamp.tz_localize("UTC") - pd.Timedelta(hours=fire_window_h),
                        tstamp.tz_localize("UTC") + pd.Timedelta(hours=fire_window_h))
            w = fires[fires.when.between(lo_, hi_)]
            flabel = f"VIIRS fire ±{fire_window_h} h (n={len(w)})"
        if len(w):
            ax.scatter(w.longitude, w.latitude, s=np.clip(w.frp * 0.28, 1.5, 42),
                       c=C.FIRE, alpha=0.55, linewidths=0, transform=PC, zorder=6,
                       label=flabel)

    if draw_traj:
        _, tla, tlo = back_trajectory(met, C.CEBU["lat"], C.CEBU["lon"], mi, hours=120)
        ax.plot(tlo, tla, color="#ffffff", lw=3.4, alpha=0.9, transform=PC, zorder=7)
        ax.plot(tlo, tla, color=C.TRAJ, lw=1.8, transform=PC, zorder=8,
                label="850 hPa back-trajectory, 120 h")
        ax.scatter(tlo[::24], tla[::24], s=26, facecolor="#ffffff",
                   edgecolor=C.TRAJ, linewidths=1.4, transform=PC, zorder=9)

    ax.plot(C.CEBU["lon"], C.CEBU["lat"], marker="*", ms=17, color=C.RECEPTOR,
            markeredgecolor="white", markeredgewidth=1.2, transform=PC, zorder=10)
    ax.annotate(C.CEBU["name"], (C.CEBU["lon"], C.CEBU["lat"]), xytext=(7, 7),
                textcoords="offset points", transform=PC, zorder=10,
                fontsize=9, color=C.INK, weight="bold")

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="#d6d3d1", alpha=0.7)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": C.INK_MUTED}

    cb = fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.88, extend="max",
                      spacing="proportional")
    cb.set_label(spec["label"], fontsize=9, color=C.INK)
    cb.ax.tick_params(labelsize=8, colors=C.INK_MUTED)
    cb.outline.set_visible(False)
    if spec.get("band_names"):
        mids = [(spec["bands"][i] + spec["bands"][i + 1]) / 2
                for i in range(len(spec["band_names"]))]
        cb.set_ticks(mids)
        cb.set_ticklabels(spec["band_names"])
        cb.ax.tick_params(labelsize=7, length=0)

    ax.legend(loc="lower left", fontsize=8, framealpha=0.92,
              facecolor="white", edgecolor="#e7e5e4",
              borderpad=0.6).get_frame().set_linewidth(0.6)

    # figure-level, not ax.set_title: cartopy shrinks the GeoAxes to the map
    # aspect and an axes title can end up clipped above the visible frame
    fig.text(0.06, 0.963, f"{spec['short']} · {tstamp:%Y-%m-%d %H:%M} UTC",
             fontsize=13.5, color=C.INK, weight="bold", va="top")
    fig.text(0.06, 0.931, f"{tstamp + pd.Timedelta(hours=8):%a %d %b %H:%M} Philippine time",
             fontsize=9.5, color=C.INK_MUTED, va="top")

    # --- receptor strip -------------------------------------------------------
    axb = fig.add_subplot(gs[1])
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    st = site.dropna(subset=[var])
    if not show_forecast:
        st = st[st.time <= now]
    series = _scaled(st[var].to_numpy(dtype="float64"), spec)
    axb.plot(st.time, series, color="#78716c", lw=1.6)
    axb.fill_between(st.time, 0, series, color="#78716c", alpha=0.13)
    axb.axvline(tstamp, color=C.FIRE, lw=1.8)
    if len(st):
        # pin the unit: pandas parses to datetime64[us], so a bare astype("int64")
        # would be microseconds while Timestamp.value is nanoseconds
        xp = st.time.to_numpy().astype("datetime64[ns]").astype("int64")
        cur = float(np.interp(np.datetime64(tstamp, "ns").astype("int64"), xp, series))
        axb.plot([tstamp], [cur], "o", ms=6, color=C.FIRE, mec="white", mew=1.2)
        axb.annotate(f" {spec['fmt'].format(cur)}", (tstamp, cur), fontsize=9,
                     color=C.FIRE, weight="bold", va="center")
    if show_forecast and len(st):
        axb.axvspan(now, st.time.max(), color="#a8a29e", alpha=0.10)

    obs = [o for o in C.GROUND_OBS if var in o]
    if obs:
        ot = [pd.Timestamp(o["time"]) for o in obs]
        ov = [o[var] for o in obs]
        axb.plot(ot, ov, "D", ms=7, color="#0f172a", mec="white", mew=1.3, zorder=6,
                 label=f"measured · {obs[0]['station']}")
        for t_, v_ in zip(ot, ov):
            axb.annotate(f"{v_:.0f}", (t_, v_), xytext=(0, 9), ha="center",
                         textcoords="offset points", fontsize=8.5, weight="bold",
                         color="#0f172a")
        axb.legend(loc="upper left", fontsize=7.5, framealpha=0.9,
                   facecolor="white", edgecolor="#e7e5e4")
    axb.set_ylabel(spec["short"], fontsize=9, color=C.INK_MUTED)
    axb.set_title(f"{spec['label']} at {C.CEBU['name']}", fontsize=9,
                  color=C.INK_MUTED, loc="left", pad=4)
    axb.tick_params(labelsize=8, colors=C.INK_MUTED)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#d6d3d1")
    axb.margins(x=0.01)

    for n, line in enumerate(C.attribution(tstamp.year, basemap=img is not None)):
        fig.text(0.06, 0.062 - n * 0.0165, line, fontsize=6.2,
                 color=C.INK_MUTED, va="top")

    out = outdir / f"{var}_{k:04d}.png"
    fig.savefig(out, facecolor=C.SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--var", default="aerosol_optical_depth", choices=list(C.VARS))
    p.add_argument("--stride", type=int, default=3, help="hours between frames")
    p.add_argument("--local", action="store_true", help="zoom to the Visayas box")
    p.add_argument("--basemap", action="store_true",
                   help="NASA GIBS true-colour imagery under the field")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--include-forecast", action="store_true",
                   help="also render hours beyond now (off: this is retrospective)")
    a = p.parse_args()

    aq, met, fires, site = load_all()
    spec = dict(C.VARS[a.var])
    if a.vmax is not None:
        spec["vmax"] = a.vmax
    bbox = C.BBOX_LOCAL if a.local else C.BBOX
    outdir = C.FRAMES / (a.var + ("_local" if a.local else "")
                         + ("_tc" if a.basemap else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.png"):
        f.unlink()

    n_t = aq["time"].size
    if not a.include_forecast:
        # trim at the real clock: the API window slides, so an offset from the
        # first sample would drift
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
        n_t = max(1, int(np.searchsorted(aq["time"].astype("datetime64[ns]"),
                                         np.datetime64(now, "ns"), "right")))
        print(f"analysis window: {aq['time'][0]}Z .. {aq['time'][n_t-1]}Z "
              f"({aq['time'].size - n_t} future hours dropped)", file=sys.stderr)

    idx = list(range(0, n_t, a.stride))
    print(f"{len(idx)} frames -> {outdir}", file=sys.stderr)
    for n, k in enumerate(idx, 1):
        out = make_frame(k, aq, met, fires, site, a.var, spec, bbox,
                         not a.no_trajectory, outdir,
                         show_forecast=a.include_forecast, basemap=a.basemap)
        print(f"  [{n:3d}] {out.name}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
