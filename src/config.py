"""Shared configuration.

The domain is deliberately much larger than Cebu: a transported plume is only
visible if the upwind fetch (Borneo, Sulawesi, Sumatra) is inside the box.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "figs"
FRAMES = FIGS / "frames"
for _d in (DATA, FIGS, FRAMES):
    _d.mkdir(parents=True, exist_ok=True)

CEBU = dict(name="Cebu City", lat=10.32, lon=123.90)
BBOX = dict(south=-6.0, north=16.0, west=104.0, east=129.0)
BBOX_LOCAL = dict(south=7.0, north=13.5, west=120.5, east=127.0)

GRID_STEP = 1.0          # CAMS global is ~0.4 deg, so 1.0 subsamples it
CHUNK = 25               # coordinates per Open-Meteo request
PAST_DAYS = 5
FORECAST_DAYS = 1        # today's hours live here; frames trim at the clock

# --- sources -----------------------------------------------------------------
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_URL = "https://api.open-meteo.com/v1/forecast"

AQ_VARS = ["aerosol_optical_depth", "pm2_5", "pm10", "carbon_monoxide",
           "nitrogen_dioxide", "sulphur_dioxide", "dust", "us_aqi"]
MET_VARS = ["wind_speed_850hPa", "wind_direction_850hPa", "boundary_layer_height",
            "wind_speed_10m", "wind_direction_10m", "precipitation"]
VIS_VARS = ["visibility"]   # own request, own model -- see VIS_MODEL

# Pinned: best_match is a moving target and silently changes the trajectories.
MET_MODEL, MET_MODEL_LABEL = "icon_seamless", "DWD ICON"
# Exception. Only GFS names visibility, and at Cebu it sits flat at its clear-air
# ceiling (inert to smoke). best_match responds (4-45 km) but hides its model.
VIS_MODEL, VIS_MODEL_LABEL = "best_match", "Open-Meteo best-match"

GIBS_URL = ("https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
            "&LAYERS={layer}&CRS=EPSG:4326&TIME={date}&BBOX={s},{w},{n},{e}"
            "&WRAP=day&FORMAT=image/jpeg&WIDTH={px}&HEIGHT={py}")
GIBS_LAYER = "VIIRS_NOAA20_CorrectedReflectance_TrueColor"
GIBS_WIDTH_PX = 1400

# Keyless files cover the last 24 h only. History needs a free MAP_KEY from
# https://firms.modaps.eosdis.nasa.gov/api/map_key/ in $FIRMS_MAP_KEY.
FIRMS_24H = {
    "VIIRS_SNPP": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
                  "suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_SouthEast_Asia_24h.csv",
    "VIIRS_NOAA20": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
                    "noaa-20-viirs-c2/csv/J1_VIIRS_C2_SouthEast_Asia_24h.csv",
}
FIRMS_AREA = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{area}/{days}"

# The only real measurements here; everything else is model output.
GROUND_OBS = [
    dict(time="2026-08-31T22:00", us_aqi=152, station="Talisay City (DENR-EMB)",
         source="SunStar Cebu, 1 Sep 2026"),
    dict(time="2026-09-01T04:00", us_aqi=172, station="Talisay City (DENR-EMB)",
         source="SunStar Cebu, 1 Sep 2026"),
]

# --- plotting ----------------------------------------------------------------
# One hue per variable, light -> dark. mask_below hides unremarkable values so
# the --basemap imagery reads through.
VARS = {
    "aerosol_optical_depth": dict(
        label="Aerosol optical depth (550 nm)", short="AOD", cmap="YlOrBr",
        vmin=0.0, vmax=2.5, log=False, fmt="{:.2f}", mask_below=0.3),
    "pm2_5": dict(
        label="PM$_{2.5}$ (µg m$^{-3}$)", short="PM2.5", cmap="RdPu",
        vmin=0.0, vmax=150.0, log=False, fmt="{:.0f}", mask_below=25.0),
    "carbon_monoxide": dict(
        label="Carbon monoxide (µg m$^{-3}$)", short="CO", cmap="BuPu",
        vmin=100.0, vmax=4000.0, log=True, fmt="{:.0f}", mask_below=250.0),
    "sulphur_dioxide": dict(
        label="Sulphur dioxide (µg m$^{-3}$)", short="SO2", cmap="PuBu",
        vmin=0.0, vmax=30.0, log=False, fmt="{:.1f}", mask_below=5.0),
    "nitrogen_dioxide": dict(
        label="Nitrogen dioxide (µg m$^{-3}$)", short="NO2", cmap="GnBu",
        vmin=0.0, vmax=20.0, log=False, fmt="{:.1f}", mask_below=4.0),
    "dust": dict(
        label="Dust (µg m$^{-3}$)", short="Dust", cmap="Oranges",
        vmin=0.0, vmax=50.0, log=False, fmt="{:.0f}", mask_below=5.0),
    # NOT a haze tracer here: 35 of its 41 km spread is diurnal (afternoon
    # convection), corr with AOD is only +0.19, and the daily minima are flat
    # through the episode. Kept for completeness; do not read it as smoke.
    "visibility": dict(
        label="Visibility (km)", short="Visibility", cmap="Blues_r",
        vmin=0.0, vmax=40.0, log=False, fmt="{:.1f}", scale=0.001,
        mask_above=25.0),
    # AQI is a status scale with defined categories, so it is drawn discrete
    # and always labelled -- never as a continuous ramp.
    "us_aqi": dict(
        label="US Air Quality Index (PM$_{2.5}$)", short="US AQI", cmap=None,
        vmin=0.0, vmax=300.0, log=False, fmt="{:.0f}",
        bands=[0, 50, 100, 150, 200, 300],
        band_colors=["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97"],
        band_names=["Good", "Moderate", "Unhealthy (sensitive)",
                    "Unhealthy", "Very unhealthy"],
        mask_below=101),
}

INK, INK_MUTED, SURFACE = "#1c1917", "#78716c", "#ffffff"
FIRE, TRAJ, RECEPTOR = "#dc2626", "#0f172a", "#0f172a"


def attribution(year: int | None = None, basemap: bool = False) -> list[str]:
    """Licence-required notices. Copernicus needs both the notice and the
    disclaimer: https://apps.ecmwf.int/datasets/licences/cams"""
    import datetime as _dt
    y = year or _dt.date.today().year
    imagery = "  ·  imagery: NASA Worldview/GIBS" if basemap else ""
    return [
        f"Contains modified Copernicus Atmosphere Monitoring Service information {y}"
        f"  ·  {MET_MODEL_LABEL} winds  ·  {VIS_MODEL_LABEL} visibility",
        f"VIIRS active fire: NASA LANCE/FIRMS{imagery}"
        f"  ·  served via Open-Meteo (CC BY 4.0)  ·  coastlines: Natural Earth",
        "Neither the European Commission nor ECMWF is responsible for any use "
        "that may be made of the information it contains.",
        "Modelled fields, not measurements  ·  single-level kinematic trajectory, "
        "synoptic scale only",
    ]
