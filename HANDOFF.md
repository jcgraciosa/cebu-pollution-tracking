# Particulate matter source tracing — handoff

Carried over from a conversation on 2026-09-01 in an unrelated project. Nothing here has been
run or verified against live services; it is a starting map, not a result. Items marked
**[check]** are ones I stated from background knowledge and that deserve confirmation before you
rely on them — release dates, latencies and product names in this field move.

---

## Open questions — answer these first

The recommendation changes substantially on each, so treat them as prerequisites rather than
details.

1. **Region.** Geostationary coverage and ground-station density vary enormously. Himawari covers
   Asia–Pacific, GOES the Americas, GEMS only Asia. Without a geostationary sensor overhead you
   lose plume tracking entirely.
2. **Episodic or chronic?** Tracing a specific haze event (a fire, a dust outbreak, a smelter
   upset) and apportioning chronic background are different problems with different tooling.
   Episodic favours trajectories and geostationary imagery; chronic favours receptor models and
   long records.
3. **Any ground data?** Even one PM2.5 monitor changes what is possible, because satellite gives
   column loading and you need a surface anchor. Check OpenAQ coverage for the area early.
4. **Latency requirement.** Stated as near-real-time or a few days' delay, which rules out some
   archive-quality products (see §5).

---

## 1. Two different questions

Do not conflate them; they need different methods.

- **What kind of source?** — traffic, biomass burning, coal, dust, sea salt, secondary aerosol.
- **Where did it come from?** — direction, distance, specific facility or region.

The strong approach is to answer both and cross them: a source *type* whose contribution peaks at
one bearing is a located source; one that is diffuse and correlates with long-range transport is
regional background.

## 2. Receptor models — "what kind of source"

These work from measured composition at the receptor, so they need speciated ground data. Listed
because they are the standard in this field, and because the satellite route below is partly a
substitute for them when ground data does not exist.

| method | needs | notes |
|---|---|---|
| **PMF** (positive matrix factorisation) | speciated concentrations + per-value uncertainties | the default. Source profiles *not* required — it finds them |
| **CMB** (chemical mass balance) | measured local source profiles | less interpretation, but only as good as the profiles |
| **ME-2 / SoFi** | as PMF, plus a priori constraints | for pushing a factor toward a known profile |

Rough data requirement for PMF: order 100+ samples and 20–30 species, with genuine per-observation
uncertainties rather than a blanket percentage.

Software: EPA PMF 5.0 (free, Windows GUI); EPA **ESAT**, a Python implementation **[check its
current status and name]**; SoFi Pro (Datalystica, commercial).

**What stays manual.** PMF returns unlabelled factors. Naming them is expert judgment, done by
recognising tracers — levoglucosan and K⁺ for biomass burning, V/Ni for heavy fuel oil, ¹⁴C for
fossil versus modern carbon, sulfate for secondary. Choosing the number of factors is also
yours, and rotational ambiguity is real, so bootstrap and displacement diagnostics matter.

## 3. Trajectories — "where from"

- **HYSPLIT** (NOAA), free, runs online via READY or locally on free GFS/GDAS met.
- **PSCF** (potential source contribution function) and **CWT** (concentration-weighted
  trajectory) turn a trajectory ensemble into a source-probability map.
- **openair** (R) is the most automatic entry point: `polarPlot` (concentration by wind speed and
  direction) often localises a point source in a single call; `trajCluster` handles trajectory
  clustering. `PySPLIT` and `splitr` are the scripting routes.

## 4. Satellite fingerprinting — source typing without ground data

**AOD alone will not identify a source.** The discrimination comes from combining trace gases:

| product | resolution | fingerprints |
|---|---|---|
| **Sentinel-5P / TROPOMI** NO₂, SO₂, CO, UV aerosol index | ~3.5 × 5.5 km, daily | NO₂ → combustion, traffic, power; SO₂ → coal, smelters, volcanoes; CO → biomass burning; UVAI → absorbing aerosol, separates dust and smoke from sulfate |
| **MODIS MAIAC** (MCD19A2) AOD | 1 km, daily | aerosol loading |
| **VIIRS / MODIS active fire** (FIRMS) | 375 m | fire as a source |
| **Himawari-9 / GOES-16/18** AOD | ~2 km, 10-minutely | plume *motion* |
| **Sentinel-2** | 10 m | individual stacks, flares, visible plumes |

Example readings: high NO₂ with low CO → traffic or power generation. High CO with fire pixels and
elevated UVAI → biomass burning. An SO₂ hotspot with no NO₂ → smelter or volcano.

**Two free reanalyses give a speciated split directly** — dust, sea salt, black carbon, organic
carbon, sulfate as separate fields: **CAMS** (ECMWF) and **MERRA-2** (NASA). Coarse (tens of km)
and model-derived rather than observed, but downloadable today and a reasonable prior for
interpreting the finer satellite fields.

## 5. Data availability by period

All of the following are free for their entire record. The constraint is when each record starts.

| dataset | record | cadence |
|---|---|---|
| ERA5 (met) | 1940– | hourly |
| MERRA-2 aerosol (speciated) | 1980– | hourly |
| Landsat | 1984– | 16-day |
| AERONET (ground AOD) | 1993– | site-dependent |
| ACAG / van Donkelaar surface PM2.5 | 1998– | monthly |
| MODIS, MAIAC AOD | 2000– | daily, 1 km |
| CAMS EAC4 reanalysis (speciated) | 2003– | 3-hourly |
| VIIRS AOD and fire | 2012– | daily |
| OpenAQ (ground PM) | ~2015– | varies by country |
| Sentinel-2 | 2015– | 5-day |
| Himawari-8/9 | 2015– | 10-minutely |
| GOES-16/17/18 | 2017– | 5–10-minutely |
| **Sentinel-5P TROPOMI** | **mid-2018–** | daily |
| GEMS (Asia only) | 2020– | hourly |

**[check] all start dates above.**

**2018 is the dividing line.** TROPOMI is what makes satellite source typing work. Before it,
OMI (2004–, ~13 × 24 km) and SCIAMACHY are far coarser — enough to separate a large coal plant
from a fire, not much finer.

## 6. Near-real-time

| product | latency **[check all]** | use |
|---|---|---|
| GOES, Himawari AOD | 10–15 min | watch the plume move |
| FIRMS active fire (NRT) | ~3 h; US/Canada ~1 min | fire detection |
| **Sentinel-5P NRTI** NO₂, SO₂, CO, AI | ~3 h | source typing — the key product |
| MODIS/VIIRS NRT AOD (LANCE) | ~3 h | loading |
| CAMS global forecast (speciated, PM2.5/PM10) | ~1 day + 5-day forecast | composition context |
| HYSPLIT on GFS | real-time | back-trajectories; also forward from a suspect source |
| OpenAQ | minutes to hours | validation |
| ERA5T (preliminary) | ~5 days | reanalysis-quality met |

**NRT endpoints differ from archive endpoints:**

- **NASA LANCE** — the NRT hub for MODIS, VIIRS, FIRMS. **Worldview** browses it same-day, no login.
- **Copernicus Data Space Ecosystem** — S5P as `NRTI` (~3 h) versus `OFFL` (days). Request NRTI explicitly.
- **AWS Open Data** — GOES and Himawari straight from S3, no registration. Least friction for geostationary.
- **ADS** (Atmosphere Data Store) — CAMS, free API key.
- **Google Earth Engine** — hosts MAIAC, S5P (including NRTI), FIRMS as collections; free for
  research and non-commercial use only. Ingestion lag can be ~1 day, so for true NRT go direct.

**Cost of going NRT:** MAIAC's 1 km AOD is not an NRT product — near-real-time AOD is the coarser
Dark Target / Deep Blue at ~10 km. NRT products also skip final calibration and ancillary data, so
they are noisier than the archive version of the same scene. Acceptable for attribution; not for
trend analysis.

## 7. A concrete pipeline to start from

1. Pull S5P NRTI NO₂ / SO₂ / CO / UVAI and FIRMS daily over the region of interest.
2. Flag episode days against OpenAQ ground PM, or CAMS PM2.5 where there is no ground station.
3. For each flagged day: HYSPLIT back-trajectories, overlay fire pixels, and pull the
   geostationary AOD loop for that window.
4. Classify the episode by its gas signature (§4).
5. If and when speciated ground data exists, add PMF and cross its factor time series against
   `polarPlot` bearings.

Steps 1–3 automate cleanly. Step 4 is where you stay in the loop — this is the "semi-automatic"
part, and it does not go away.

## 8. Limits to state up front

- **Satellite AOD is a column, not a surface concentration.** Converting to PM2.5 needs boundary
  layer height and humidity correction, and that step carries most of the error. The ACAG global
  PM2.5 product does it for you if you would rather not.
- **Polar orbiters give one overpass per day** (~10:30 and ~13:30 local), which aliases anything
  with a diurnal cycle. Geostationary is the fix where available.
- **Cloud cover removes days non-randomly** — often the humid days when secondary aerosol is worst.
- **Resolution limits attribution.** At S5P's ~5 km a single facility is blurred; street-level
  sources are not resolvable from space at all.
- Satellite alone usually gives source *type* and *direction*, not definitive attribution to a
  named emitter without ground truth or an emissions inventory.

## 9. Non-satellite open data worth having

- **EDGAR** — global emissions by sector, 1970– **[check current end year]**, annual. Gives the
  prior source map to test hypotheses against.
- **OpenAQ** — global ground PM measurements, free API. Validation. Check density for your area
  before planning around it.
- **GFED / GFAS** — fire emissions estimates, for converting fire counts into loadings.
- **AERONET** — ground AOD, for validating the satellite column.

## 10. Software

`openair` (R), HYSPLIT + `PySPLIT`/`splitr`, EPA PMF 5.0, EPA ESAT **[check]**, Google Earth
Engine Python API, `satpy` and the ESA Atmospheric Toolbox / HARP for S5P handling, `xarray` +
`cdsapi` for CAMS and ERA5.
