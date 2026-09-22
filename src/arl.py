"""Reader for NOAA ARL packed meteorology -- the format HYSPLIT consumes.

Each record is a 50-byte ASCII label followed by NX*NY packed bytes. Packing is
a first difference: a byte holds the change from the previous grid point along
the row, and each row starts from the first value of the row above. So a field
is reconstructed by two cumulative sums, not by scaling each byte independently.

    from arl import ArlFile
    f = ArlFile("data/arl/20260918_gfs0p25")
    print(f.times[:3], f.levels[:8], f.vars_at(1))
    T = f.field("TEMP", level=1, t=0, box=dict(south=5, north=15,
                                               west=118, east=128))

Why bother: gfs0p25 carries 65 levels with 40-80 m spacing through the lowest
400 m, against the six levels and 1.6 km gap available from the forecast APIs.
That is the difference between resolving the trade inversion and missing it.
"""
from __future__ import annotations
import os
import numpy as np

LABEL = 50


def _lab(b: bytes) -> dict:
    s = b.decode("latin-1")
    return dict(yy=int(s[0:2]), mm=int(s[2:4]), dd=int(s[4:6]), hh=int(s[6:8]),
                ff=int(s[8:10]), ll=int(s[10:12]), grid=s[12:14],
                var=s[14:18].strip(), nexp=int(s[18:22]),
                prec=float(s[22:36]), var1=float(s[36:50]))


class ArlFile:
    """Index an ARL file and unpack fields on demand."""

    def __init__(self, path):
        self.path = str(path)
        self.fh = open(self.path, "rb")
        head = self.fh.read(LABEL + 108).decode("latin-1")
        h = head[LABEL:]
        self.model = h[0:4]
        # 12 reals of 7 chars: pole lat/lon, ref lat/lon (the grid increment for
        # a lat-lon grid), size, orient, tang, sync x/y, sync lat/lon, dummy
        r = [float(h[9 + 7 * i:16 + 7 * i]) for i in range(12)]
        self.dlat, self.dlon = r[2], r[3]
        self.sync_x, self.sync_y, self.sync_lat, self.sync_lon = r[7], r[8], r[9], r[10]
        # NX, NY, NZ are written as I3, so a 1440-point global grid comes back
        # as "440". Recover it from the increment and confirm against the file
        # size: the record length has to divide the file exactly.
        nx, ny, self.nz = int(h[93:96]), int(h[96:99]), int(h[99:102])
        self.kflag = int(h[102:104])           # 4 = hybrid sigma-pressure
        size = os.path.getsize(self.path)
        for cand in (nx, nx + 1000, nx + 2000, int(round(360 / self.dlon))):
            if cand > 0 and size % (LABEL + cand * ny) == 0:
                nx = cand
                break
        else:
            raise ValueError(f"cannot reconcile NX={nx} NY={ny} with {size} bytes")
        self.nx, self.ny = nx, ny
        self.reclen = LABEL + self.nx * self.ny
        self.lat = self.sync_lat + (np.arange(self.ny) - (self.sync_y - 1)) * self.dlat
        self.lon = self.sync_lon + (np.arange(self.nx) - (self.sync_x - 1)) * self.dlon
        self.lon = np.where(self.lon > 180, self.lon - 360, self.lon) \
            if self.lon.max() > 360 else self.lon
        self._index()

    def _index(self):
        """Walk the labels once; remember where every (time, level, var) sits."""
        size = os.path.getsize(self.path)
        n = size // self.reclen
        self.map, times, levels = {}, [], []
        for k in range(n):
            self.fh.seek(k * self.reclen)
            L = _lab(self.fh.read(LABEL))
            if L["var"] == "INDX":
                t = (L["yy"], L["mm"], L["dd"], L["hh"])
                if t not in times:
                    times.append(t)
                continue
            t = (L["yy"], L["mm"], L["dd"], L["hh"])
            if t not in times:
                times.append(t)
            self.map[(t, L["ll"], L["var"])] = k
            if L["ll"] not in levels:
                levels.append(L["ll"])
        self.times = times
        self.levels = sorted(levels)

    def vars_at(self, level):
        t = self.times[0]
        return sorted({v for (tt, ll, v) in self.map if tt == t and ll == level})

    def _slice(self, box):
        if box is None:
            return slice(None), slice(None)
        j = np.where((self.lat >= box["south"]) & (self.lat <= box["north"]))[0]
        lon = np.where(self.lon > 180, self.lon - 360, self.lon)
        i = np.where((lon >= box["west"]) & (lon <= box["east"]))[0]
        return slice(j[0], j[-1] + 1), slice(i[0], i[-1] + 1)

    def field(self, var, level, t=0, box=None):
        """Unpack one 2D field. Returns (values, lat, lon)."""
        key = (self.times[t] if isinstance(t, int) else t, level, var)
        if key not in self.map:
            raise KeyError(f"{key} not in {os.path.basename(self.path)}")
        k = self.map[key]
        self.fh.seek(k * self.reclen)
        L = _lab(self.fh.read(LABEL))
        raw = np.frombuffer(self.fh.read(self.nx * self.ny), dtype=np.uint8)
        d = (raw.astype("float32") - 127.0) / (2.0 ** (7 - L["nexp"]))
        d = d.reshape(self.ny, self.nx)
        # first value of each row accumulates down the first column
        starts = L["var1"] + np.cumsum(d[:, 0])
        out = starts[:, None] + np.cumsum(
            np.concatenate([np.zeros((self.ny, 1), "float32"), d[:, 1:]], axis=1), axis=1)
        js, is_ = self._slice(box)
        return out[js, is_].astype("float32"), self.lat[js], self.lon[is_]


def cube(paths, var, levels=None, box=None):
    """Stack one variable over files, levels and times -> (t, z, lat, lon)."""
    files = [ArlFile(p) for p in paths]
    levels = levels if levels is not None else [l for l in files[0].levels if l > 0]
    stamps, out = [], []
    for f in files:
        for ti, t in enumerate(f.times):
            if (t, levels[0], var) not in f.map:
                continue
            lay = [f.field(var, l, ti, box)[0] for l in levels]
            out.append(np.stack(lay))
            stamps.append(t)
    a, la, lo = files[0].field(var, levels[0], 0, box)
    return np.stack(out), stamps, np.array(levels), la, lo


if __name__ == "__main__":
    import sys
    f = ArlFile(sys.argv[1])
    print(f"{f.model}  {f.nx} x {f.ny} x {f.nz}   d={f.dlat}/{f.dlon}")
    print(f"lat {f.lat[0]:.2f}..{f.lat[-1]:.2f}  lon {f.lon[0]:.2f}..{f.lon[-1]:.2f}")
    print(f"times {len(f.times)}: {f.times[0]} .. {f.times[-1]}")
    print(f"levels {len(f.levels)}  surface vars {f.vars_at(0)}")
    print(f"upper vars {f.vars_at(1)}")
