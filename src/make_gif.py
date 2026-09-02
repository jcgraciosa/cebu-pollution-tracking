"""Assemble rendered frames into a GIF, and an MP4 when ffmpeg is available
(far smaller for long loops).

    python src/make_gif.py --var aerosol_optical_depth --fps 2
    python src/make_gif.py --var us_aqi --basemap
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
import imageio.v2 as imageio
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config as C


def build_gif(frames, out, fps, scale):
    imgs = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        # quantise per frame so the palette follows the plume, not frame 1
        imgs.append(im.quantize(colors=192, method=Image.MEDIANCUT).convert("RGB"))
    imageio.mimsave(out, imgs, duration=1.0 / fps, loop=0, subrectangles=True)
    return out


def build_mp4(frames, out, fps):
    if not shutil.which("ffmpeg"):
        return None
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{f.resolve()}'\nduration {1/fps:.4f}\n"
                           for f in frames) + f"file '{frames[-1].resolve()}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
             "-r", str(fps), "-crf", "22", str(out)], check=True)
        return out
    except subprocess.CalledProcessError as e:                  # noqa: BLE001
        print(f"  ffmpeg failed: {e}", file=sys.stderr)
        return None
    finally:
        lst.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--var", default="aerosol_optical_depth", choices=list(C.VARS))
    p.add_argument("--local", action="store_true")
    p.add_argument("--basemap", action="store_true")
    p.add_argument("--fps", type=float, default=2, help="frames per second")
    p.add_argument("--scale", type=float, default=0.7, help="GIF downscale factor")
    p.add_argument("--every", type=int, default=1,
                   help="keep every Nth frame (for a small README loop)")
    p.add_argument("--out", default=None, help="output stem")
    a = p.parse_args()

    name = a.var + ("_local" if a.local else "") + ("_tc" if a.basemap else "")
    frames = sorted((C.FRAMES / name).glob("*.png"))[::a.every]
    if not frames:
        sys.exit(f"no frames in {C.FRAMES / name} -- run plot_maps.py first")

    stem = a.out or name
    gif = build_gif(frames, C.FIGS / f"{stem}.gif", a.fps, a.scale)
    print(f"{len(frames)} frames -> {gif}  ({gif.stat().st_size/1e6:.1f} MB)",
          file=sys.stderr)
    if mp4 := build_mp4(frames, C.FIGS / f"{stem}.mp4", a.fps):
        print(f"{'':>{len(str(len(frames)))}}          -> {mp4}"
              f"  ({mp4.stat().st_size/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
