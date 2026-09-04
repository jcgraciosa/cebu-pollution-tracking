"""Assemble rendered frames into a GIF, and an MP4 when ffmpeg is available
(far smaller for long loops).

    python src/make_gif.py --var aerosol_optical_depth --fps 2
    python src/make_gif.py --var us_aqi --basemap
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config as C


def build_gif(frames, out, fps, scale, width=None, hold_last=1.5):
    """Write the GIF, holding the final frame so the current hour is readable.

    Durations are per-frame MILLISECONDS. Passing seconds here silently writes
    duration=0 and the GIF then plays as fast as the viewer allows.
    """
    rgb = []
    for f in frames:
        im = Image.open(f).convert("RGB")
        if width:
            scale = width / im.width
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        rgb.append(im)

    # ONE palette for the whole sequence. Quantising per frame lets the same
    # value land on slightly different RGB each frame, which reads as flicker.
    w, h = rgb[0].size
    tile = max(1, len(rgb) // 24)
    sample = rgb[::tile]
    strip = Image.new("RGB", (w // 3, (h // 3) * len(sample)))
    for i, im in enumerate(sample):
        strip.paste(im.resize((w // 3, h // 3), Image.LANCZOS), (0, i * (h // 3)))
    master = strip.quantize(colors=256, method=Image.MEDIANCUT)
    imgs = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in rgb]
    per = [max(20, round(1000 / fps))] * len(imgs)
    per[-1] = max(per[-1], round(hold_last * 1000))
    # optimize=True lets PIL rebuild per-frame palettes, which reintroduces
    # the flicker the shared palette exists to prevent
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=per,
                 loop=0, optimize=False, disposal=2)
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
    p.add_argument("--fps", type=float, default=4, help="frames per second")
    p.add_argument("--hold", type=float, default=1.0,
                   help="seconds to hold the final (current) frame")
    p.add_argument("--width", type=int, default=900,
                   help="GIF pixel width; independent of the render DPI")
    p.add_argument("--scale", type=float, default=None,
                   help="downscale factor, used only if --width is 0")
    p.add_argument("--every", type=int, default=1,
                   help="keep every Nth frame (for a small README loop)")
    p.add_argument("--out", default=None, help="output stem")
    a = p.parse_args()

    name = a.var + ("_local" if a.local else "") + ("_tc" if a.basemap else "")
    frames = sorted((C.FRAMES / name).glob("*.png"))[::a.every]
    if not frames:
        sys.exit(f"no frames in {C.FRAMES / name} -- run plot_maps.py first")

    stem = a.out or name
    gif = build_gif(frames, C.FIGS / f"{stem}.gif", a.fps,
                    a.scale or 1.0, a.width or None, hold_last=a.hold)
    print(f"{len(frames)} frames -> {gif}  ({gif.stat().st_size/1e6:.1f} MB)",
          file=sys.stderr)
    if mp4 := build_mp4(frames, C.FIGS / f"{stem}.mp4", a.fps):
        print(f"{'':>{len(str(len(frames)))}}          -> {mp4}"
              f"  ({mp4.stat().st_size/1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
