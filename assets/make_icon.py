"""Generate a macOS app icon: a license plate with a mosaic effect over it,
on a blue-to-teal gradient squircle background."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = Path(__file__).parent
SCALE = 4
SIZE = 1024 * SCALE


def rounded_rect_mask(size, radius):
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return mask


def make_background():
    # Diagonal gradient, deep navy -> teal.
    c1 = np.array([18, 30, 56], dtype=np.float32)     # #121E38
    c2 = np.array([15, 148, 158], dtype=np.float32)    # #0F949E
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    t = (xs.astype(np.float32) + ys.astype(np.float32)) / (2 * SIZE)
    t = np.clip(t, 0, 1)[..., None]
    grad = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    bg = Image.fromarray(grad, mode="RGB").convert("RGBA")

    # Soft radial highlight near the top-left for depth.
    hl = Image.new("L", (SIZE, SIZE), 0)
    hd = ImageDraw.Draw(hl)
    hd.ellipse(
        [-SIZE * 0.35, -SIZE * 0.45, SIZE * 0.85, SIZE * 0.65], fill=70
    )
    hl = hl.filter(ImageFilter.GaussianBlur(SIZE * 0.12))
    white = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 255))
    bg = Image.composite(white, bg, hl)

    corner = int(SIZE * 0.225)
    mask = rounded_rect_mask((SIZE, SIZE), corner)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(bg, (0, 0), mask)
    return canvas


def make_plate_layer():
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    plate_w, plate_h = int(SIZE * 0.66), int(SIZE * 0.40)
    px = (SIZE - plate_w) // 2
    py = int(SIZE * 0.335)
    radius = int(plate_h * 0.16)

    # Drop shadow.
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    off = int(SIZE * 0.018)
    sd.rounded_rectangle(
        [px - off, py + off * 2, px + plate_w - off, py + plate_h + off * 2],
        radius=radius,
        fill=(0, 0, 0, 130),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(SIZE * 0.02))
    canvas = Image.alpha_composite(canvas, shadow)

    # Plate border (slightly larger rounded rect, mid-gray) + white face.
    border_pad = int(SIZE * 0.008)
    plate = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)
    pd.rounded_rectangle(
        [px - border_pad, py - border_pad, px + plate_w + border_pad, py + plate_h + border_pad],
        radius=radius + border_pad,
        fill=(150, 158, 168, 255),
    )
    pd.rounded_rectangle(
        [px, py, px + plate_w, py + plate_h],
        radius=radius,
        fill=(250, 250, 250, 255),
    )
    canvas = Image.alpha_composite(canvas, plate)

    # Bolt holes.
    bolts = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bolts)
    bolt_r = int(SIZE * 0.014)
    bolt_y = py + int(plate_h * 0.16)
    for bx in (px + int(plate_w * 0.10), px + plate_w - int(plate_w * 0.10)):
        bd.ellipse([bx - bolt_r, bolt_y - bolt_r, bx + bolt_r, bolt_y + bolt_r], fill=(190, 196, 204, 255))
        inner = int(bolt_r * 0.45)
        bd.ellipse([bx - inner, bolt_y - inner, bx + inner, bolt_y + inner], fill=(140, 148, 158, 255))
    canvas = Image.alpha_composite(canvas, bolts)

    # Small text-like marks in the top strip, hinting at plate lettering
    # without the icon depending on any real characters being legible.
    text = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    td = ImageDraw.Draw(text)
    strip_cy = py + int(plate_h * 0.16)
    bar_h = int(plate_h * 0.09)
    bar_y0, bar_y1 = strip_cy - bar_h // 2, strip_cy + bar_h // 2
    gap = int(plate_w * 0.035)
    widths = [0.10, 0.07, 0.16, 0.09, 0.09]
    cursor = px + int(plate_w * 0.24)
    for wfrac in widths:
        bw = int(plate_w * wfrac)
        td.rounded_rectangle(
            [cursor, bar_y0, cursor + bw, bar_y1], radius=bar_h // 2, fill=(70, 76, 88, 255)
        )
        cursor += bw + gap
    canvas = Image.alpha_composite(canvas, text)

    # Mosaic block overlay across the lower ~70% of the plate, masked to the
    # plate's rounded-rect shape so it never spills outside it.
    mosaic_top = py + int(plate_h * 0.30)
    mosaic = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    md = ImageDraw.Draw(mosaic)
    cols, rows = 7, 3
    tile_w = plate_w / cols
    tile_h = (plate_h - (mosaic_top - py)) / rows
    tones = [
        (58, 66, 82), (77, 87, 106), (96, 108, 128),
        (68, 100, 112), (48, 90, 98), (110, 120, 138),
        (85, 78, 100), (100, 95, 90),
    ]
    for r in range(rows):
        for c in range(cols):
            x0 = px + c * tile_w
            y0 = mosaic_top + r * tile_h
            # Deterministic pseudo-random pick so neighboring tiles rarely
            # repeat the same tone (avoids a checkerboard look).
            key = (r * 928371 + c * 137591 + r * c * 51 + 7) % len(tones)
            color = tones[key]
            md.rectangle([x0, y0, x0 + tile_w + 1, y0 + tile_h + 1], fill=(*color, 255))

    plate_mask = Image.new("L", (SIZE, SIZE), 0)
    pmd = ImageDraw.Draw(plate_mask)
    pmd.rounded_rectangle([px, py, px + plate_w, py + plate_h], radius=radius, fill=255)
    canvas.paste(mosaic, (0, 0), Image.composite(mosaic.split()[3], Image.new("L", (SIZE, SIZE), 0), plate_mask))

    return canvas


def main():
    bg = make_background()
    plate = make_plate_layer()
    final = Image.alpha_composite(bg, plate)
    final = final.resize((1024, 1024), Image.LANCZOS)
    final.save(OUT_DIR / "icon_1024.png")
    print("saved", OUT_DIR / "icon_1024.png")


if __name__ == "__main__":
    main()
