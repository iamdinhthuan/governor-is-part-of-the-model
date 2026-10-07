"""Graphical abstract for the JSA submission (run via uv with matplotlib + pillow).

Every number is taken from the manuscript (Table 6 / tab:ladder, Sec. 9 summary).
Output: <out>/04_graphical_abstract_JSA.tif (600 dpi, LZW) and a PDF twin.

Usage: python3 make_graphical_abstract.py <output_dir>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

HOST, GPU, GOV = "#DCEBF7", "#FBE3CC", "#E2F1E6"
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED = "#1A1A1A", "#4D4D4D"

W, H = 7.0, 2.7
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def panel(x, y, w, h, fc="white", ec="0.45", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))


def block(x, y, w, h, fc, label=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="0.35", lw=0.5))
    if label and w > 0.11:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.5, color=INK)


def arrow(p, q, color="0.25", lw=1.4, ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0))


def lanes(x0, y0, names):
    for i, n in enumerate(names):
        ax.text(x0 - 0.05, y0 - i * 0.26 + 0.09, n, ha="right", va="center", fontsize=7.2,
                color=MUTED)


# ---- left: sequential application --------------------------------------
lx, ly, lw_, lh = 0.10, 0.78, 2.62, 1.80
panel(lx, ly, lw_, lh)
ax.text(lx + 0.12, ly + lh - 0.17, "Sequential application", fontsize=9.5, fontweight="bold",
        va="center", color=INK)
ax.text(lx + 0.12, ly + lh - 0.40, "engine 4.2 ms back-to-back, 13.2 ms in the application",
        fontsize=7.0, va="center", color=MUTED)
tx, ty = lx + 0.50, ly + lh - 0.80
lanes(tx, ty, ["CPU", "GPU"])
x = tx
for _ in range(2):
    block(x, ty, 0.34, 0.18, HOST, "decode")
    block(x + 0.34, ty, 0.10, 0.18, HOST)
    x += 0.44
    block(x, ty - 0.26, 0.46, 0.18, GPU, "engine")
    x += 0.46
    block(x, ty, 0.10, 0.18, HOST)
    x += 0.10
ax.plot([tx, x], [ty - 0.31, ty - 0.31], color="0.6", lw=0.5)
ax.text(lx + 0.12, ly + 0.47, "GPU governor sees a low duty cycle:", fontsize=7.6,
        va="center", color=INK)
ax.text(lx + 0.12, ly + 0.25, "GPU held at its 306 MHz floor", fontsize=8.6, va="center",
        fontweight="bold", color=VERM)

# ---- middle: what changes ---------------------------------------------
mx = lx + lw_ + 0.08
arrow((mx, ly + lh / 2 + 0.08), (mx + 0.62, ly + lh / 2 + 0.08), color=INK)
ax.text(mx + 0.31, ly + lh / 2 + 0.47, "restructure\nhost only", ha="center", va="center",
        fontsize=7.6, color=INK, linespacing=1.15)
ax.text(mx + 0.31, ly + lh / 2 - 0.30, "same engine,\nidentical\npredictions", ha="center",
        va="center", fontsize=7.0, color=MUTED, linespacing=1.15)

# ---- right: restructured host pipeline ---------------------------------
rx = mx + 0.70
rw = W - 0.10 - rx
panel(rx, ly, rw, lh)
ax.text(rx + 0.12, ly + lh - 0.17, "Restructured host pipeline", fontsize=9.5,
        fontweight="bold", va="center", color=INK)
ax.text(rx + 0.12, ly + lh - 0.40, "prefetch, overlap, second decode worker", fontsize=7.6,
        va="center", color=MUTED)
tx2, ty2, dy, bh = rx + 0.62, ly + lh - 0.67, 0.23, 0.17
for i, n in enumerate(["CPU 1", "CPU 2", "GPU"]):
    ax.text(tx2 - 0.05, ty2 - i * dy + bh / 2, n, ha="right", va="center", fontsize=7.2,
            color=MUTED)
g = 0.19
for k in range(8):
    block(tx2 + k * g, ty2 - 2 * dy, g, bh, GPU)
for k in range(4):
    block(tx2 + k * 2 * g, ty2, 2 * g - 0.02, bh, HOST, "decode")
    if k < 3:
        block(tx2 + g + k * 2 * g, ty2 - dy, 2 * g - 0.02, bh, HOST, "decode")
ax.text(tx2 + 8 * g + 0.08, ty2 - 2 * dy + bh / 2, "GPU stays busy", ha="left",
        va="center", fontsize=7.2, color=MUTED)
ax.text(rx + 0.12, ly + 0.47, "the governor raises the GPU clock:", fontsize=7.6,
        va="center", color=INK)
ax.text(rx + 0.12, ly + 0.25, "1019 MHz under default governors", fontsize=8.6, va="center",
        fontweight="bold", color=GREEN)

# ---- result strip -------------------------------------------------------
sy, sh = 0.10, 0.58
panel(0.10, sy, W - 0.20, sh, fc=GOV, ec="0.55")
cells = [("50.0 \u2192 213.8 images/s", "Jetson Orin Nano Super,\ndefault governors"),
         ("2.4\u00d7 throughput", "over the best sequential\npipeline at locked max clocks"),
         ("150.7 \u2192 74.5 mJ/image", "COCO AP 0.4026\nunchanged"),
         ("Pi 5 + Hailo-8, QCM6490", "gains transfer; the mechanism\nmoves to the host")]
cw = (W - 0.20) / len(cells)
for i, (big, small) in enumerate(cells):
    cx = 0.10 + cw * (i + 0.5)
    ax.text(cx, sy + sh * 0.72, big, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=BLUE if i < 3 else INK)
    ax.text(cx, sy + sh * 0.31, small, ha="center", va="center", fontsize=6.7, color=MUTED,
            linespacing=1.15)
    if i:
        ax.plot([0.10 + cw * i] * 2, [sy + 0.09, sy + sh - 0.09], color="0.7", lw=0.6)

fig.savefig(OUT / "graphical_abstract.pdf")
png = OUT / "graphical_abstract_600dpi.png"
fig.savefig(png, dpi=600)

from PIL import Image  # noqa: E402

im = Image.open(png).convert("RGB")
im.save(OUT / "04_graphical_abstract_JSA.tif", compression="tiff_lzw", dpi=(600, 600))
png.unlink()
print("graphical abstract", im.size)
