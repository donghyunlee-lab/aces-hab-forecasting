#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-panel study-area map for the two-basin (Geum + Nakdong) cohort.

Shapefile layers (nationwide national/local river networks and provincial
boundaries) are read from the directory named by the ACES_GIS_DIR environment
variable. Station coordinates are the official automatic-network registry
positions served by the Water Environment Information System map API
(getAutoFeature, retrieved 2026-08-21); they replace the hand-entered Geum
coordinates used by the earlier single-basin map.
"""

import os
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker
from pyproj import Transformer
from shapely.geometry import Point, box as shapely_box

GIS = Path(os.environ["ACES_GIS_DIR"]) if "ACES_GIS_DIR" in os.environ else (
    Path(__file__).resolve().parents[1] / "gis_data"
)
OUT = Path(__file__).resolve().parent / "outputs"
FIGDIR = Path(os.environ.get("ACES_FIG_DIR", OUT))

# Official WEIS automatic-network coordinates (lon, lat; EPSG:4326).
GEUM = {
    "Gongju": (127.140, 36.462),
    "Daecheongho": (127.554, 36.430),
    "Gapcheon": (127.393, 36.447),
    "Buyeo": (126.967, 36.336),
    "Yongdamho": (127.485, 35.935),
}
NAKDONG = {
    "Seongseo": (128.492, 35.819),
    "Dasan": (128.403, 35.849),
    "Jinju": (128.162, 35.241),
    "Chilseo": (128.438, 35.388),
    "Jeokpo": (128.358, 35.613),
}
# Per-station label offsets in axis-fraction units (dx, dy), tuned to avoid
# overlap between nearby markers and river lines.
LABEL_OFFSETS = {
    "Dasan": (-0.10, 0.025),
    "Seongseo": (0.10, 0.012),
    "Daecheongho": (0.11, 0.018),
    "Gongju": (-0.075, 0.018),
    "Gapcheon": (0.01, 0.032),
}

PANELS = {
    "b": {"title": "Geum River basin", "sites": GEUM, "extent": (126.70, 35.66, 127.86, 36.74)},
    "c": {"title": "Nakdong River basin", "sites": NAKDONG, "extent": (127.86, 34.97, 129.02, 36.05)},
}

TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def project_extent(extent):
    x0, y0 = TO_3857.transform(extent[0], extent[1])
    x1, y1 = TO_3857.transform(extent[2], extent[3])
    return x0, y0, x1, y1


def sites_gdf(sites):
    frame = gpd.GeoDataFrame(
        [{"Site": name, "geometry": Point(lon, lat)} for name, (lon, lat) in sites.items()],
        crs="EPSG:4326",
    )
    return frame.to_crs("EPSG:3857")


def degree_formatters():
    def format_lon(x, _):
        lon, _lat = TO_4326.transform(x, 0)
        return f"{lon:.1f}°E"

    def format_lat(y, _):
        _lon, lat = TO_4326.transform(0, y)
        return f"{lat:.1f}°N"

    return mticker.FuncFormatter(format_lon), mticker.FuncFormatter(format_lat)


def draw_scalebar(ax, x0, y0, x1, y1, length_m=40000):
    sx = x0 + (x1 - x0) * 0.06
    sy = y0 + (y1 - y0) * 0.05
    height = (y1 - y0) * 0.012
    ax.add_patch(mpatches.Rectangle((sx, sy), length_m / 2, height, facecolor="black", zorder=13))
    ax.add_patch(
        mpatches.Rectangle(
            (sx + length_m / 2, sy), length_m / 2, height, facecolor="white", edgecolor="black", zorder=13
        )
    )
    label_y = sy + height + (y1 - y0) * 0.006
    for frac, text in ((0.0, "0"), (0.5, f"{length_m // 2000}"), (1.0, f"{length_m // 1000} km")):
        ax.text(sx + length_m * frac, label_y, text, ha="center", fontsize=8, zorder=14)


def draw_north(ax, x0, y0, x1, y1):
    ax_x = x1 - (x1 - x0) * 0.08
    tail_y = y1 - (y1 - y0) * 0.115
    head_y = tail_y + (y1 - y0) * 0.045
    ax.annotate(
        "",
        xy=(ax_x, head_y),
        xytext=(ax_x, tail_y),
        arrowprops=dict(facecolor="black", width=3, headwidth=9, headlength=10),
        zorder=15,
    )
    ax.text(
        ax_x,
        head_y + (y1 - y0) * 0.008,
        "N",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        zorder=15,
    )


def main():
    print("[1] Loading layers...")
    # Resolve by glob: the Hangul shapefile names are NFD-encoded on disk.
    sido = gpd.read_file(next((GIS / "admin_sido").glob("*.shp"))).to_crs("EPSG:3857")
    national = gpd.read_file(next((GIS / "rivers_national").glob("*.shp"))).to_crs("EPSG:3857")
    local = gpd.read_file(next((GIS / "rivers_local").glob("*.shp"))).to_crs("EPSG:3857")
    mainland = gpd.GeoDataFrame([1], geometry=[shapely_box(124.5, 33.8, 129.7, 38.7)], crs="EPSG:4326").to_crs(
        "EPSG:3857"
    )
    sido = sido[sido.geometry.intersects(mainland.geometry[0])]
    print(f"    sido {len(sido)}, national rivers {len(national)}, local rivers {len(local)}")

    fmt_lon, fmt_lat = degree_formatters()
    fig = plt.figure(figsize=(15.2, 6.6), dpi=200, facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[0.94, 1.0, 1.0], wspace=0.14)

    # Panel (a): peninsula locator with basin frames.
    print("[2] Panel (a): locator...")
    ax = fig.add_subplot(grid[0, 0])
    kx0, ky0 = TO_3857.transform(124.9, 33.9)
    kx1, ky1 = TO_3857.transform(129.8, 38.8)
    ax.set_xlim(kx0, kx1)
    ax.set_ylim(ky0, ky1)
    sido.plot(ax=ax, edgecolor="0.45", facecolor="0.94", linewidth=0.5, zorder=1)
    for key, color in (("b", "#B2182B"), ("c", "#2166AC")):
        x0, y0, x1, y1 = project_extent(PANELS[key]["extent"])
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.6, zorder=5
            )
        )
        ax.text(
            (x0 + x1) / 2,
            y1 + (ky1 - ky0) * 0.012,
            f"({key})",
            color=color,
            ha="center",
            fontsize=11,
            fontweight="bold",
            zorder=6,
        )
    for sites in (GEUM, NAKDONG):
        pts = sites_gdf(sites)
        ax.scatter(pts.geometry.x, pts.geometry.y, s=8, color="red", edgecolor="black", linewidth=0.3, zorder=7)
    ax.set_title("(a) Republic of Korea", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(fmt_lon)
    ax.yaxis.set_major_formatter(fmt_lat)
    ax.tick_params(labelsize=8)
    draw_north(ax, kx0, ky0, kx1, ky1)

    # Panels (b) and (c): basin detail.
    for key in ("b", "c"):
        panel = PANELS[key]
        print(f"[3] Panel ({key}): {panel['title']}...")
        ax = fig.add_subplot(grid[0, 1 if key == "b" else 2])
        x0, y0, x1, y1 = project_extent(panel["extent"])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        sido.plot(ax=ax, edgecolor="0.75", facecolor="white", linewidth=0.5, zorder=1)
        local.cx[x0:x1, y0:y1].plot(ax=ax, color="#4A7BA7", linewidth=0.45, alpha=0.55, zorder=2)
        national.cx[x0:x1, y0:y1].plot(ax=ax, color="#1F5A96", linewidth=1.7, alpha=0.9, zorder=3)
        pts = sites_gdf(panel["sites"])
        for _, row in pts.iterrows():
            ax.plot(
                row.geometry.x,
                row.geometry.y,
                marker="*",
                color="red",
                markersize=13,
                markeredgecolor="black",
                markeredgewidth=0.7,
                zorder=10,
            )
            dx, dy = LABEL_OFFSETS.get(row["Site"], (0.0, 0.018))
            ax.text(
                row.geometry.x + dx * (x1 - x0),
                row.geometry.y + dy * (y1 - y0),
                row["Site"],
                fontsize=8.5,
                fontweight="bold",
                ha="center",
                va="bottom",
                path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")],
                zorder=11,
            )
        ax.set_title(f"({key}) {panel['title']}", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(fmt_lon)
        ax.yaxis.set_major_formatter(fmt_lat)
        ax.tick_params(labelsize=8)
        draw_scalebar(ax, x0, y0, x1, y1)
        draw_north(ax, x0, y0, x1, y1)

    handles = [
        plt.Line2D([], [], color="#1F5A96", linewidth=1.7, label="National rivers"),
        plt.Line2D([], [], color="#4A7BA7", linewidth=0.8, alpha=0.7, label="Local rivers"),
        plt.Line2D([], [], marker="*", color="red", markeredgecolor="black", linestyle="", markersize=11, label="Monitoring station"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.04))

    OUT.mkdir(exist_ok=True)
    png = OUT / "study_area_two_basins.png"
    pdf = OUT / "study_area_two_basins.pdf"
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for src in (png, pdf):
        target = FIGDIR / src.name
        target.write_bytes(src.read_bytes())
    print(f"[4] Saved {png.name} / {pdf.name} and copied to {FIGDIR}")


if __name__ == "__main__":
    main()
