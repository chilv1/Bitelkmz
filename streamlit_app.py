import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import tempfile
import os
import zipfile
import simplekml

# ==================================================
# UI
# ==================================================
st.set_page_config(page_title="GeoHeatmap Generator", layout="centered")

st.title("GeoHeatmap Generator")
st.write("Visualize telecom density CSV data on Map & export to KMZ")

st.subheader("Settings")
GRID_RES = st.number_input("Grid Resolution (px)", value=2000)
RADIUS = st.number_input("Blur Radius (sigma)", value=30)
THRESHOLD_RATIO = st.number_input("Threshold Ratio (0–1)", value=0.3)

uploaded_files = st.file_uploader("Upload CSV Files", accept_multiple_files=True)

if st.button("Generate KMZ"):

    if not uploaded_files:
        st.error("No files uploaded")
        st.stop()

    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f, sep=None, engine="python")
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    # auto detect columns
    lat_cols = [c for c in df_all.columns if "lat" in c.lower()]
    lon_cols = [c for c in df_all.columns if "lon" in c.lower()]
    op_cols  = [c for c in df_all.columns if "carrier" in c.lower()]

    if not lat_cols or not lon_cols or not op_cols:
        st.error("Missing lat/lon/operator column")
        st.stop()

    LAT_COL = lat_cols[0]
    LON_COL = lon_cols[0]
    OPERATOR_COL = op_cols[0]

    df_all[LAT_COL] = pd.to_numeric(df_all[LAT_COL], errors="coerce")
    df_all[LON_COL] = pd.to_numeric(df_all[LON_COL], errors="coerce")
    df_all = df_all.dropna(subset=[LAT_COL, LON_COL])

    # compute bounds
    lon = df_all[LON_COL].to_numpy()
    lat = df_all[LAT_COL].to_numpy()
    xmin, xmax = lon.min(), lon.max()
    ymin, ymax = lat.min(), lat.max()

    dx = xmax - xmin
    dy = ymax - ymin

    # expand bounds
    xmin -= dx * 0.02
    xmax += dx * 0.02
    ymin -= dy * 0.02
    ymax += dy * 0.02

    # color per operator
    OPERATOR_COLORS = {
        "Entel": "#28FF00",     # neon green
        "Claro": "#FF0000",     # red
        "Movistar": "#00FFF6",  # cyan
        "Bitel": "#FFA500",     # orange
    }

    def build_heatmap(df):
        lon = df[LON_COL].to_numpy()
        lat = df[LAT_COL].to_numpy()

        # normalize
        xn = (lon - xmin) / (xmax - xmin + 1e-9)
        yn = (lat - ymin) / (ymax - ymin + 1e-9)

        xi = np.clip((xn * (GRID_RES - 1)).astype(int), 0, GRID_RES - 1)
        yi = np.clip(((1 - yn) * (GRID_RES - 1)).astype(int), 0, GRID_RES - 1)

        grid = np.zeros((GRID_RES, GRID_RES), dtype=float)
        np.add.at(grid, (yi, xi), 1)

        heat = gaussian_filter(grid, sigma=RADIUS)

        # adaptive max
        maxh = np.nanpercentile(heat, 99.5)
        heat = np.clip(heat / maxh, 0, 1)

        return heat

    tmpdir = tempfile.mkdtemp()
    layers = {}

    for op, hex_color in OPERATOR_COLORS.items():
        df_op = df_all[df_all[OPERATOR_COL].str.lower() == op]

        if df_op.empty:
            continue

        st.write(f"Processing {op}...")

        H = build_heatmap(df_op)

        # convert hex to RGBA
        c = tuple(int(hex_color.lstrip("#")[i:i+2], 16)/255 for i in (0,2,4))
        rgba = np.zeros((GRID_RES, GRID_RES, 4))
        rgba[...,0] = c[0]
        rgba[...,1] = c[1]
        rgba[...,2] = c[2]
        rgba[...,3] = H * 0.8

        png_path = os.path.join(tmpdir, f"{op}.png")
        plt.imsave(png_path, rgba)
        layers[op] = png_path

    # build KMZ
    kmz_path = os.path.join(tmpdir, "operators_heatmap.kmz")
    kml_path = os.path.join(tmpdir, "doc.kml")
    kml = simplekml.Kml()

    for op, png in layers.items():
        ground = kml.newgroundoverlay(name=op)
        ground.icon.href = os.path.basename(png)
        ground.latlonbox.north = ymax
        ground.latlonbox.south = ymin
        ground.latlonbox.east = xmax
        ground.latlonbox.west = xmin

    kml.save(kml_path)

    with zipfile.ZipFile(kmz_path, "w") as z:
        z.write(kml_path, "doc.kml")
        for op, png in layers.items():
            z.write(png, os.path.basename(png))

    with open(kmz_path, "rb") as f:
        st.download_button("Download KMZ", data=f, file_name="operators_heatmap.kmz")
