import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import tempfile
import os
import zipfile
import simplekml

# ========================================
# UI
# ========================================
st.set_page_config(page_title="GeoHeatmap Generator", layout="centered")

st.title("GeoHeatmap Generator")
st.write("Visualize telecom density CSV data on Map & Export to KMZ")

# SETTINGS
st.subheader("SETTINGS")
grid_res = st.number_input("Grid Resolution (px)", value=2000, min_value=200, max_value=6000)
blur_sigma = st.number_input("Blur Radius (sigma)", value=30, min_value=0, max_value=200)
threshold_ratio = st.number_input("Threshold Ratio (0-1)", value=0.3, min_value=0.0, max_value=1.0)

# FILE INPUT
st.subheader("INPUT DATA")
files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

run = st.button("Run Generation")

# ========================================
# COMPUTE HEATMAP
# ========================================
def compute_heatmap(df, lat_col, lon_col, xmin, xmax, ymin, ymax, res, blur):
    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()

    xn = (lon - xmin) / (xmax - xmin + 1e-9)
    yn = (lat - ymin) / (ymax - ymin + 1e-9)

    xi = np.clip((xn * (res - 1)).astype(int), 0, res - 1)
    yi = np.clip(((1 - yn) * (res - 1)).astype(int), 0, res - 1)

    grid = np.zeros((res, res), dtype=float)
    for x, y in zip(xi, yi):
        grid[y, x] += 1

    if blur > 0:
        grid = gaussian_filter(grid, sigma=blur)

    return grid

# ========================================
# RUN
# ========================================
if run:
    if not files:
        st.error("Bạn chưa upload file CSV nào.")
        st.stop()

    dfs = []
    st.write("Processing input files...")

    for f in files:
        df = pd.read_csv(f, sep=None, engine="python")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Identify columns automatically
    possible_lat = [c for c in df_all.columns if "lat" in c.lower()]
    possible_lon = [c for c in df_all.columns if "lon" in c.lower()]

    if not possible_lat or not possible_lon:
        st.error("Không tìm thấy cột lat/lon trong file CSV.")
        st.stop()

    LAT_COL = possible_lat[0]
    LON_COL = possible_lon[0]

    df_all[LAT_COL] = pd.to_numeric(df_all[LAT_COL], errors="coerce")
    df_all[LON_COL] = pd.to_numeric(df_all[LON_COL], errors="coerce")
    df_all = df_all.dropna(subset=[LAT_COL, LON_COL])

    # Compute map bounds
    xmin, xmax = df_all[LON_COL].min(), df_all[LON_COL].max()
    ymin, ymax = df_all[LAT_COL].min(), df_all[LAT_COL].max()

    # Compute heatmap
    st.write("Generating heatmap...")
    H = compute_heatmap(df_all, LAT_COL, LON_COL, xmin, xmax, ymin, ymax, grid_res, blur_sigma)

    # Apply threshold
    mx = H.max()
    th = mx * threshold_ratio
    H[H < th] = 0

    # SAVE PNG + KMZ
    st.write("Creating KMZ overlay...")
    tmpdir = tempfile.mkdtemp()
    png_path = os.path.join(tmpdir, "heatmap.png")
    kml_path = os.path.join(tmpdir, "doc.kml")
    kmz_path = os.path.join(tmpdir, "heatmap.kmz")

    # Normalize & save PNG
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(H, cmap="hot", interpolation="nearest")
    plt.axis("off")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Create KML
    kml = simplekml.Kml()
    ground = kml.newgroundoverlay(name="Heatmap")
    ground.icon.href = "heatmap.png"
    ground.latlonbox.north = ymax
    ground.latlonbox.south = ymin
    ground.latlonbox.east = xmax
    ground.latlonbox.west = xmin
    kml.save(kml_path)

    # Pack KMZ
    with zipfile.ZipFile(kmz_path, "w") as z:
        z.write(kml_path, "doc.kml")
        z.write(png_path, "heatmap.png")

    # Offer download
    with open(kmz_path, "rb") as f:
        st.download_button("Download KMZ", f, file_name="heatmap.kmz", mime="application/vnd.google-earth.kmz")

    st.success("DONE!")
