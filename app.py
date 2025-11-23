import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tempfile
import zipfile
import os

st.set_page_config(page_title="GeoHeatmap KMZ", layout="centered")

st.title("GeoHeatmap → KMZ (No-Code Tool)")
st.write("Upload file CSV (lat, lon) → sinh heatmap KMZ để mở bằng Google Earth.")

def generate_heatmap_kmz_simple(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    grid_size: int = 512,
    levels: int = 5,
) -> str:
    lats = df[lat_col].to_numpy()
    lons = df[lon_col].to_numpy()
    weights = np.ones_like(lats, dtype=float)

    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    if min_lat == max_lat:
        max_lat += 0.0001
    if min_lon == max_lon:
        max_lon += 0.0001

    grid = np.zeros((grid_size, grid_size), dtype=float)
    x = ((lons - min_lon) / (max_lon - min_lon) * (grid_size - 1)).astype(int)
    y = ((max_lat - lats) / (max_lat - min_lat) * (grid_size - 1)).astype(int)

    np.add.at(grid, (y, x), weights)

    max_val = grid.max() or 1.0
    grid_norm = (grid / max_val * 255).astype(np.uint8)

    bins = np.linspace(0, 255, levels + 1)
    idx = np.digitize(grid_norm, bins, right=True)

    colors = np.array(
        [
            [0, 0, 0, 0],
            [255, 255, 0, 80],
            [255, 165, 0, 140],
            [255, 69, 0, 180],
            [255, 0, 0, 220],
            [255, 0, 0, 220],
        ],
        dtype=np.uint8,
    )

    rgba = colors[idx]
    img = Image.fromarray(rgba, mode="RGBA")

    tmp_dir = tempfile.mkdtemp()
    img_path = os.path.join(tmp_dir, "heatmap.png")
    img.save(img_path)

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Heatmap</name>
    <GroundOverlay>
      <name>Heatmap</name>
      <Icon>
        <href>heatmap.png</href>
      </Icon>
      <LatLonBox>
        <north>{max_lat}</north>
        <south>{min_lat}</south>
        <east>{max_lon}</east>
        <west>{min_lon}</west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>"""

    kml_path = os.path.join(tmp_dir, "doc.kml")
    with open(kml_path, "w", encoding="utf-8") as f:
        f.write(kml)

    kmz_path = os.path.join(tmp_dir, "heatmap.kmz")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(kml_path, "doc.kml")
        z.write(img_path, "heatmap.png")

    return kmz_path


uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Các cột phát hiện trong file:")
    st.write(list(df.columns))

    lat_col = st.selectbox("Chọn cột Latitude", df.columns, index=0)
    lon_col = st.selectbox("Chọn cột Longitude", df.columns, index=1 if len(df.columns) > 1 else 0)

    grid_size = st.slider("Độ phân giải heatmap (pixels)", 128, 2048, 512, step=128)
    levels = st.slider("Số mức màu (3–10)", 3, 10, 5)

    if st.button("Tạo KMZ"):
        with st.spinner("Đang xử lý dữ liệu và tạo heatmap..."):
            kmz_path = generate_heatmap_kmz_simple(
                df=df,
                lat_col=lat_col,
                lon_col=lon_col,
                grid_size=grid_size,
                levels=levels,
            )
            st.success("Tạo KMZ thành công!")

            with open(kmz_path, "rb") as f:
                st.download_button(
                    label="Download file heatmap.kmz",
                    data=f,
                    file_name="heatmap.kmz",
                    mime="application/vnd.google-earth.kmz",
                )
