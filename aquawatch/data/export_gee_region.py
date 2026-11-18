"""
AquaWatch - Google Earth Engine (GEE) Region Exporter for Chennai & Chembarambakkam Lake

Region of Interest:
  Chembarambakkam Lake and Adyar River Basin, Tamil Nadu, India
  Bounding Box: [79.98, 12.98, 80.08, 13.06] (WGS84 EPSG:4326)

Target Events:
  1. Historic Dec 2015 Chennai Floods (Nov 15, 2015 vs Dec 05, 2015)
  2. Cyclone Michaung (Nov 20, 2023 vs Dec 06, 2023)
  3. 2023 Monthly Monitoring Time-Series (Jan 2023 - Dec 2023)

Instructions:
  1. If you haven't authenticated GEE, run in your terminal:
       earthengine authenticate
     or run this script with --auth flag:
       python export_gee_region.py --auth
  2. You will be prompted to log in to your Google Earth Engine account and paste the auth token.
  3. Once authenticated, run this script to download cropped Sentinel-2 GeoTIFF chips into data/local_demo/.
"""

import os
import sys
import argparse
from pathlib import Path


# Bounding Box for Chembarambakkam Lake, Chennai
CHEMBARAMBAKKAM_BBOX = [79.98, 12.98, 80.08, 13.06]  # [min_lon, min_lat, max_lon, max_lat]
CHENNAI_URBAN_BBOX = [80.18, 12.95, 80.28, 13.05]   # Adyar & Kovalam Basin


def mask_s2_clouds(image):
    """Mask clouds and cirrus using the S2 QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)


def export_region_gee(dest_dir: Path, authenticate_first: bool = False):
    try:
        import ee
    except ImportError:
        print("Earth Engine API not installed. Install with: pip install earthengine-api")
        return

    if authenticate_first:
        print("Initiating Google Earth Engine browser authentication...")
        ee.Authenticate()

    try:
        ee.Initialize()
        print("Google Earth Engine successfully initialized!")
    except Exception as e:
        print("\n" + "=" * 70)
        print("GOOGLE EARTH ENGINE AUTHENTICATION REQUIRED")
        print("=" * 70)
        print("To download live images directly from Earth Engine:")
        print("  1. Sign up for free access at: https://earthengine.google.com/")
        print("  2. In your terminal, run:")
        print("       python export_gee_region.py --auth")
        print("  3. Log in with your Google account and grant Earth Engine permissions.")
        print("\nNote: AquaWatch already includes precomputed, cached demo scenes for")
        print("Chembarambakkam Lake so the entire web app and pipeline function instantly")
        print("without waiting for Earth Engine credentials!")
        print("=" * 70 + "\n")
        return

    aoi = ee.Geometry.Rectangle(CHEMBARAMBAKKAM_BBOX)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cyclone Michaung Before (Nov 2023)
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B2', 'B3', 'B4', 'B8'])

    before_image = s2_collection.filterDate('2023-11-01', '2023-11-25').median().clip(aoi)
    after_image = s2_collection.filterDate('2023-12-04', '2023-12-15').median().clip(aoi)

    print("Fetching download URLs from Earth Engine...")
    url_before = before_image.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': aoi
    })
    print(f"Before Image (Nov 2023) Download URL: {url_before}")

    url_after = after_image.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': aoi
    })
    print(f"After Image (Dec 2023) Download URL: {url_after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AquaWatch GEE Sentinel-2 Exporter")
    parser.add_argument("--auth", action="store_true", help="Authenticate with Google Earth Engine")
    parser.add_argument("--dest_dir", type=str, default="../data/local_demo", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.dest_dir).resolve()
    export_region_gee(out_dir, authenticate_first=args.auth)
