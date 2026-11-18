"""
AquaWatch - High-Performance FastAPI Backend Server

Serves:
- Real test evaluation metrics (U-Net vs NDWI) and training curves
- Precomputed bi-temporal flood change maps for Chembarambakkam Lake / Chennai
- Monthly water extent time-series monitoring
- Qualitative comparison galleries and failure analysis
- Live inference endpoint for user-uploaded satellite GeoTIFF / PNG chips
"""

import os
import sys
import json
import base64
import io
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import tifffile
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Resolve directory paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
CACHED_DIR = RESULTS_DIR / "cached_regions"
ML_DIR = PROJECT_DIR / "ml"

# Add ml/ to sys.path so we can import dataset/model
sys.path.insert(0, str(ML_DIR))
try:
    from model import build_model
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

try:
    from weather_model import EfficientNetWeatherClassifier
    from gradcam import WeatherGradCAM, pil_to_base64_data_uri
    from weather_dataset import CLASS_NAMES, get_weather_transforms
    WEATHER_AVAILABLE = True
except Exception as e:
    print(f"[AquaWatch API] Weather import note: {e}")
    WEATHER_AVAILABLE = False

try:
    from rain_model import RainLSTM
    RAIN_AVAILABLE = True
except Exception as e:
    print(f"[AquaWatch API] Rain import note: {e}")
    RAIN_AVAILABLE = False

app = FastAPI(
    title="AquaWatch API",
    description="Backend API for Sentinel-2 Satellite Flood Segmentation and Change Detection",
    version="1.0.0"
)

# Enable CORS for frontend Vite development server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve images from results/
if RESULTS_DIR.exists():
    app.mount("/api/static", StaticFiles(directory=str(RESULTS_DIR)), name="static")

# Load U-Net model into memory if checkpoint exists
DEVICE = torch.device("cpu")
MODEL = None
if MODEL_AVAILABLE:
    ckpt_path = RESULTS_DIR / "checkpoints" / "best_model.pt"
    if ckpt_path.exists():
        try:
            MODEL = build_model("unet_resnet34", in_channels=4, classes=1, encoder_weights=None)
            checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)
            if "model_state_dict" in checkpoint:
                MODEL.load_state_dict(checkpoint["model_state_dict"])
            else:
                MODEL.load_state_dict(checkpoint)
            MODEL.eval()
            print(f"[AquaWatch API] Checkpoint loaded successfully from {ckpt_path}")
        except Exception as e:
            print(f"[AquaWatch API] Could not load checkpoint: {e}")

# Load Weather Classifier model & Grad-CAM engine
WEATHER_MODEL = None
WEATHER_CAM = None
if WEATHER_AVAILABLE:
    w_ckpt_path = RESULTS_DIR / "checkpoints" / "weather_efficientnet_b0.pt"
    if w_ckpt_path.exists():
        try:
            WEATHER_MODEL = EfficientNetWeatherClassifier(num_classes=4, pretrained=False)
            WEATHER_MODEL.load_state_dict(torch.load(str(w_ckpt_path), map_location=DEVICE))
            WEATHER_MODEL.eval()
            WEATHER_CAM = WeatherGradCAM(WEATHER_MODEL)
            print(f"[AquaWatch API] Weather EfficientNet-B0 loaded from {w_ckpt_path}")
        except Exception as e:
            print(f"[AquaWatch API] Could not load weather checkpoint: {e}")

# Load RainLSTM model & Scaler
RAIN_MODEL = None
RAIN_SCALER = None
if RAIN_AVAILABLE:
    r_ckpt_path = RESULTS_DIR / "checkpoints" / "rain_lstm.pt"
    r_scaler_path = RESULTS_DIR / "checkpoints" / "rain_scaler.json"
    if r_ckpt_path.exists() and r_scaler_path.exists():
        try:
            with open(r_scaler_path, "r", encoding="utf-8") as f:
                RAIN_SCALER = json.load(f)
            RAIN_MODEL = RainLSTM(num_features=len(RAIN_SCALER["features"]), hidden_size=64, num_layers=2)
            RAIN_MODEL.load_state_dict(torch.load(str(r_ckpt_path), map_location=DEVICE))
            RAIN_MODEL.eval()
            print(f"[AquaWatch API] RainLSTM loaded successfully from {r_ckpt_path}")
        except Exception as e:
            print(f"[AquaWatch API] Could not load RainLSTM: {e}")


def file_to_base64_data_uri(file_path: Path, mime_type: str = "image/png") -> str:
    """Read a local image file and encode as a base64 Data URI."""
    if not file_path.exists():
        return ""
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


@app.get("/")
def root():
    return {
        "project": "AquaWatch",
        "title": "Flood & Waterbody Change Detection from Sentinel Satellite Imagery",
        "status": "Online",
        "docs_url": "/docs"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": (MODEL is not None),
        "results_dir_exists": RESULTS_DIR.exists()
    }


@app.get("/api/regions")
def get_regions():
    """List available demo regions with geographic and temporal metadata."""
    catalog_path = CACHED_DIR / "regions_catalog.json"
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/analyze")
def analyze_change(
    region: str = Query(..., description="Region ID (e.g. chembarambakkam_lake)"),
    before: Optional[str] = Query(None, description="Before date"),
    after: Optional[str] = Query(None, description="After date")
):
    """
    Retrieve precomputed bi-temporal flood change maps, water areas in km²,
    and percent change for the specified demo region.
    """
    reg_dir = CACHED_DIR / region
    meta_path = reg_dir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Region '{region}' not found.")

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Attach direct base64 payloads to ensure instant display with zero network latency
    b64_images = {
        "before_rgb": file_to_base64_data_uri(reg_dir / "before_rgb.png"),
        "after_rgb": file_to_base64_data_uri(reg_dir / "after_rgb.png"),
        "before_mask": file_to_base64_data_uri(reg_dir / "before_mask.png"),
        "after_mask": file_to_base64_data_uri(reg_dir / "after_mask.png"),
        "change_map": file_to_base64_data_uri(reg_dir / "change_map.png")
    }
    data["base64_images"] = b64_images
    return data


@app.get("/api/timeseries")
def get_timeseries(region: str = Query("chembarambakkam_lake")):
    """Get monthly water area monitoring time series with marked flood events."""
    ts_path = RESULTS_DIR / "timeseries.json"
    if not ts_path.exists():
        raise HTTPException(status_code=404, detail="Time-series data not found.")
    with open(ts_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/metrics")
def get_metrics():
    """
    Return real computed test metrics comparing U-Net vs NDWI baselines,
    plus training and validation loss/IoU curves.
    """
    metrics_path = RESULTS_DIR / "metrics.json"
    curves_path = RESULTS_DIR / "curves.json"

    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Run evaluation first.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    curves_data = {}
    if curves_path.exists():
        with open(curves_path, "r", encoding="utf-8") as f:
            curves_data = json.load(f)

    return {
        "metrics": metrics_data,
        "curves": curves_data
    }


@app.get("/api/qualitative")
def get_qualitative():
    """Return qualitative test samples and failure analysis."""
    failure_path = RESULTS_DIR / "failure_analysis.json"
    failures = []
    if failure_path.exists():
        with open(failure_path, "r", encoding="utf-8") as f:
            failures = json.load(f)

    # List qualitative images
    qual_dir = RESULTS_DIR / "qualitative"
    samples = []
    if qual_dir.exists():
        for p in sorted(qual_dir.glob("sample_*.png")):
            samples.append({
                "filename": p.name,
                "url": f"/api/static/qualitative/{p.name}",
                "base64": file_to_base64_data_uri(p)
            })

    return {
        "gallery_url": "/api/static/figures/qualitative_gallery.png",
        "failure_figure_url": "/api/static/figures/failure_analysis.png",
        "samples": samples,
        "failures": failures
    }


@app.post("/api/predict")
async def predict_water_mask(file: UploadFile = File(...)):
    """
    Live prediction endpoint: upload a Sentinel-2 GeoTIFF or standard image
    and receive predicted water segmentation mask, area in km², and water percentage.
    """
    MAX_SIZE = 15 * 1024 * 1024  # 15 MB limit
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 15MB.")

    filename = file.filename.lower()
    try:
        if filename.endswith((".tif", ".tiff")):
            arr = tifffile.imread(io.BytesIO(contents))
            if arr.ndim == 3 and arr.shape[2] == 13:
                arr = np.transpose(arr, (2, 0, 1))
            # If 13 bands: select B2, B3, B4, B8
            if arr.shape[0] >= 8:
                bands_arr = arr[[1, 2, 3, 7], :, :].astype(np.float32) / 10000.0
            elif arr.shape[0] >= 4:
                bands_arr = arr[:4, :, :].astype(np.float32) / 10000.0
            else:
                # Replicate 3 bands to 4
                bands_arr = np.repeat(arr[:3, :, :].astype(np.float32) / 255.0, 1, axis=0)
                bands_arr = np.concatenate([bands_arr, bands_arr[:1]], axis=0)
        else:
            # Standard PNG / JPEG: convert RGB to 4 channels
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            arr = np.array(pil_img).transpose((2, 0, 1)).astype(np.float32) / 255.0
            # 4th channel approximation
            bands_arr = np.concatenate([arr, arr[1:2]], axis=0)

        # Ensure spatial dimensions
        c, h, w = bands_arr.shape
        bands_arr = np.clip(bands_arr, 0.0, 1.0)
        tensor = torch.from_numpy(bands_arr).unsqueeze(0).float()

        # Run inference
        if MODEL is not None:
            with torch.no_grad():
                logits = MODEL(tensor)
                pred_binary = (torch.sigmoid(logits).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            # Fallback NDWI: (Green - NIR) / (Green + NIR)
            green = bands_arr[1]
            nir = bands_arr[3]
            ndwi = (green - nir) / (green + nir + 1e-7)
            pred_binary = (ndwi > 0.0).astype(np.uint8)

        water_pixels = int(np.sum(pred_binary == 1))
        total_pixels = pred_binary.size
        water_pct = round((water_pixels / total_pixels) * 100.0, 2)
        area_km2 = round((water_pixels * 100.0) / 1_000_000.0, 3)

        # Colorize
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        colored[pred_binary == 0] = [45, 55, 72]      # Land
        colored[pred_binary == 1] = [29, 140, 248]    # Water Blue

        pil_out = Image.fromarray(colored)
        buf = io.BytesIO()
        pil_out.save(buf, format="PNG")
        mask_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "status": "success",
            "filename": file.filename,
            "image_dimensions": f"{w}x{h}",
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "water_percentage": water_pct,
            "estimated_water_area_km2": area_km2,
            "mask_base64": mask_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/weather/classify")
async def classify_weather_image(file: UploadFile = File(...)):
    """
    Classify outdoor sky/weather conditions using pretrained EfficientNet-B0.
    Returns:
    - Top predicted weather class and probability
    - Top 3 classes with percentage confidences
    - Grad-CAM heatmap blended over the image (as Base64 Data URI)
    - Original image (as Base64 Data URI)
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB limit
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    if not file.filename.lower().endswith(valid_exts):
        raise HTTPException(status_code=400, detail=f"Unsupported format. Allowed: {valid_exts}")

    if WEATHER_MODEL is None or not WEATHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Weather classification model not currently loaded.")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Preprocess for EfficientNet-B0 (224x224, ImageNet normalization)
        transform = get_weather_transforms("test")
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = WEATHER_MODEL(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Top predictions
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_class = CLASS_NAMES[top_idx].capitalize()
        top_prob = round(float(probs[top_idx]) * 100.0, 2)

        predictions = []
        for idx in sorted_indices[:3]:
            predictions.append({
                "class": CLASS_NAMES[idx].capitalize(),
                "confidence": round(float(probs[idx]) * 100.0, 2)
            })

        # Generate Grad-CAM for top predicted class
        gradcam_b64 = ""
        if WEATHER_CAM is not None:
            # Re-enable gradients for backward pass
            with torch.enable_grad():
                cam_map, _ = WEATHER_CAM.generate_heatmap(input_tensor, target_class=top_idx)
                blended_pil = WEATHER_CAM.overlay_on_image(pil_img, cam_map, alpha=0.45)
                gradcam_b64 = pil_to_base64_data_uri(blended_pil, fmt="JPEG")

        orig_b64 = pil_to_base64_data_uri(pil_img, fmt="JPEG")

        return {
            "status": "success",
            "filename": file.filename,
            "image_size": f"{orig_w}x{orig_h}",
            "top_prediction": top_class,
            "confidence": top_prob,
            "predictions": predictions,
            "gradcam_base64": gradcam_b64,
            "original_base64": orig_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather classification error: {str(e)}")


@app.get("/api/weather/metrics")
def get_weather_metrics():
    """Return real computed test metrics, baseline comparisons, confusion matrix, and failure cases."""
    m_path = RESULTS_DIR / "weather_metrics.json"
    if not m_path.exists():
        raise HTTPException(status_code=404, detail="Weather metrics file not found. Run training first.")
    with open(m_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/weather/samples")
def get_weather_samples():
    """Return 4 preloaded test sample images (Cloudy, Rain, Shine, Sunrise) with precomputed predictions."""
    samples_dir = RESULTS_DIR / "weather_samples"
    samples = []
    classes = ["cloudy", "rain", "shine", "sunrise"]

    for c in classes:
        p = samples_dir / f"{c}_sample.jpg"
        if p.exists():
            samples.append({
                "class": c.capitalize(),
                "filename": p.name,
                "url": f"/api/static/weather_samples/{p.name}",
                "base64": file_to_base64_data_uri(p, mime_type="image/jpeg")
            })
    return samples


# =====================================================================
# SOUTH INDIA LOCATIONS & BASINS CATALOG
# =====================================================================

SOUTH_INDIA_LOCATIONS = [
    # Tamil Nadu
    {
        "id": "chennai",
        "name": "Chennai",
        "state": "Tamil Nadu",
        "lat": 13.0827,
        "lng": 80.2707,
        "basin_name": "Chembarambakkam Lake Reservoir & Adyar Basin",
        "basin_lat": 13.0118,
        "basin_lng": 80.0215,
        "lake_baseline_km2": 26.35,
        "lake_capacity_km2": 46.18
    },
    {
        "id": "coimbatore",
        "name": "Coimbatore",
        "state": "Tamil Nadu",
        "lat": 11.0168,
        "lng": 76.9558,
        "basin_name": "Bhavanisagar Dam & Noyyal River Basin",
        "basin_lat": 11.4700,
        "basin_lng": 77.1300,
        "lake_baseline_km2": 28.5,
        "lake_capacity_km2": 52.0
    },
    {
        "id": "madurai",
        "name": "Madurai",
        "state": "Tamil Nadu",
        "lat": 9.9252,
        "lng": 78.1198,
        "basin_name": "Vaigai Dam Reservoir Catchment",
        "basin_lat": 10.0542,
        "basin_lng": 77.5894,
        "lake_baseline_km2": 18.4,
        "lake_capacity_km2": 36.2
    },
    {
        "id": "tiruchirappalli",
        "name": "Tiruchirappalli",
        "state": "Tamil Nadu",
        "lat": 10.7905,
        "lng": 78.7047,
        "basin_name": "Mukkombu Upper Anaicut & Kaveri Delta",
        "basin_lat": 10.8710,
        "basin_lng": 78.5820,
        "lake_baseline_km2": 24.0,
        "lake_capacity_km2": 48.5
    },

    # Karnataka
    {
        "id": "bengaluru",
        "name": "Bengaluru",
        "state": "Karnataka",
        "lat": 12.9716,
        "lng": 77.5946,
        "basin_name": "Bellandur & Varthur Urban Lake Basins",
        "basin_lat": 12.9370,
        "basin_lng": 77.6710,
        "lake_baseline_km2": 19.8,
        "lake_capacity_km2": 39.4
    },
    {
        "id": "mysuru",
        "name": "Mysuru",
        "state": "Karnataka",
        "lat": 12.2958,
        "lng": 76.6394,
        "basin_name": "Krishna Raja Sagara (KRS) Dam & Cauvery",
        "basin_lat": 12.4278,
        "basin_lng": 76.5728,
        "lake_baseline_km2": 38.6,
        "lake_capacity_km2": 72.0
    },
    {
        "id": "mangaluru",
        "name": "Mangaluru",
        "state": "Karnataka",
        "lat": 12.9141,
        "lng": 74.8560,
        "basin_name": "Netravati & Gurupura Estuary Basin",
        "basin_lat": 12.8420,
        "basin_lng": 74.8430,
        "lake_baseline_km2": 22.4,
        "lake_capacity_km2": 46.8
    },

    # Kerala
    {
        "id": "kochi",
        "name": "Kochi",
        "state": "Kerala",
        "lat": 9.9312,
        "lng": 76.2673,
        "basin_name": "Vembanad Wetland & Periyar River Basin",
        "basin_lat": 9.7500,
        "basin_lng": 76.4000,
        "lake_baseline_km2": 45.2,
        "lake_capacity_km2": 96.0
    },
    {
        "id": "thiruvananthapuram",
        "name": "Thiruvananthapuram",
        "state": "Kerala",
        "lat": 8.5241,
        "lng": 76.9366,
        "basin_name": "Aruvikkara & Peppara Dam Reservoir",
        "basin_lat": 8.5670,
        "basin_lng": 77.0190,
        "lake_baseline_km2": 15.2,
        "lake_capacity_km2": 31.0
    },
    {
        "id": "kozhikode",
        "name": "Kozhikode",
        "state": "Kerala",
        "lat": 11.2588,
        "lng": 75.7804,
        "basin_name": "Kakkayam Dam & Chaliyar River Basin",
        "basin_lat": 11.5540,
        "basin_lng": 75.9220,
        "lake_baseline_km2": 17.6,
        "lake_capacity_km2": 37.2
    },

    # Andhra Pradesh
    {
        "id": "visakhapatnam",
        "name": "Visakhapatnam",
        "state": "Andhra Pradesh",
        "lat": 17.6868,
        "lng": 83.2185,
        "basin_name": "Meghadrigedda & Gambheeram Reservoirs",
        "basin_lat": 17.7560,
        "basin_lng": 83.1890,
        "lake_baseline_km2": 16.8,
        "lake_capacity_km2": 34.0
    },
    {
        "id": "vijayawada",
        "name": "Vijayawada",
        "state": "Andhra Pradesh",
        "lat": 16.5062,
        "lng": 80.6480,
        "basin_name": "Prakasam Barrage & Krishna River Basin",
        "basin_lat": 16.5090,
        "basin_lng": 80.6050,
        "lake_baseline_km2": 32.5,
        "lake_capacity_km2": 66.0
    },
    {
        "id": "tirupati",
        "name": "Tirupati",
        "state": "Andhra Pradesh",
        "lat": 13.6288,
        "lng": 79.4192,
        "basin_name": "Kalyani Dam Reservoir & Swarnamukhi Basin",
        "basin_lat": 13.6550,
        "basin_lng": 79.2880,
        "lake_baseline_km2": 12.8,
        "lake_capacity_km2": 25.5
    },

    # Telangana
    {
        "id": "hyderabad",
        "name": "Hyderabad",
        "state": "Telangana",
        "lat": 17.3850,
        "lng": 78.4867,
        "basin_name": "Hussain Sagar, Osman Sagar & Musi River Basin",
        "basin_lat": 17.4239,
        "basin_lng": 78.4738,
        "lake_baseline_km2": 25.2,
        "lake_capacity_km2": 49.5
    },
    {
        "id": "warangal",
        "name": "Warangal",
        "state": "Telangana",
        "lat": 17.9689,
        "lng": 79.5941,
        "basin_name": "Bhadrakali & Waddepally Reservoir Lakes",
        "basin_lat": 17.9890,
        "basin_lng": 79.5780,
        "lake_baseline_km2": 14.1,
        "lake_capacity_km2": 27.6
    },

    # Puducherry (UT)
    {
        "id": "puducherry",
        "name": "Puducherry",
        "state": "Puducherry (UT)",
        "lat": 11.9416,
        "lng": 79.8083,
        "basin_name": "Ousteri (Ussudu) Lake & Sankaraparani Basin",
        "basin_lat": 11.9540,
        "basin_lng": 79.7420,
        "lake_baseline_km2": 11.5,
        "lake_capacity_km2": 23.2
    }
]


@app.get("/api/live/locations")
def get_live_locations():
    """Return catalog of monitored South Indian cities and river basins across all southern states."""
    return SOUTH_INDIA_LOCATIONS


# =====================================================================
# PHASE 10: RAIN PREDICTION ENDPOINTS
# =====================================================================

@app.get("/api/predict/rain")
def predict_rain_tomorrow(
    location_id: Optional[str] = Query("chennai", description="South Indian location ID"),
    lat: Optional[float] = Query(None, description="Custom latitude"),
    lng: Optional[float] = Query(None, description="Custom longitude")
):
    """
    Predict 24-hour next-day rain probability for any South Indian state / city
    using RainLSTM over the past 10-day meteorological sequence from Open-Meteo.
    Includes the official 7-day external forecast from Open-Meteo.
    """
    # Resolve target location
    selected_loc = next((l for l in SOUTH_INDIA_LOCATIONS if l["id"] == location_id), SOUTH_INDIA_LOCATIONS[0])
    target_lat = lat if lat is not None else selected_loc["lat"]
    target_lng = lng if lng is not None else selected_loc["lng"]
    loc_display_name = f"{selected_loc['name']}, {selected_loc['state']}"

    is_live = False
    seq_data = None
    recent_dates = []
    external_forecast = []

    # 1. Fetch live past 14 days and 7-day forecast from Open-Meteo for target coordinates
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={target_lat}&longitude={target_lng}&"
            f"daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max,cloud_cover_mean,"
            f"precipitation_probability_max&past_days=14&forecast_days=7&timezone=Asia%2FKolkata"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AquaWatch/1.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            payload = json.loads(response.read().decode())
            daily = payload.get("daily", {})

        feature_keys = RAIN_SCALER["features"] if RAIN_SCALER else [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "relative_humidity_2m_mean", "surface_pressure_mean", "wind_speed_10m_max", "cloud_cover_mean"
        ]
        
        total_days = len(daily.get("time", []))
        past_end_idx = max(10, total_days - 7)
        past_start_idx = past_end_idx - 10

        seq_matrix = []
        for i in range(past_start_idx, past_end_idx):
            row = [float(daily[k][i]) if daily[k][i] is not None else 0.0 for k in feature_keys]
            seq_matrix.append(row)
            recent_dates.append(daily["time"][i])

        if len(seq_matrix) == 10:
            seq_data = np.array(seq_matrix, dtype=np.float32)
            is_live = True

        for i in range(past_end_idx, total_days):
            external_forecast.append({
                "date": daily["time"][i],
                "precipitation_sum_mm": daily.get("precipitation_sum", [])[i],
                "precipitation_probability_pct": daily.get("precipitation_probability_max", [])[i],
                "temp_max_c": daily.get("temperature_2m_max", [])[i],
                "temp_min_c": daily.get("temperature_2m_min", [])[i]
            })

    except Exception as e:
        print(f"[AquaWatch API] Open-Meteo live query notice ({loc_display_name}): {e}")

    # Fallback to local 10-year dataset if network error or incomplete
    if seq_data is None:
        csv_path = PROJECT_DIR / "data" / "chennai_weather_10yr.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            last_10 = df.tail(10)
            seq_data = last_10[RAIN_SCALER["features"]].values.astype(np.float32)
            recent_dates = last_10["time"].tolist()
            is_live = False

    # Inference using RainLSTM
    prob = 0.5
    if RAIN_MODEL is not None and RAIN_SCALER is not None and seq_data is not None:
        mean = np.array(RAIN_SCALER["mean"], dtype=np.float32)
        scale = np.array(RAIN_SCALER["scale"], dtype=np.float32)
        scaled_seq = (seq_data - mean) / scale
        tensor_in = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = RAIN_MODEL(tensor_in)
            prob = float(torch.sigmoid(logit).item())

    prob_pct = round(prob * 100.0, 1)
    rain_predicted = (prob >= 0.50)

    recent_history = []
    if seq_data is not None:
        for idx, d in enumerate(recent_dates):
            recent_history.append({
                "date": d,
                "temp_max": round(float(seq_data[idx][0]), 1),
                "temp_min": round(float(seq_data[idx][1]), 1),
                "rain_mm": round(float(seq_data[idx][2]), 1),
                "humidity": round(float(seq_data[idx][3]), 1),
                "pressure": round(float(seq_data[idx][4]), 1),
                "wind_kmh": round(float(seq_data[idx][5]), 1),
                "cloud_pct": round(float(seq_data[idx][6]), 1)
            })

    return {
        "status": "success",
        "location": loc_display_name,
        "location_id": selected_loc["id"],
        "coordinates": {"lat": target_lat, "lng": target_lng},
        "basin_name": selected_loc["basin_name"],
        "prediction_horizon": "Next 24 Hours (Tomorrow)",
        "rain_predicted": rain_predicted,
        "rain_probability_pct": prob_pct,
        "confidence_level": "High" if abs(prob_pct - 50.0) > 25 else "Moderate",
        "verdict": "Rain Expected (>= 1.0mm)" if rain_predicted else "Rain Unlikely (< 1.0mm)",
        "model_architecture": "AquaWatch RainLSTM (PyTorch 2-Layer LSTM with Dropout)",
        "data_source": "Live Open-Meteo Weather API" if is_live else "Historical Meteorological Archive (Offline Mode)",
        "is_live_data": is_live,
        "lookback_window_days": 10,
        "recent_history": recent_history,
        "external_forecast_7d": external_forecast,
        "external_forecast_source": "Open-Meteo Global Forecasting Engine (Labeled External Forecast)"
    }


@app.get("/api/predict/rain/metrics")
def get_rain_metrics():
    """Return real computed test metrics across all 4 models: Persistence, Logistic Regression, GBDT, and RainLSTM."""
    m_path = RESULTS_DIR / "rain_metrics.json"
    if not m_path.exists():
        raise HTTPException(status_code=404, detail="Rain metrics file not found. Run training script first.")
    with open(m_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================================
# PHASE 11: LIVE MONITORING & FLOOD RISK ENGINE
# =====================================================================

@app.get("/api/live/monitor")
def get_live_monitor_data(
    location_id: Optional[str] = Query("chennai", description="South Indian location ID"),
    lat: Optional[float] = Query(None, description="Custom latitude"),
    lng: Optional[float] = Query(None, description="Custom longitude")
):
    """
    Live monitoring hub supporting all South Indian states (Tamil Nadu, Karnataka, Kerala, Andhra Pradesh, Telangana, Puducherry):
    - Real-time Open-Meteo conditions for the selected location
    - NASA GIBS (MODIS/VIIRS) daily satellite imagery tile endpoints
    - RainViewer live weather radar tile endpoints
    - Transparent rule-based Flood Risk Indicator score combining 3-day rain forecast & local basin retention
    """
    # 1. Resolve selected location
    selected_loc = next((l for l in SOUTH_INDIA_LOCATIONS if l["id"] == location_id), SOUTH_INDIA_LOCATIONS[0])
    target_lat = lat if lat is not None else selected_loc["lat"]
    target_lng = lng if lng is not None else selected_loc["lng"]

    # 2. Fetch current weather and 3-day rain forecast from Open-Meteo
    curr_weather = {
        "temperature_2m": 28.5,
        "relative_humidity_2m": 72.0,
        "precipitation": 0.0,
        "surface_pressure": 1009.5,
        "wind_speed_10m": 12.0,
        "cloud_cover": 55.0,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    }
    forecast_3d_rain_mm = 0.0
    rain_fallback = False

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={target_lat}&longitude={target_lng}&"
            f"current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,cloud_cover&"
            f"daily=precipitation_sum&timezone=Asia%2FKolkata&forecast_days=3"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AquaWatch/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if "current" in data:
                curr_weather = data["current"]
            if "daily" in data and "precipitation_sum" in data["daily"]:
                precip_list = [p for p in data["daily"]["precipitation_sum"] if p is not None]
                forecast_3d_rain_mm = round(float(sum(precip_list)), 1)
    except Exception as e:
        print(f"[AquaWatch API] Open-Meteo live monitor notice ({selected_loc['name']}): {e}")
        rain_fallback = True
        forecast_3d_rain_mm = 3.5

    # 3. Reservoir Basin retention baseline
    lake_baseline = selected_loc.get("lake_baseline_km2", 24.0)
    lake_capacity = selected_loc.get("lake_capacity_km2", 48.0)
    lake_current_km2 = lake_baseline

    # For Chennai, use real Sentinel-2 calculated lake area from timeseries.json
    if selected_loc["id"] == "chennai":
        ts_path = RESULTS_DIR / "timeseries.json"
        if ts_path.exists():
            try:
                with open(ts_path, "r", encoding="utf-8") as f:
                    ts_data = json.load(f)
                    if "data" in ts_data and len(ts_data["data"]) > 0:
                        lake_current_km2 = float(ts_data["data"][-1].get("water_area_km2", 26.35))
            except Exception:
                pass
    else:
        # Dynamic retention estimate based on recent rainfall accumulation
        rain_adjustment = min(lake_capacity - lake_baseline, (forecast_3d_rain_mm / 100.0) * (lake_capacity - lake_baseline) * 0.4)
        lake_current_km2 = round(lake_baseline + rain_adjustment, 2)

    # 4. Transparent Rule-Based Flood Risk Computation
    # Heavy rainfall warning threshold is 100mm in 72 hours
    rain_score = min(100.0, (forecast_3d_rain_mm / 100.0) * 100.0)
    # Lake fullness subscore
    lake_score = min(100.0, max(0.0, ((lake_current_km2 - (lake_baseline * 0.85)) / (lake_capacity - (lake_baseline * 0.85))) * 100.0))
    flood_risk_score = round(0.55 * rain_score + 0.45 * lake_score, 1)

    if flood_risk_score < 25.0:
        risk_level = "LOW"
        risk_color = "#10b981"  # Emerald
        advisory = f"Normal reservoir retention across {selected_loc['basin_name']}. Routine monitoring active."
    elif flood_risk_score < 50.0:
        risk_level = "MODERATE"
        risk_color = "#f59e0b"  # Amber
        advisory = f"Elevated catchment runoff expected into {selected_loc['basin_name']}. Monitor weir discharges."
    elif flood_risk_score < 75.0:
        risk_level = "HIGH"
        risk_color = "#ea580c"  # Orange
        advisory = f"Precautionary sluice gate discharge alert for downstream floodplains of {selected_loc['basin_name']}."
    else:
        risk_level = "CRITICAL"
        risk_color = "#ef4444"  # Red
        advisory = f"Severe inundation hazard across {selected_loc['basin_name']}. Emergency protocol advised."

    # 5. NASA GIBS Tiles
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    nasa_tiles = {
        "satellite_date": yesterday,
        "label": "Latest Available Daily Pass (Not Real-Time)",
        "modis_truecolor_template": (
            f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"MODIS_Terra_CorrectedReflectance_TrueColor/default/{yesterday}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
        ),
        "viirs_truecolor_template": (
            f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{yesterday}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
        ),
        "modis_clouds_template": (
            f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"MODIS_Terra_Cloud_Optical_Thickness/default/{yesterday}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png"
        ),
        "attribution": "NASA Global Imagery Browse Services (GIBS) / EOSDIS"
    }

    # 6. RainViewer Live Radar Tiles
    radar_info = {
        "status": "active",
        "latest_timestamp": None,
        "radar_tile_template": None,
        "source": "RainViewer Open Weather Radar API"
    }
    try:
        req_rv = urllib.request.Request("https://api.rainviewer.com/public/weather-maps.json", headers={"User-Agent": "AquaWatch/1.0"})
        with urllib.request.urlopen(req_rv, timeout=4) as rv_resp:
            rv_data = json.loads(rv_resp.read().decode())
            host = rv_data.get("host", "https://tilecache.rainviewer.com")
            past_radar = rv_data.get("radar", {}).get("past", [])
            if past_radar:
                latest_radar = past_radar[-1]
                path = latest_radar.get("path")
                radar_info["latest_timestamp"] = latest_radar.get("time")
                radar_info["radar_tile_template"] = f"{host}{path}/256/{{z}}/{{x}}/{{y}}/2/1_1.png"
    except Exception as e:
        print(f"[AquaWatch API] RainViewer notice: {e}")
        radar_info["status"] = "offline_fallback"

    return {
        "status": "success",
        "selected_location": selected_loc,
        "center_coordinates": {"lat": target_lat, "lng": target_lng},
        "basin_coordinates": {"lat": selected_loc["basin_lat"], "lng": selected_loc["basin_lng"]},
        "current_weather": curr_weather,
        "flood_risk_engine": {
            "score": flood_risk_score,
            "level": risk_level,
            "color": risk_color,
            "advisory": advisory,
            "formula": "Score = 0.55 * (3d_Rain_mm / 100mm * 100) + 0.45 * ((Lake_Area - Baseline) / Delta * 100)",
            "parameters": {
                "forecast_3d_rain_mm": forecast_3d_rain_mm,
                "current_lake_area_km2": lake_current_km2,
                "lake_normal_baseline_km2": lake_baseline,
                "lake_flood_capacity_km2": lake_capacity,
                "rain_subscore": round(rain_score, 1),
                "lake_fullness_subscore": round(lake_score, 1)
            },
            "disclaimer": "⚠️ Indicative research score based on meteorological forecasts and satellite lake retention. Not an official disaster warning."
        },
        "nasa_gibs": nasa_tiles,
        "rainviewer_radar": radar_info,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
