# AquaWatch: 7-Minute College Major Project Presentation Script

Use this structured script to present AquaWatch smoothly during your project evaluation, seminar, or viva voce examination.

---

### Minute 0:00 – 1:00: Title & Multi-Model Architecture
- **Opening:** *"Good morning respected evaluators and faculty. Today I am presenting **AquaWatch: Multi-Model Deep Learning for Flood, Weather, and Rain Intelligence**."*
- **Problem Statement:** Floods are the most destructive natural disasters worldwide. Traditional disaster management relies on delayed post-event damage assessments or simplistic thresholding indices that trigger false alarms in cloud shadows and turbid rivers.
- **AquaWatch Solution:** We have developed an end-to-end multi-model environmental intelligence platform uniting:
  1. **Satellite Inundation Segmentation (U-Net ResNet34)** from 10-meter Sentinel-2 multispectral imagery.
  2. **Mobile Sky Weather Classification (EfficientNet-B0 + Grad-CAM)** with convolutional saliency heatmaps.
  3. **24-Hour Rain Prediction (RainLSTM & GBDT)** across 10 years of meteorological records.
  4. **Live Basin Monitoring** streaming NASA GIBS daily passes, RainViewer Doppler radar, and a transparent flood retention risk engine.

---

### Minute 1:00 – 2:00: Model 1 — Satellite Flood U-Net
- **Dataset:** Official hand-labeled benchmark of **Sen1Floods11** (446 chips of $512 \times 512$ across 11 global flood events, CVPRW 2020).
- **Multispectral Adaptation:** Standard ImageNet models accept 3 RGB channels. We adapted the first convolutional layer to accept **4 bands (Blue, Green, Red, and Near-Infrared B8)**, initializing the NIR channel with the mean of RGB weights to maintain transfer learning advantages.
- **Handling Cloud Occlusions:** Official ground truth contains `-1` for clouds and invalid pixels. Our custom `MaskedBCEDiceLoss` dynamically masks out these pixels, preventing occlusions from penalizing gradient updates.
- **Test Set Results (>20.5M Pixels):**
  - **Classical NDWI Baseline:** 54.28% IoU (Otsu fails because bimodal assumptions collapse on minority water scenes).
  - **AquaWatch U-Net:** **83.21% Water IoU (+28.93% absolute gain), 90.83% Dice, and 97.82% overall accuracy**.

---

### Minute 2:00 – 3:00: Model 2 — Weather Detective (EfficientNet-B0 + Grad-CAM)
- **Motivation:** Ground-level sky imagery provides real-time atmospheric validation before satellite data can be downlinked.
- **Dataset:** Kaggle Multi-class Weather Dataset (1,125 images: Cloudy, Rain, Shine, Sunrise) partitioned with a fixed-seed 70/15/15 stratified split.
- **Architecture:** Pretrained **EfficientNet-B0** fine-tuned with AdamW, Cosine Annealing, and Label Smoothing.
- **Results:**
  - **96.45% Test Accuracy** and **96.45% Macro F1** across 169 unseen test images.
  - Beat Simple CNN from scratch (85.21%) and Logistic Regression (82.84%).
  - **100% Rain Recall & F1:** The model never missed an active rainfall event.
- **Grad-CAM Explainability:** We integrated Gradient-weighted Class Activation Mapping directly into the web UI. Operators can toggle a heatmap overlay to visually verify that predictions are driven by dark cloud formations rather than background artifacts.

---

### Minute 3:00 – 4:00: Model 3 — 24-Hour Rain Prediction (Time-Series DL)
- **Dataset:** 10-year daily weather record (2014–2024: 3,653 days) for Chennai from Open-Meteo Historical Archive across 7 meteorological variables.
- **Sequence Modeling:** Trained a **2-layer PyTorch RainLSTM** with 10-day sliding lookback windows and compared it against Persistence, Logistic Regression, and Gradient-Boosted Trees (GBDT).
- **Strict Chronological Split:** Train (2014–2020), Validation (2021–2022), Test (2023–2024). Zero random shuffling to prevent temporal data leakage.
- **Scientific Honesty Benchmark:**
  - GBDT achieved top ROC-AUC (**0.828**) because decision trees cleanly isolate sharp humidity and pressure cutoffs.
  - RainLSTM achieved top Rain Recall (**73.03%**) and **0.816 ROC-AUC**, successfully learning multi-day antecedent moisture accumulation leading up to Cyclone Michaung.

---

### Minute 4:00 – 5:00: Live Basin Monitor & Flood Risk Engine
- **Live Geospatial Telemetry:**
  - **NASA GIBS WMTS:** Overlays daily true-color satellite passes from MODIS Terra and VIIRS SNPP, plus Cloud Optical Thickness (clearly labeled *"latest available pass, not real-time"*).
  - **RainViewer Live Radar:** Streams real-time Doppler radar reflectivity tiles updating every 10 minutes.
  - **Open-Meteo Current Telemetry:** Ground temperature, humidity, wind, and pressure in Chennai.
- **Transparent Rule-Based Flood Risk Formula:**
  $$\text{Score} = 0.55 \times \left(\frac{R_{3\text{d}}}{100\text{ mm}} \times 100\right) + 0.45 \times \left(\frac{A_{\text{lake}} - 23.0\text{ km}^2}{23.2\text{ km}^2} \times 100\right)$$
  Combines 3-day rainfall forecast with latest Sentinel-2 Chembarambakkam Lake water retention to compute Low, Moderate, High, or Critical risk scores with full mathematical transparency.

---

### Minute 5:00 – 6:00: Case Study & Interactive Web Demo
- **Demo Region:** Chembarambakkam Lake, Chennai (origin of the Adyar River).
- **Cyclone Michaung Analysis:**
  - Pre-flood baseline (Oct 07, 2023): **26.35 km²**.
  - Post-cyclone peak (Jan 15, 2024): **46.18 km²**.
  - Measured inundation surge: **+21.24 km² (+75.3%)**, swelling far beyond the 24.5 km² full reservoir capacity.
- **Live Web Features:**
  - Split slider comparing satellite RGB vs AI inundation change map (Orange = New Flooding, Blue = Permanent Water, Purple = Receded).
  - 12-month cloud-filtered 2023 time series backed by real Sentinel-2 Level-2A data.
  - Instant 1-click Weather Detective samples with Grad-CAM heatmap toggles.

---

### Minute 6:00 – 7:00: Limitations, Future Work & Conclusion
- **Scientific Honesty & Failure Modes:**
  - Optical satellite revisit time is 5 days, and thick cloud covers can obscure visual sensors during active storms.
  - Thin cirrus cloud margins and urban building shadows can occasionally cause edge noise in visible bands.
- **Future Scope:**
  - Multimodal fusion with **Sentinel-1 SAR Radar (C-Band VV/VH)**, which penetrates through monsoonal rain and clouds for 24/7 all-weather flood mapping.
- **Closing:** *"AquaWatch demonstrates that combining multispectral computer vision, explainable sky classification, and time-series deep learning creates a robust, transparent environmental monitoring system. Thank you, and I look forward to your questions."*
