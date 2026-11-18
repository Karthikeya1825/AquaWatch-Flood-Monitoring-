# AquaWatch 🌊 — Satellite Flood & Weather Intelligence

> An interactive web dashboard that detects flood water from Sentinel-2 satellite imagery, predicts 24-hour rainfall, and classifies outdoor sky conditions with visual heatmaps.

---

## 🔗 Quick Links (How to Open the Project)

* **Local Web Dashboard:** [http://127.0.0.1:5173/](http://127.0.0.1:5173/)  
* **Backend API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
* **GitHub Repository:** [https://github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-](https://github.com/Karthikeya1825/AquaWatch-Flood-Monitoring-)

---

## ⚡ 1-Click Launch (No Terminal Needed)

If you have cloned this repository on your Windows computer:
1. Double-click **`START_AQUAWATCH.bat`**  
   *(It starts the backend, starts the frontend, and automatically opens the website in your browser).*
2. **Want to view it on your phone or share it with a friend/examiner?**  
   Double-click **`SHARE_ONLINE_PUBLIC_URL.bat`** to get an instant, secure public link you can open on any device anywhere.
3. To stop the servers when you're done, double-click **`STOP_AQUAWATCH.bat`**.

---

## 📌 What is AquaWatch?

When severe weather or cyclones hit (like Cyclone Michaung in December 2023), disaster response teams and city officials need to know:
1. *Where exactly is the water rising right now, and how much land is flooded?*
2. *Will it rain heavily again in the next 24 hours?*
3. *What do ground sky and cloud conditions look like in real time?*

AquaWatch brings all three answers together into a single web dashboard by combining **three deep learning models** with **live satellite and radar feeds**.

---

## 🧠 The 3 Deep Learning Models Inside

### 1. Flood Detection from Space (U-Net with ResNet34)
* **What it does:** Looks at 10-meter resolution multispectral satellite photos (Sentinel-2) across 4 bands (Blue, Green, Red, Near-Infrared) and creates a pixel-by-pixel map showing exactly where water is.
* **Why not simple color thresholding?** Traditional formulas like NDWI get tricked by building shadows, muddy/sedimented rivers, and cloud fringes. Our U-Net learns spatial shape and context.
* **Real Test Result:** **83.2% Water IoU** and **90.8% Dice Score** on the official Sen1Floods11 test split (+28.9% higher than classical NDWI).
* **Case Study:** Measured the expansion of Chembarambakkam Lake during Cyclone Michaung (swelling from a baseline of **26.35 km²** to a flood peak of **46.18 km²**, a **+21.2 km² / +75.3% surge**).

### 2. Weather Detective: Sky & Cloud Classifier (EfficientNet-B0 + Grad-CAM)
* **What it does:** Upload any outdoor sky photo (or take one from your phone camera). It identifies whether the weather is **Cloudy**, **Rainy**, **Sunny/Shine**, or **Sunrise/Sunset**.
* **Real Test Result:** **96.5% Test Accuracy** across 169 unseen test photos, with **100% recall on rain** (it never missed an active rain scene).
* **Grad-CAM Heatmaps:** You can flip the "Show Grad-CAM" switch on the website to see a red/yellow visual heatmap showing the exact cloud shapes that made the AI make its prediction.

### 3. 24-Hour Rain Predictor (RainLSTM Sequence Model)
* **What it does:** Takes the past 10 days of weather (temperatures, humidity, air pressure, wind speed, cloud cover, and rain) and predicts the probability that it will rain tomorrow.
* **Trained on:** 10 full years of daily weather records for Chennai (2014–2024: 3,653 days).
* **Real Test Result:** Evaluated on the 2023–2024 test year using strict chronological splitting (no data leakage). Achieved **0.816 ROC-AUC** and **73.0% Rain Recall**, learning multi-day moisture build-up preceding cyclonic storms.

---

## 🛰️ Live Map Features

In the **Live Basin Monitor** section:
* **Interactive Leaflet Map:** Panning and zooming across South India.
* **Location Switcher for South Indian States:** Check live weather, flood risk, and rain predictions across 16 major cities and river basins in **Tamil Nadu**, **Karnataka**, **Kerala**, **Andhra Pradesh**, **Telangana**, and **Puducherry** (Chennai, Bengaluru, Kochi, Hyderabad, Vijayawada, Coimbatore, Mysuru, Visakhapatnam, etc.).
* **NASA Daily Satellite Passes:** Toggle latest MODIS Terra and VIIRS SNPP true-color optical satellite imagery.
* **RainViewer Live Radar:** Real-time Doppler weather radar showing moving rainstorms and intensity over the Bay of Bengal and mainland.
* **Transparent Flood Risk Score:** Combines 3-day rainfall forecast with local reservoir capacity to show an indicative risk level (Low, Moderate, High, Critical) with the exact formula displayed.

---

## 💻 Tech Stack

* **Machine Learning & Deep Learning:** PyTorch, Torchvision, Segmentation Models PyTorch (SMP), Scikit-Learn, Albumentations
* **Backend API:** FastAPI, Uvicorn, NumPy, Pillow, Rasterio
* **Frontend Dashboard:** React 19, Vite, Tailwind CSS, Leaflet, Framer Motion, Recharts, Lucide Icons
* **Data Sources:** ESA Copernicus Sentinel-2 L2A, Sen1Floods11, Open-Meteo Historical & Forecast APIs, NASA GIBS WMTS, RainViewer Doppler Radar

---

## 🛠️ How to Run from Source Code (Terminal)

If you prefer running commands manually:

### Step 1: Install Python Dependencies
```bash
pip install -r aquawatch/requirements.txt
```

### Step 2: Start the Backend (Port 8000)
```bash
py -m uvicorn main:app --app-dir aquawatch/backend --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Start the Frontend (Port 5173)
```bash
cd aquawatch/frontend
npm install
npm run dev
```
Open **[http://127.0.0.1:5173/](http://127.0.0.1:5173/)** in your browser.

---

## 🎓 Academic Defense & Viva Resources

For college presentation and viva voce exams, complete reference guides are included in the repo:
* **[VIVA_PREPARATION.md](aquawatch/VIVA_PREPARATION.md):** 20 detailed questions and answers covering CNNs, transfer learning, Grad-CAM, time-based splits, why GBDT vs LSTM perform as they do, and satellite revisit constraints.
* **[PRESENTATION_GUIDE.md](aquawatch/PRESENTATION_GUIDE.md):** A 7-minute slide-by-slide presentation outline with timing cues.

---

## 📄 License
Academic and research use under the MIT License.
Contains modified Copernicus Sentinel data [2015–2024] and Open-Meteo weather data.
