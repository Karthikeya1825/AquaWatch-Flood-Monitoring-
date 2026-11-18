import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Satellite,
  Droplets,
  Layers,
  Activity,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
  Info,
  Calendar,
  MapPin,
  CheckCircle2,
  ChevronDown,
  Upload,
  BarChart3,
  Sliders,
  ShieldAlert,
  Compass,
  Database
} from 'lucide-react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  LineChart,
  Line,
  Legend
} from 'recharts'
import ImageSlider from './components/ImageSlider'
import LiveMonitor from './components/LiveMonitor'
import WeatherDetective from './components/WeatherDetective'
import ResultsMultiModel from './components/ResultsMultiModel'

const API_BASE = "http://127.0.0.1:8000"

export default function App() {
  const [regions, setRegions] = useState([])
  const [selectedRegionId, setSelectedRegionId] = useState("chembarambakkam_lake")
  const [regionData, setRegionData] = useState(null)
  const [activeViewMode, setActiveViewMode] = useState("satellite") // 'satellite' | 'mask' | 'change'
  const [timeseriesData, setTimeseriesData] = useState(null)
  const [metricsData, setMetricsData] = useState(null)
  const [qualitativeData, setQualitativeData] = useState(null)
  const [backendOnline, setBackendOnline] = useState(false)
  const [learnMoreOpen, setLearnMoreOpen] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const [activeSection, setActiveSection] = useState("hero")

  // Scroll-spy active state handler
  useEffect(() => {
    const sectionIds = [
      "hero",
      "problem",
      "how-it-works",
      "live-monitor",
      "weather-detective",
      "flood-analysis",
      "timeseries",
      "results",
      "about"
    ]
    const handleScroll = () => {
      const scrollPos = window.scrollY + 140
      for (let i = sectionIds.length - 1; i >= 0; i--) {
        const el = document.getElementById(sectionIds[i])
        if (el && el.offsetTop <= scrollPos) {
          setActiveSection(sectionIds[i])
          break
        }
      }
    }
    window.addEventListener("scroll", handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Fetch initial API data
  useEffect(() => {
    async function loadData() {
      try {
        const healthRes = await fetch(`${API_BASE}/api/health`)
        if (healthRes.ok) setBackendOnline(true)

        const [regRes, tsRes, metRes, qualRes] = await Promise.all([
          fetch(`${API_BASE}/api/regions`),
          fetch(`${API_BASE}/api/timeseries`),
          fetch(`${API_BASE}/api/metrics`),
          fetch(`${API_BASE}/api/qualitative`)
        ])

        if (regRes.ok) {
          const regJson = await regRes.json()
          setRegions(regJson)
          if (regJson.length > 0) setSelectedRegionId(regJson[0].id)
        }
        if (tsRes.ok) setTimeseriesData(await tsRes.json())
        if (metRes.ok) setMetricsData(await metRes.json())
        if (qualRes.ok) setQualitativeData(await qualRes.json())
      } catch (err) {
        console.warn("Backend not yet connected; using fallback data.", err)
      }
    }
    loadData()
  }, [])

  // Fetch analyzed region details when region changes
  useEffect(() => {
    async function fetchRegion() {
      if (!selectedRegionId) return
      try {
        const res = await fetch(`${API_BASE}/api/analyze?region=${selectedRegionId}`)
        if (res.ok) {
          const data = await res.json()
          setRegionData(data)
        }
      } catch (err) {
        console.error("Failed to load region data:", err)
      }
    }
    fetchRegion()
  }, [selectedRegionId])

  // File upload handler for live inference
  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setUploadStatus("Processing satellite chip...")
    setUploadResult(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData
      })
      if (!res.ok) throw new Error("Inference failed")
      const result = await res.json()
      setUploadResult(result)
      setUploadStatus("Prediction completed!")
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`)
    }
  }

  // Selected image pair based on active view mode
  const currentImages = () => {
    if (!regionData) return { before: "", after: "" }
    const imgs = regionData.base64_images || {}
    if (activeViewMode === "satellite") {
      return {
        before: imgs.before_rgb || `${API_BASE}${regionData.images.before_rgb}`,
        after: imgs.after_rgb || `${API_BASE}${regionData.images.after_rgb}`
      }
    } else if (activeViewMode === "mask") {
      return {
        before: imgs.before_mask || `${API_BASE}${regionData.images.before_mask}`,
        after: imgs.after_mask || `${API_BASE}${regionData.images.after_mask}`
      }
    } else {
      // Change map comparison against baseline
      return {
        before: imgs.before_rgb || `${API_BASE}${regionData.images.before_rgb}`,
        after: imgs.change_map || `${API_BASE}${regionData.images.change_map}`
      }
    }
  }

  const images = currentImages()

  // Real or fallback headline stats
  const headline = metricsData?.metrics?.headline_stats || {
    unet_iou: 0.8315,
    improvement_pct: 9.73,
    regions_analyzed: 11
  }

  return (
    <div className="min-h-screen bg-[#F6FAFD] text-[#0F2438] selection:bg-[#1D8CF8] selection:text-white">
      {/* 0. STICKY NAVBAR */}
      <header className="sticky top-0 z-50 bg-[#0B1F33]/90 backdrop-blur-md border-b border-slate-800 text-white transition-all">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#1D8CF8] to-[#19C3B1] flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Droplets className="w-6 h-6 text-white" />
            </div>
            <div>
              <span className="font-extrabold text-xl tracking-tight bg-gradient-to-r from-white via-slate-100 to-[#19C3B1] bg-clip-text text-transparent">
                AquaWatch
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs text-slate-400 font-medium border-l border-slate-700 pl-2">
                Deep Learning Satellite Earth Observation
              </span>
            </div>
          </div>

          <nav className="hidden md:flex items-center space-x-5 text-sm font-medium text-slate-300">
            <a
              href="#problem"
              className={`transition-colors py-1 ${
                activeSection === "problem"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              Problem
            </a>
            <a
              href="#how-it-works"
              className={`transition-colors py-1 ${
                activeSection === "how-it-works"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              How It Works
            </a>
            <a
              href="#live-monitor"
              className={`transition-colors py-1 ${
                activeSection === "live-monitor"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              Live Monitor
            </a>
            <a
              href="#weather-detective"
              className={`transition-colors py-1 ${
                activeSection === "weather-detective"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              Weather Detective
            </a>
            <a
              href="#flood-analysis"
              className={`transition-colors py-1 ${
                activeSection === "flood-analysis"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              Flood Analysis
            </a>
            <a
              href="#timeseries"
              className={`transition-colors py-1 ${
                activeSection === "timeseries"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              Time Series
            </a>
            <a
              href="#results"
              className={`transition-colors py-1 ${
                activeSection === "results"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              Results
            </a>
            <a
              href="#about"
              className={`transition-colors py-1 ${
                activeSection === "about"
                  ? "text-[#19C3B1] font-bold border-b-2 border-[#19C3B1]"
                  : "text-slate-300 hover:text-[#19C3B1]"
              }`}
            >
              About
            </a>
          </nav>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs px-2.5 py-1 rounded-full bg-slate-800/80 border border-slate-700">
              <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span className="text-slate-300 font-mono text-[11px]">
                {backendOnline ? 'FastAPI Online' : 'Cached Mode'}
              </span>
            </div>
            <a
              href="#demo"
              className="bg-[#1D8CF8] hover:bg-blue-500 text-white px-4 py-2 rounded-xl text-sm font-semibold shadow-md shadow-blue-500/20 transition-all hover:scale-105 active:scale-95"
            >
              Try Demo
            </a>
          </div>
        </div>
      </header>

      {/* 1. HERO SECTION */}
      <section id="hero" className="relative bg-[#0B1F33] text-white pt-16 pb-28 overflow-hidden scroll-mt-24">
        {/* Subtle background glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] bg-[#1D8CF8]/15 rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center space-x-2 bg-slate-800/80 border border-slate-700/80 rounded-full px-4 py-1.5 text-xs text-[#19C3B1] font-semibold mb-6 shadow-lg"
          >
            <Satellite className="w-4 h-4 text-[#19C3B1] animate-bounce" />
            <span>Multi-Model AI: Satellite Inundation + Weather Vision + Rain Prediction</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-tight"
          >
            Flood, Weather and Rain Intelligence from Satellite and Sky Imagery
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-6 text-lg sm:text-xl text-slate-300 max-w-3xl mx-auto font-normal leading-relaxed"
          >
            An integrated deep learning system combining Sentinel-2 multispectral flood segmentation,
            mobile sky classification with Grad-CAM explainability, and 24-hour meteorological rain sequence forecasting.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-8 flex flex-wrap items-center justify-center gap-4"
          >
            <a
              href="#live-monitor"
              className="bg-[#1D8CF8] hover:bg-blue-500 text-white font-semibold px-6 py-3.5 rounded-xl shadow-lg shadow-blue-500/25 transition-all hover:scale-105 active:scale-95 flex items-center space-x-2"
            >
              <span>Explore Live Basin Monitor</span>
              <ArrowRight className="w-4 h-4" />
            </a>
            <a
              href="#weather-detective"
              className="bg-slate-800 hover:bg-slate-700/80 text-slate-200 border border-slate-700 font-semibold px-6 py-3.5 rounded-xl transition-all hover:scale-105 active:scale-95"
            >
              Weather Detective (Grad-CAM)
            </a>
            <a
              href="#flood-analysis"
              className="bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold px-6 py-3.5 rounded-xl transition-all hover:scale-105 active:scale-95"
            >
              Satellite Flood Analysis
            </a>
          </motion.div>

          {/* 3 Real Headline Statistics from the 3 Deep Learning Systems */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl mx-auto"
          >
            <div className="bg-slate-800/60 backdrop-blur-md border border-slate-700/70 p-6 rounded-2xl shadow-xl text-center">
              <span className="block text-4xl lg:text-5xl font-extrabold text-[#1D8CF8]">
                83.2%
              </span>
              <span className="block mt-2 text-sm font-semibold text-slate-200">Test Water IoU</span>
              <span className="block text-xs text-slate-400 mt-1">Sentinel-2 Flood U-Net (Sen1Floods11)</span>
            </div>

            <div className="bg-slate-800/60 backdrop-blur-md border border-slate-700/70 p-6 rounded-2xl shadow-xl text-center">
              <span className="block text-4xl lg:text-5xl font-extrabold text-[#19C3B1]">
                96.5%
              </span>
              <span className="block mt-2 text-sm font-semibold text-slate-200">Weather Accuracy</span>
              <span className="block text-xs text-slate-400 mt-1">EfficientNet-B0 Sky Classifier + Grad-CAM</span>
            </div>

            <div className="bg-slate-800/60 backdrop-blur-md border border-slate-700/70 p-6 rounded-2xl shadow-xl text-center">
              <span className="block text-4xl lg:text-5xl font-extrabold text-[#FF8A3D]">
                0.828
              </span>
              <span className="block mt-2 text-sm font-semibold text-slate-200">Rain ROC-AUC</span>
              <span className="block text-xs text-slate-400 mt-1">24h Rain Predictor (10-Yr Chennai Data)</span>
            </div>
          </motion.div>
        </div>

        {/* Soft Wave SVG at bottom of hero */}
        <div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none pointer-events-none">
          <svg className="relative block w-full h-12 text-[#F6FAFD]" viewBox="0 0 1200 120" preserveAspectRatio="none">
            <path
              d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V120H0V95.8C59.71,118,152.47,117.48,216,104.53,252.12,97.16,287.41,72.76,321.39,56.44Z"
              fill="currentColor"
            />
          </svg>
        </div>
      </section>

      {/* 2. THE PROBLEM */}
      <section id="problem" className="py-20 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 scroll-mt-24">
        <div className="text-center max-w-3xl mx-auto">
          <span className="text-xs font-bold uppercase tracking-wider text-[#1D8CF8]">Motivation & Impact</span>
          <h2 className="mt-2 text-3xl sm:text-4xl font-extrabold text-[#0F2438]">Why Flood Mapping Needs AI</h2>
          <p className="mt-3 text-[#5B7083] text-base sm:text-lg">
            Rapid environmental disasters require instantaneous mapping to safeguard lives and water security.
          </p>
        </div>

        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8">
          <motion.div
            whileHover={{ y: -6 }}
            className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200/80 hover:shadow-xl transition-all"
          >
            <div className="w-12 h-12 rounded-xl bg-orange-50 border border-orange-200 text-[#FF8A3D] flex items-center justify-center mb-6">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-[#0F2438]">Floods Cause Huge Damage</h3>
            <p className="mt-3 text-[#5B7083] text-sm leading-relaxed">
              Disasters displace millions, inundate urban infrastructure, and destroy agricultural crops. Manual ground assessment is dangerous, slow, and often impossible during active storms.
            </p>
          </motion.div>

          <motion.div
            whileHover={{ y: -6 }}
            className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200/80 hover:shadow-xl transition-all"
          >
            <div className="w-12 h-12 rounded-xl bg-blue-50 border border-blue-200 text-[#1D8CF8] flex items-center justify-center mb-6">
              <Satellite className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-[#0F2438]">Why Satellites?</h3>
            <p className="mt-3 text-[#5B7083] text-sm leading-relaxed">
              Earth observation constellations like Sentinel-2 pass over any location every 5 days, providing 10-meter multispectral data for whole-basin emergency mapping without risking human life.
            </p>
          </motion.div>

          <motion.div
            whileHover={{ y: -6 }}
            className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200/80 hover:shadow-xl transition-all"
          >
            <div className="w-12 h-12 rounded-xl bg-teal-50 border border-teal-200 text-[#19C3B1] flex items-center justify-center mb-6">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold text-[#0F2438]">Why Deep Learning?</h3>
            <p className="mt-3 text-[#5B7083] text-sm leading-relaxed">
              Classical pixel thresholds (like NDWI) get confused by cloud shadows, sediment turbidity, and urban building reflections. U-Net learns spatial shapes, textures, and context.
            </p>
          </motion.div>
        </div>
      </section>

      {/* 3. HOW IT WORKS */}
      <section id="how-it-works" className="py-20 bg-slate-100/60 border-y border-slate-200/80 scroll-mt-24">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto">
            <span className="text-xs font-bold uppercase tracking-wider text-[#19C3B1]">Pipeline Overview</span>
            <h2 className="mt-2 text-3xl sm:text-4xl font-extrabold text-[#0F2438]">How AquaWatch Works</h2>
            <p className="mt-3 text-[#5B7083] text-base">A transparent, 4-step deep learning workflow from raw photons to actionable disaster maps.</p>
          </div>

          <div className="mt-14 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 relative">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center text-center">
              <div className="w-12 h-12 rounded-full bg-blue-100 text-[#1D8CF8] font-extrabold flex items-center justify-center text-lg mb-4">
                1
              </div>
              <h4 className="font-bold text-lg text-[#0F2438]">Satellite Image</h4>
              <p className="text-xs text-[#5B7083] mt-2">
                Ingests 4 Sentinel-2 optical bands (RGB + NIR) at 10m ground resolution, normalized from 10,000x TOA reflectance.
              </p>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center text-center">
              <div className="w-12 h-12 rounded-full bg-teal-100 text-[#19C3B1] font-extrabold flex items-center justify-center text-lg mb-4">
                2
              </div>
              <h4 className="font-bold text-lg text-[#0F2438]">U-Net AI Model</h4>
              <p className="text-xs text-[#5B7083] mt-2">
                A 34-layer ResNet encoder extracts multispectral features across scales, passing contextual skip connections to the decoder.
              </p>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center text-center">
              <div className="w-12 h-12 rounded-full bg-orange-100 text-[#FF8A3D] font-extrabold flex items-center justify-center text-lg mb-4">
                3
              </div>
              <h4 className="font-bold text-lg text-[#0F2438]">Water Mask</h4>
              <p className="text-xs text-[#5B7083] mt-2">
                Predicts a pixel-by-pixel binary classification (Water vs Land), ignoring cloud-occluded pixels through masked loss training.
              </p>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col items-center text-center">
              <div className="w-12 h-12 rounded-full bg-purple-100 text-[#8B5CF6] font-extrabold flex items-center justify-center text-lg mb-4">
                4
              </div>
              <h4 className="font-bold text-lg text-[#0F2438]">Change Report</h4>
              <p className="text-xs text-[#5B7083] mt-2">
                Compares masks between two acquisition dates, computing flooded vs receded areas in square kilometers (1 px = 100 m²).
              </p>
            </div>
          </div>

          {/* Expandable Beginner Explanation */}
          <div className="mt-10 max-w-3xl mx-auto">
            <button
              onClick={() => setLearnMoreOpen(!learnMoreOpen)}
              className="w-full bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-semibold py-3 px-5 rounded-xl shadow-sm flex items-center justify-between text-sm transition-all"
            >
              <span className="flex items-center space-x-2">
                <Info className="w-4 h-4 text-[#1D8CF8]" />
                <span>Beginner Guide: What is U-Net, IoU, and NDWI?</span>
              </span>
              <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${learnMoreOpen ? 'rotate-180' : ''}`} />
            </button>

            <AnimatePresence>
              {learnMoreOpen && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-white border-x border-b border-slate-200 rounded-b-xl p-6 text-sm text-[#5B7083] space-y-4 shadow-sm"
                >
                  <div>
                    <strong className="text-[#0F2438]">U-Net Architecture:</strong> A deep learning network shaped like a "U". The downsampling half compresses the satellite image to understand context ("is there a river or cloud here?"), while the upsampling half expands it back to reconstruct crisp boundaries with skip connections.
                  </div>
                  <div>
                    <strong className="text-[#0F2438]">Intersection-over-Union (IoU):</strong> Also called the Jaccard Index. It measures the overlap between AI predictions and human ground truth: <code className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-800">Overlap Area / (Predicted Area + True Area - Overlap Area)</code>. Unlike raw accuracy, IoU does not artificially inflate when water is a tiny fraction of the image.
                  </div>
                  <div>
                    <strong className="text-[#0F2438]">NDWI (Normalized Difference Water Index):</strong> The traditional mathematical formula <code className="bg-slate-100 px-1.5 py-0.5 rounded text-slate-800">(Green - NIR) / (Green + NIR)</code>. Water reflects green light but absorbs near-infrared. While fast, it has no spatial understanding and produces false alarms in shadow-heavy terrain.
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </section>

      {/* 3. LIVE BASIN MONITOR (NASA GIBS + RAINVIEWER + OPEN-METEO + RAINLSTM) */}
      <LiveMonitor apiBase={API_BASE} />

      {/* 4. WEATHER DETECTIVE (EFFICIENTNET-B0 + GRAD-CAM EXPLAINABILITY) */}
      <WeatherDetective apiBase={API_BASE} />

      {/* 5. SATELLITE FLOOD ANALYSIS */}
      <section id="flood-analysis" className="py-24 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 scroll-mt-24">
        <div id="demo" className="scroll-mt-24" />
        <div className="text-center max-w-3xl mx-auto mb-12">
          <span className="text-xs font-bold uppercase tracking-wider text-[#1D8CF8]">Interactive Lab</span>
          <h2 className="mt-2 text-3xl sm:text-4xl font-extrabold text-[#0F2438]">Live Satellite Flood Analysis</h2>
          <p className="mt-3 text-[#5B7083] text-base sm:text-lg">
            Select a region, adjust the slider to compare pre- and post-flood imagery, and explore automated surface water delta metrics.
          </p>
        </div>

        <div className="bg-white rounded-3xl p-6 sm:p-10 shadow-xl border border-slate-200/80 grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls Column (Left) */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-6">
            <div>
              {/* Region Selector */}
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
                Select Disaster Region
              </label>
              <div className="relative">
                <select
                  value={selectedRegionId}
                  onChange={(e) => setSelectedRegionId(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-4 pr-10 py-3 text-sm font-semibold text-[#0F2438] appearance-none focus:outline-none focus:ring-2 focus:ring-[#1D8CF8] truncate cursor-pointer shadow-sm hover:border-[#1D8CF8] transition-colors"
                  title="Select disaster region"
                >
                  {regions.map((r) => (
                    <option key={r.id} value={r.id} className="py-1 text-slate-800">
                      {r.name} • {r.event_label}
                    </option>
                  ))}
                </select>
                <ChevronDown className="w-4 h-4 text-slate-500 absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none" />
              </div>

              {/* Data Provenance & Transparency Card */}
              {regionData && (
                <div className="mt-3 bg-slate-50 border border-slate-200/90 rounded-2xl p-4 text-xs space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-1.5">
                    <span className="font-bold text-slate-800 flex items-center space-x-1.5 text-xs">
                      <MapPin className="w-3.5 h-3.5 text-[#1D8CF8]" />
                      <span>{regionData.state_country}</span>
                    </span>
                    {regionData.stats?.is_real_data || regionData.is_real_data ? (
                      <span className="inline-flex items-center space-x-1 text-[11px] font-bold text-emerald-800 bg-emerald-100 px-2.5 py-1 rounded-full border border-emerald-300 shadow-sm">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        <span>100% Real Sentinel-2 L2A</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 text-[11px] font-bold text-blue-800 bg-blue-100 px-2.5 py-1 rounded-full border border-blue-200 shadow-sm">
                        <Info className="w-3.5 h-3.5 text-blue-600" />
                        <span>Official Sen1Floods11 Chip</span>
                      </span>
                    )}
                  </div>
                  <p className="text-slate-600 text-xs leading-relaxed">
                    {regionData.description}
                  </p>
                  <div className="pt-2 border-t border-slate-200/80 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500 font-mono">
                    <span>Tile: <strong>{regionData.mgrs_tile || 'MGRS 44PLV'}</strong></span>
                    <span>Resolution: <strong>10m/pixel (Sentinel-2)</strong></span>
                  </div>
                </div>
              )}

              {/* Date Information */}
              <div className="mt-6 grid grid-cols-2 gap-3">
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                  <span className="text-[11px] font-bold text-slate-400 uppercase block">Before Date</span>
                  <div className="flex items-center space-x-1.5 mt-1 text-sm font-bold text-[#1D8CF8]">
                    <Calendar className="w-3.5 h-3.5" />
                    <span>{regionData?.dates?.[0] || "Nov 2023"}</span>
                  </div>
                </div>
                <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
                  <span className="text-[11px] font-bold text-slate-400 uppercase block">After Date</span>
                  <div className="flex items-center space-x-1.5 mt-1 text-sm font-bold text-[#FF8A3D]">
                    <Calendar className="w-3.5 h-3.5" />
                    <span>{regionData?.dates?.[1] || "Dec 2023"}</span>
                  </div>
                </div>
              </div>

              {/* View Switcher Tabs */}
              <div className="mt-6">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
                  Display Layer Mode
                </label>
                <div className="grid grid-cols-3 gap-2 bg-slate-100 p-1.5 rounded-xl text-xs font-semibold">
                  <button
                    onClick={() => setActiveViewMode("satellite")}
                    className={`py-2 px-3 rounded-lg transition-all ${
                      activeViewMode === "satellite"
                        ? "bg-white text-[#1D8CF8] shadow-sm font-bold"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    Satellite RGB
                  </button>
                  <button
                    onClick={() => setActiveViewMode("mask")}
                    className={`py-2 px-3 rounded-lg transition-all ${
                      activeViewMode === "mask"
                        ? "bg-white text-[#1D8CF8] shadow-sm font-bold"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    AI Water Mask
                  </button>
                  <button
                    onClick={() => setActiveViewMode("change")}
                    className={`py-2 px-3 rounded-lg transition-all ${
                      activeViewMode === "change"
                        ? "bg-white text-[#FF8A3D] shadow-sm font-bold"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    Change Map
                  </button>
                </div>
              </div>
            </div>

            {/* Quantitative Stats Card */}
            {regionData && (
              <div className="bg-slate-50 border border-slate-200/90 p-5 rounded-2xl space-y-4">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">
                  Measured Surface Water Metrics
                </span>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-xs text-slate-500 block">Baseline Water Area</span>
                    <span className="text-xl font-extrabold text-slate-800">
                      {regionData.stats.area_before_km2} km²
                    </span>
                  </div>
                  <div>
                    <span className="text-xs text-slate-500 block">Post-Flood Water Area</span>
                    <span className="text-xl font-extrabold text-[#FF8A3D]">
                      {regionData.stats.area_after_km2} km²
                    </span>
                  </div>
                </div>

                <div className="pt-3 border-t border-slate-200 flex items-center justify-between">
                  <div>
                    <span className="text-xs text-slate-500 block">Inundation Expansion</span>
                    <span className="text-sm font-bold text-emerald-600">
                      +{regionData.stats.flooded_area_km2} km² new water
                    </span>
                  </div>
                  <span className="bg-orange-100 text-[#FF8A3D] text-xs font-extrabold px-3 py-1 rounded-full border border-orange-200">
                    +{regionData.stats.pct_change}% Change
                  </span>
                </div>
              </div>
            )}

            {/* Color Legend */}
            <div className="bg-slate-900 text-white p-4 rounded-xl text-xs space-y-2 border border-slate-800">
              <span className="font-bold text-slate-300 block mb-1">Color Key:</span>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-sm bg-[#FF8A3D]" />
                  <span className="text-slate-300 font-medium">New Flooding</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-sm bg-[#1D8CF8]" />
                  <span className="text-slate-300 font-medium">Permanent Water</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-sm bg-[#8B5CF6]" />
                  <span className="text-slate-300 font-medium">Water Lost / Receded</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="w-3 h-3 rounded-sm bg-[#1E293B] border border-slate-600" />
                  <span className="text-slate-300 font-medium">Dry Land</span>
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Comparison Slider Column (Right) */}
          <div className="lg:col-span-7 flex flex-col items-center justify-center">
            {images.before && images.after ? (
              <ImageSlider
                beforeImage={images.before}
                afterImage={images.after}
                beforeLabel={activeViewMode === "change" ? "Baseline RGB" : "Before Flood"}
                afterLabel={activeViewMode === "change" ? "AI Flood Change Map" : "After Flood"}
              />
            ) : (
              <div className="w-full aspect-square rounded-2xl bg-slate-100 border-2 border-dashed border-slate-300 flex items-center justify-center text-slate-400">
                Loading satellite chips...
              </div>
            )}
            <p className="mt-3 text-xs text-slate-400 flex items-center space-x-1.5">
              <Sliders className="w-3.5 h-3.5 text-[#1D8CF8]" />
              <span>Drag the vertical handle left or right to inspect inundation contours</span>
            </p>
          </div>
        </div>

        {/* Bonus Live Upload Component */}
        <div className="mt-12 bg-white rounded-3xl p-8 border border-slate-200/90 shadow-sm max-w-4xl mx-auto">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-9 h-9 rounded-xl bg-blue-50 text-[#1D8CF8] flex items-center justify-center">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-[#0F2438]">Upload Custom Satellite Image (Live Inference)</h3>
              <p className="text-xs text-[#5B7083]">Test the trained U-Net model on a local Sentinel-2 GeoTIFF or RGB image.</p>
            </div>
          </div>

          <div className="border-2 border-dashed border-slate-300 hover:border-[#1D8CF8] transition-colors rounded-2xl p-6 text-center cursor-pointer relative bg-slate-50/50">
            <input
              type="file"
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            <p className="text-sm font-semibold text-slate-700">Drop a GeoTIFF (.tif) or standard image here, or click to browse</p>
            <p className="text-xs text-slate-400 mt-1">Accepts 13-band Sentinel-2 or standard 4-band/RGB crops (up to 15MB)</p>
          </div>

          {uploadStatus && (
            <p className="text-xs text-slate-600 font-medium mt-3 text-center">
              {uploadStatus}
            </p>
          )}

          {uploadResult && (
            <div className="mt-6 p-4 bg-slate-50 border border-slate-200 rounded-2xl flex flex-col sm:flex-row items-center gap-6">
              <img
                src={uploadResult.mask_base64}
                alt="Predicted Mask"
                className="w-48 h-48 rounded-xl shadow border border-slate-200 object-cover"
              />
              <div className="space-y-2 text-sm text-slate-700">
                <span className="font-bold text-[#0F2438] block text-base">Prediction Summary</span>
                <p>Water Coverage: <strong className="text-[#1D8CF8]">{uploadResult.water_percentage}%</strong></p>
                <p>Estimated Water Area: <strong>{uploadResult.estimated_water_area_km2} km²</strong></p>
                <p>Total Water Pixels: <strong>{uploadResult.water_pixels?.toLocaleString()} px</strong></p>
                <span className="inline-block mt-2 text-xs font-semibold px-2.5 py-1 bg-emerald-100 text-emerald-800 rounded-full">
                  Inference Complete (ResNet34 Backbone)
                </span>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* 5. WATER OVER TIME */}
      <section id="timeseries" className="py-20 bg-slate-100/60 border-y border-slate-200/80 scroll-mt-24">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-2xl mx-auto mb-10">
            <span className="text-xs font-bold uppercase tracking-wider text-[#19C3B1]">Long-Term Monitoring</span>
            <h2 className="mt-2 text-3xl sm:text-4xl font-extrabold text-[#0F2438]">Water Surface Area Over Time</h2>
            <p className="mt-3 text-[#5B7083] text-base">
              Real multi-temporal Sentinel-2 Level-2A observations (Tile 44PLV) across 12 cloud-filtered satellite passes in 2023, capturing the seasonal cycle and Cyclone Michaung flood expansion.
            </p>
          </div>

          <div className="bg-white p-6 sm:p-8 rounded-3xl shadow-lg border border-slate-200/90">
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={timeseriesData?.data || []}
                  margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="waterAreaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#1D8CF8" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#1D8CF8" stopOpacity={0.05}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="month" stroke="#64748B" fontSize={12} />
                  <YAxis stroke="#64748B" fontSize={12} unit=" km²" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0B1F33', border: 'none', borderRadius: '12px', color: '#fff' }}
                    itemStyle={{ color: '#19C3B1' }}
                  />
                  <ReferenceLine
                    y={24.5}
                    label={{ value: 'Full Reservoir Capacity (24.5 km²)', fill: '#FF8A3D', fontSize: 11 }}
                    stroke="#FF8A3D"
                    strokeDasharray="4 4"
                  />
                  <Area
                    type="monotone"
                    dataKey="area_km2"
                    name="Water Area"
                    stroke="#1D8CF8"
                    strokeWidth={3}
                    fillOpacity={1}
                    fill="url(#waterAreaGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-6 flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-orange-50 border border-orange-200 text-xs text-orange-950">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-[#FF8A3D]" />
                <span className="font-bold">Marked Flood Anomaly: Cyclone Michaung (Dec 2023)</span>
              </div>
              <span>Water extent expanded by <strong>+21.2 km² (+75.3%)</strong> during inundation, swelling the lake and surrounding Adyar lowlands to <strong>46.18 km²</strong> (far exceeding the 24.5 km² full reservoir capacity).</span>
            </div>
          </div>
        </div>
      </section>

      {/* 6. MULTI-MODEL BENCHMARKS & EVALUATION */}
      <ResultsMultiModel
        apiBase={API_BASE}
        floodMetrics={metricsData}
        baselineMetrics={metricsData?.baseline}
      />

      {/* 8. ABOUT & TECH STACK */}
      <section id="about" className="py-20 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 scroll-mt-24">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <span className="text-xs font-bold uppercase tracking-wider text-[#1D8CF8]">Architecture & Sources</span>
          <h2 className="mt-2 text-3xl sm:text-4xl font-extrabold text-[#0F2438]">Technology & Data Sources</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-4">
            <h3 className="font-bold text-lg text-[#0F2438] flex items-center space-x-2">
              <Database className="w-5 h-5 text-[#1D8CF8]" />
              <span>Official Public Datasets</span>
            </h3>
            <ul className="text-xs text-[#5B7083] space-y-2.5">
              <li>
                <strong className="text-slate-800">Sen1Floods11 (v1.1):</strong> Hand-labeled benchmark dataset by Cloud to Street (CVPRW 2020), hosted on public GCS bucket <code className="bg-slate-100 px-1 py-0.5 rounded">gs://sen1floods11</code>.
              </li>
              <li>
                <strong className="text-slate-800">Copernicus Sentinel-2:</strong> Multispectral MSI instrument at 10m spatial resolution across 13 spectral bands.
              </li>
              <li>
                <strong className="text-slate-800">Google Earth Engine:</strong> Harmonized Surface Reflectance (<code className="bg-slate-100 px-1 py-0.5 rounded">COPERNICUS/S2_SR_HARMONIZED</code>) for local Chennai monitoring.
              </li>
            </ul>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm space-y-4">
            <h3 className="font-bold text-lg text-[#0F2438] flex items-center space-x-2">
              <Layers className="w-5 h-5 text-[#19C3B1]" />
              <span>Deep Learning & Web Stack</span>
            </h3>
            <div className="flex flex-wrap gap-2 text-xs font-semibold">
              <span className="bg-blue-50 text-[#1D8CF8] px-3 py-1.5 rounded-lg border border-blue-200">PyTorch 2.2</span>
              <span className="bg-teal-50 text-[#19C3B1] px-3 py-1.5 rounded-lg border border-teal-200">U-Net (ResNet34)</span>
              <span className="bg-purple-50 text-[#8B5CF6] px-3 py-1.5 rounded-lg border border-purple-200">Segmentation Models PyTorch</span>
              <span className="bg-slate-100 text-slate-800 px-3 py-1.5 rounded-lg border border-slate-300">Albumentations</span>
              <span className="bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-lg border border-emerald-200">FastAPI</span>
              <span className="bg-cyan-50 text-cyan-700 px-3 py-1.5 rounded-lg border border-cyan-200">React + Vite</span>
              <span className="bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-lg border border-indigo-200">Tailwind CSS</span>
              <span className="bg-orange-50 text-[#FF8A3D] px-3 py-1.5 rounded-lg border border-orange-200">Recharts</span>
            </div>
            <div className="pt-2 text-xs text-slate-400">
              College Major Project • Prepared for Presentation & Viva Voce Review
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="bg-[#0B1F33] text-slate-400 py-12 border-t border-slate-800 text-xs">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-3">
          <p className="font-semibold text-slate-300">
            AquaWatch: Deep Learning for Flood and Waterbody Change Detection using Sentinel Satellite Imagery
          </p>
          <p>
            Contains modified Copernicus Sentinel data [2015–2023]. Dataset provided by Cloud to Street (CVPR Workshops 2020).
          </p>
          <p className="text-slate-500 text-[11px]">
            Academic disclaimer: This system is developed for academic evaluation, research, and technical demonstration. It is not an official government flood forecasting or civil protection system.
          </p>
        </div>
      </footer>
    </div>
  )
}
