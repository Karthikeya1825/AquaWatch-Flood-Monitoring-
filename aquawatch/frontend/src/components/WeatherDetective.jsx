import React, { useState, useEffect, useRef } from 'react'
import {
  Cloud,
  CloudRain,
  Sun,
  Sunrise,
  Upload,
  Camera,
  CheckCircle2,
  Sparkles,
  Eye,
  AlertCircle,
  HelpCircle,
  ArrowRight
} from 'lucide-react'

export default function WeatherDetective({ apiBase = "http://127.0.0.1:8000" }) {
  const [samples, setSamples] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUri, setPreviewUri] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showGradCam, setShowGradCam] = useState(true)
  const fileInputRef = useRef(null)
  const cameraInputRef = useRef(null)

  // Fetch sample test images on load
  useEffect(() => {
    async function loadSamples() {
      try {
        const res = await fetch(`${apiBase}/api/weather/samples`)
        if (res.ok) {
          const data = await res.json()
          setSamples(data)
          // Automatically pick the first sample (e.g. Rain or Cloudy) for zero-click immediate demo
          if (data.length > 0) {
            handleSampleClick(data[1] || data[0])
          }
        }
      } catch (e) {
        console.error("Could not load weather samples:", e)
      }
    }
    loadSamples()
  }, [apiBase])

  const getWeatherIcon = (className, sizeClass = "w-8 h-8") => {
    switch (className?.toLowerCase()) {
      case 'rain':
        return <CloudRain className={`${sizeClass} text-blue-400`} />
      case 'cloudy':
        return <Cloud className={`${sizeClass} text-slate-300`} />
      case 'shine':
      case 'sunny':
        return <Sun className={`${sizeClass} text-amber-400`} />
      case 'sunrise':
        return <Sunrise className={`${sizeClass} text-orange-400`} />
      default:
        return <Sparkles className={`${sizeClass} text-cyan-400`} />
    }
  }

  // Handle uploaded file or camera capture
  const handleFile = async (file) => {
    if (!file) return
    setError(null)
    setAnalyzing(true)
    setResult(null)

    // Local preview
    const reader = new FileReader()
    reader.onload = (e) => setPreviewUri(e.target.result)
    reader.readAsDataURL(file)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch(`${apiBase}/api/weather/classify`, {
        method: "POST",
        body: formData
      })
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}))
        throw new Error(errJson.detail || `Server returned status ${res.status}`)
      }
      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error("Classification error:", err)
      setError(err.message || "Failed to classify image.")
    } finally {
      setAnalyzing(false)
    }
  }

  // Handle clicking one of the 4 test sample chips
  const handleSampleClick = async (sample) => {
    setError(null)
    setAnalyzing(true)
    setResult(null)
    setPreviewUri(sample.base64 || `${apiBase}${sample.url}`)

    try {
      // Convert base64 data URI to Blob
      const resBlob = await fetch(sample.base64 || `${apiBase}${sample.url}`)
      const blob = await resBlob.blob()
      const file = new File([blob], sample.filename, { type: "image/jpeg" })

      const formData = new FormData()
      formData.append("file", file)

      const res = await fetch(`${apiBase}/api/weather/classify`, {
        method: "POST",
        body: formData
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error("Sample classify error:", err)
      setError("Could not analyze sample image.")
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <section id="weather-detective" className="py-20 bg-slate-950 text-slate-100 scroll-mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-3xl mx-auto mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            Computer Vision & Grad-CAM Explainability
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Weather Detective: Sky & Cloud Classifier
          </h2>
          <p className="mt-3 text-slate-400 text-sm sm:text-base">
            Upload an outdoor sky image or capture from mobile camera. Our fine-tuned EfficientNet-B0
            identifies the weather condition in real time and highlights the exact cloud/sun regions driving the prediction.
          </p>
        </div>

        {/* 4 Clickable Quick Samples */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              ⚡ Instant 1-Click Verification Samples (From Kaggle Test Split):
            </span>
            <span className="text-xs text-slate-500 hidden sm:inline">
              Test set accuracy: 96.45% &bull; Zero data leakage
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {samples.map((s, idx) => (
              <button
                key={idx}
                onClick={() => handleSampleClick(s)}
                disabled={analyzing}
                className="group relative flex items-center gap-3 p-3 rounded-xl bg-slate-900 border border-slate-800 hover:border-cyan-500/60 hover:bg-slate-800/80 transition text-left overflow-hidden disabled:opacity-50"
              >
                <div className="w-12 h-12 rounded-lg overflow-hidden flex-shrink-0 bg-slate-800 border border-slate-700">
                  <img
                    src={s.base64 || `${apiBase}${s.url}`}
                    alt={s.class}
                    className="w-full h-full object-cover group-hover:scale-110 transition duration-300"
                  />
                </div>
                <div>
                  <span className="text-xs font-bold text-white group-hover:text-cyan-400 transition block">
                    {s.class}
                  </span>
                  <span className="text-[11px] text-slate-400">Click to run</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Main Interface: Upload / Dropzone + Results Card */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Left Column: Drag & Drop / Mobile Capture */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 flex flex-col justify-between min-h-[460px]">
            <div>
              <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Upload className="w-5 h-5 text-cyan-400" />
                Upload Sky Image
              </h3>
              <p className="text-xs text-slate-400 mb-5">
                Supports JPG, PNG, WEBP (Max 10MB). Automatically normalized to 224x224 with ImageNet parameters.
              </p>

              {/* Dropzone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0])
                }}
                className="border-2 border-dashed border-slate-700 hover:border-cyan-500 rounded-xl p-8 text-center cursor-pointer transition bg-slate-950/50"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFile(e.target.files[0])
                  }}
                />
                <input
                  type="file"
                  ref={cameraInputRef}
                  className="hidden"
                  accept="image/*"
                  capture="environment"
                  onChange={(e) => {
                    if (e.target.files?.[0]) handleFile(e.target.files[0])
                  }}
                />

                <div className="w-14 h-14 mx-auto rounded-full bg-slate-800/80 flex items-center justify-center text-cyan-400 mb-3 border border-slate-700">
                  <Upload className="w-6 h-6" />
                </div>
                <p className="text-sm font-semibold text-slate-200">
                  Drag and drop sky photo here, or <span className="text-cyan-400">browse</span>
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Works on clouds, rain, direct sunshine, or sunrise/sunset horizons
                </p>
              </div>

              {/* Camera Trigger for Mobile */}
              <div className="mt-4 flex gap-3">
                <button
                  type="button"
                  onClick={() => cameraInputRef.current?.click()}
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-xs font-semibold border border-slate-700 transition"
                >
                  <Camera className="w-4 h-4 text-cyan-400" />
                  Take Live Photo on Camera
                </button>
              </div>
            </div>

            {/* Educational Model Callout */}
            <div className="mt-6 pt-4 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Deep Learning Backbone: EfficientNet-B0 (5.3M params)</span>
              <span className="text-cyan-400 font-semibold">Trained with AdamW & Grad-CAM</span>
            </div>
          </div>

          {/* Right Column: Dynamic Analysis & Grad-CAM View */}
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 min-h-[460px] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Eye className="w-5 h-5 text-cyan-400" />
                  Classification & Visual Attribution
                </h3>

                {result && (
                  <div className="flex items-center gap-2">
                    <label className="inline-flex items-center gap-2 text-xs font-semibold text-slate-300 cursor-pointer bg-slate-800 px-3 py-1 rounded-lg border border-slate-700">
                      <input
                        type="checkbox"
                        checked={showGradCam}
                        onChange={(e) => setShowGradCam(e.target.checked)}
                        className="rounded text-cyan-500 focus:ring-0 bg-slate-900 border-slate-600"
                      />
                      🔥 Show Grad-CAM
                    </label>
                  </div>
                )}
              </div>

              {/* Loading State */}
              {analyzing && (
                <div className="h-64 flex flex-col items-center justify-center text-center">
                  <div className="w-12 h-12 border-4 border-cyan-500/20 border-t-cyan-400 rounded-full animate-spin mb-4"></div>
                  <p className="text-sm font-semibold text-slate-200">
                    EfficientNet-B0 Inference & Gradient Backpropagation...
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Generating class activation map across conv_head feature maps
                  </p>
                </div>
              )}

              {/* Error State */}
              {error && !analyzing && (
                <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {/* Prediction Result Display */}
              {result && !analyzing && (
                <div className="space-y-5">
                  {/* Image Display: Original vs Grad-CAM heatmap */}
                  <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-black aspect-video max-h-56 flex items-center justify-center">
                    <img
                      src={
                        showGradCam && result.gradcam_base64
                          ? result.gradcam_base64
                          : result.original_base64 || previewUri
                      }
                      alt="Weather classification"
                      className="w-full h-full object-contain"
                    />

                    <div className="absolute bottom-2 left-2 px-2.5 py-1 rounded-md bg-slate-950/80 backdrop-blur border border-slate-800 text-[10px] font-semibold text-white">
                      {showGradCam ? "🔥 Grad-CAM Attention Heatmap" : "📷 Original Input"}
                    </div>

                    <div className="absolute top-2 right-2 px-2.5 py-1 rounded-md bg-cyan-500/90 text-white font-extrabold text-xs shadow-lg">
                      {result.top_prediction} ({result.confidence}%)
                    </div>
                  </div>

                  {/* Top Prediction Banner */}
                  <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-3.5">
                      <div className="p-2.5 rounded-xl bg-slate-800 border border-slate-700">
                        {getWeatherIcon(result.top_prediction)}
                      </div>
                      <div>
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                          Detected Condition
                        </span>
                        <h4 className="text-xl font-extrabold text-white">
                          {result.top_prediction}
                        </h4>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-2xl font-black text-cyan-400">
                        {result.confidence}%
                      </span>
                      <span className="text-[11px] text-slate-400 block font-medium">Confidence</span>
                    </div>
                  </div>

                  {/* Confidence Distribution Bars */}
                  <div>
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-2">
                      Softmax Probability Distribution (Top-3):
                    </span>
                    <div className="space-y-2">
                      {result.predictions?.map((pred, i) => (
                        <div key={i} className="text-xs">
                          <div className="flex justify-between text-slate-300 font-semibold mb-1">
                            <span>{pred.class}</span>
                            <span className="text-cyan-400">{pred.confidence}%</span>
                          </div>
                          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500"
                              style={{ width: `${pred.confidence}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Placeholder before running */}
              {!result && !analyzing && !error && (
                <div className="h-64 flex flex-col items-center justify-center text-center text-slate-500">
                  <Eye className="w-10 h-10 mb-2 opacity-40" />
                  <p className="text-sm">Select a test sample or upload an image to see deep learning inference</p>
                </div>
              )}
            </div>

            <p className="text-[11px] text-slate-500 mt-4 pt-3 border-t border-slate-800 text-center">
              Grad-CAM computes gradients with respect to EfficientNet-B0's final convolutional layer to reveal spatial salience.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
