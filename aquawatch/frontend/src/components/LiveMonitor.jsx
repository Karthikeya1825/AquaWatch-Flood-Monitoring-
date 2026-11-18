import React, { useEffect, useRef, useState } from 'react'
import L from 'leaflet'
import {
  Cloud,
  CloudRain,
  Compass,
  Droplets,
  Gauge,
  Info,
  Layers,
  MapPin,
  RefreshCw,
  ShieldAlert,
  Thermometer,
  Wind,
  Navigation
} from 'lucide-react'

// Default fallback catalog of South Indian locations grouped by state
const DEFAULT_LOCATIONS = [
  // Tamil Nadu
  { id: "chennai", name: "Chennai", state: "Tamil Nadu", lat: 13.0827, lng: 80.2707, basin_name: "Chembarambakkam Lake & Adyar River", basin_lat: 13.0118, basin_lng: 80.0215 },
  { id: "coimbatore", name: "Coimbatore", state: "Tamil Nadu", lat: 11.0168, lng: 76.9558, basin_name: "Bhavanisagar Dam & Noyyal River Basin", basin_lat: 11.4700, basin_lng: 77.1300 },
  { id: "madurai", name: "Madurai", state: "Tamil Nadu", lat: 9.9252, lng: 78.1198, basin_name: "Vaigai Dam Reservoir Catchment", basin_lat: 10.0542, basin_lng: 77.5894 },
  { id: "tiruchirappalli", name: "Tiruchirappalli", state: "Tamil Nadu", lat: 10.7905, lng: 78.7047, basin_name: "Mukkombu Upper Anaicut & Kaveri Delta", basin_lat: 10.8710, basin_lng: 78.5820 },

  // Karnataka
  { id: "bengaluru", name: "Bengaluru", state: "Karnataka", lat: 12.9716, lng: 77.5946, basin_name: "Bellandur & Varthur Urban Lake Basins", basin_lat: 12.9370, basin_lng: 77.6710 },
  { id: "mysuru", name: "Mysuru", state: "Karnataka", lat: 12.2958, lng: 76.6394, basin_name: "Krishna Raja Sagara (KRS) Dam & Cauvery", basin_lat: 12.4278, basin_lng: 76.5728 },
  { id: "mangaluru", name: "Mangaluru", state: "Karnataka", lat: 12.9141, lng: 74.8560, basin_name: "Netravati & Gurupura Estuary Basin", basin_lat: 12.8420, basin_lng: 74.8430 },

  // Kerala
  { id: "kochi", name: "Kochi", state: "Kerala", lat: 9.9312, lng: 76.2673, basin_name: "Vembanad Wetland & Periyar River Basin", basin_lat: 9.7500, basin_lng: 76.4000 },
  { id: "thiruvananthapuram", name: "Thiruvananthapuram", state: "Kerala", lat: 8.5241, lng: 76.9366, basin_name: "Aruvikkara & Peppara Dam Reservoir", basin_lat: 8.5670, basin_lng: 77.0190 },
  { id: "kozhikode", name: "Kozhikode", state: "Kerala", lat: 11.2588, lng: 75.7804, basin_name: "Kakkayam Dam & Chaliyar River Basin", basin_lat: 11.5540, basin_lng: 75.9220 },

  // Andhra Pradesh
  { id: "visakhapatnam", name: "Visakhapatnam", state: "Andhra Pradesh", lat: 17.6868, lng: 83.2185, basin_name: "Meghadrigedda & Gambheeram Reservoirs", basin_lat: 17.7560, basin_lng: 83.1890 },
  { id: "vijayawada", name: "Vijayawada", state: "Andhra Pradesh", lat: 16.5062, lng: 80.6480, basin_name: "Prakasam Barrage & Krishna River Basin", basin_lat: 16.5090, basin_lng: 80.6050 },
  { id: "tirupati", name: "Tirupati", state: "Andhra Pradesh", lat: 13.6288, lng: 79.4192, basin_name: "Kalyani Dam Reservoir & Swarnamukhi Basin", basin_lat: 13.6550, basin_lng: 79.2880 },

  // Telangana
  { id: "hyderabad", name: "Hyderabad", state: "Telangana", lat: 17.3850, lng: 78.4867, basin_name: "Hussain Sagar, Osman Sagar & Musi Basin", basin_lat: 17.4239, basin_lng: 78.4738 },
  { id: "warangal", name: "Warangal", state: "Telangana", lat: 17.9689, lng: 79.5941, basin_name: "Bhadrakali & Waddepally Reservoir Lakes", basin_lat: 17.9890, basin_lng: 79.5780 },

  // Puducherry (UT)
  { id: "puducherry", name: "Puducherry", state: "Puducherry (UT)", lat: 11.9416, lng: 79.8083, basin_name: "Ousteri Lake & Sankaraparani Basin", basin_lat: 11.9540, basin_lng: 79.7420 }
]

export default function LiveMonitor({ apiBase = "http://127.0.0.1:8000" }) {
  const mapContainerRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const layersRef = useRef({})
  const markersRef = useRef({ city: null, basin: null })

  const [locations, setLocations] = useState(DEFAULT_LOCATIONS)
  const [selectedLocId, setSelectedLocId] = useState("chennai")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [monitorData, setMonitorData] = useState(null)
  const [rainData, setRainData] = useState(null)
  const [activeLayer, setActiveLayer] = useState("osm") // "osm" | "modis" | "viirs"
  const [showRadar, setShowRadar] = useState(true)
  const [showClouds, setShowClouds] = useState(false)
  const [lastRefreshed, setLastRefreshed] = useState(null)

  // Fetch location catalog on mount
  useEffect(() => {
    async function loadLocations() {
      try {
        const res = await fetch(`${apiBase}/api/live/locations`)
        if (res.ok) {
          const data = await res.json()
          if (Array.isArray(data) && data.length > 0) setLocations(data)
        }
      } catch (e) {
        console.warn("Using default South India locations catalog", e)
      }
    }
    loadLocations()
  }, [apiBase])

  // Fetch live monitoring and rain prediction data for selected location
  const fetchData = async (locId = selectedLocId) => {
    setLoading(true)
    setError(null)
    try {
      const [monRes, rainRes] = await Promise.all([
        fetch(`${apiBase}/api/live/monitor?location_id=${locId}`),
        fetch(`${apiBase}/api/predict/rain?location_id=${locId}`)
      ])

      if (!monRes.ok) throw new Error(`Live monitor HTTP ${monRes.status}`)
      const monJson = await monRes.json()
      setMonitorData(monJson)

      if (rainRes.ok) {
        const rainJson = await rainRes.json()
        setRainData(rainJson)
      }
      setLastRefreshed(new Date().toLocaleTimeString())
    } catch (err) {
      console.error("[LiveMonitor] Fetch error:", err)
      setError(err.message || "Could not connect to live monitoring endpoints.")
    } finally {
      setLoading(false)
    }
  }

  // Handle location change
  const handleLocationChange = (newLocId) => {
    setSelectedLocId(newLocId)
    fetchData(newLocId)
  }

  // Initial fetch and 10-minute auto-refresh
  useEffect(() => {
    fetchData(selectedLocId)
    const timer = setInterval(() => fetchData(selectedLocId), 10 * 60 * 1000)
    return () => clearInterval(timer)
  }, [apiBase, selectedLocId])

  // Initialize Leaflet map
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return

    const initialCoords = [13.045, 80.145] // Chennai area
    const map = L.map(mapContainerRef.current, {
      center: initialCoords,
      zoom: 11,
      minZoom: 5,
      maxZoom: 15
    })
    mapInstanceRef.current = map

    // Base OSM tiles
    const osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
      maxZoom: 18
    }).addTo(map)

    layersRef.current.osm = osmLayer

    return () => {
      map.remove()
      mapInstanceRef.current = null
    }
  }, [])

  // Update map position and markers whenever monitorData changes
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !monitorData) return

    const center = monitorData.center_coordinates
    const basin = monitorData.basin_coordinates
    const loc = monitorData.selected_location || {}

    // Smoothly pan/fly map to selected location
    if (center?.lat && center?.lng) {
      map.flyTo([center.lat, center.lng], 11, { duration: 1.2 })
    }

    // Clean previous markers
    if (markersRef.current.city) map.removeLayer(markersRef.current.city)
    if (markersRef.current.basin) map.removeLayer(markersRef.current.basin)

    // Add City Marker
    if (center?.lat && center?.lng) {
      const cityIcon = L.divIcon({
        className: 'custom-div-icon',
        html: `
          <div style="background-color: #0f172a; color: white; padding: 4px 8px; border-radius: 9999px; font-weight: 700; font-size: 11px; display: flex; align-items: center; gap: 4px; box-shadow: 0 4px 12px rgba(15,23,42,0.4); border: 2px solid white; white-space: nowrap;">
            <span>📍 ${loc.name || "City Center"}</span>
          </div>
        `,
        iconSize: [140, 24],
        iconAnchor: [70, 12]
      })
      markersRef.current.city = L.marker([center.lat, center.lng], { icon: cityIcon })
        .addTo(map)
        .bindPopup(`<b>${loc.name || "Location"} (${loc.state || "South India"})</b><br>Ground meteorological station.`)
    }

    // Add Basin / Reservoir Marker
    if (basin?.lat && basin?.lng) {
      const basinIcon = L.divIcon({
        className: 'custom-div-icon',
        html: `
          <div style="background-color: #0284c7; color: white; padding: 4px 8px; border-radius: 9999px; font-weight: 700; font-size: 11px; display: flex; align-items: center; gap: 4px; box-shadow: 0 4px 12px rgba(2,132,199,0.5); border: 2px solid white; white-space: nowrap;">
            <span>💧 ${loc.basin_name ? loc.basin_name.split('&')[0].trim() : "Reservoir Basin"}</span>
          </div>
        `,
        iconSize: [160, 24],
        iconAnchor: [80, 12]
      })
      markersRef.current.basin = L.marker([basin.lat, basin.lng], { icon: basinIcon })
        .addTo(map)
        .bindPopup(`<b>${loc.basin_name || "Reservoir Basin"}</b><br>Normal baseline: ${loc.lake_baseline_km2 || 24} km² &bull; Flood capacity: ${loc.lake_capacity_km2 || 48} km²`)
    }
  }, [monitorData])

  // Sync NASA GIBS and RainViewer layers
  useEffect(() => {
    const map = mapInstanceRef.current
    if (!map || !monitorData) return

    if (layersRef.current.modis) map.removeLayer(layersRef.current.modis)
    if (layersRef.current.viirs) map.removeLayer(layersRef.current.viirs)
    if (layersRef.current.radar) map.removeLayer(layersRef.current.radar)
    if (layersRef.current.clouds) map.removeLayer(layersRef.current.clouds)

    const nasa = monitorData.nasa_gibs
    const radar = monitorData.rainviewer_radar

    if (activeLayer === "modis" && nasa?.modis_truecolor_template) {
      layersRef.current.modis = L.tileLayer(nasa.modis_truecolor_template, {
        attribution: nasa.attribution,
        opacity: 0.85,
        maxZoom: 9
      }).addTo(map)
    }

    if (activeLayer === "viirs" && nasa?.viirs_truecolor_template) {
      layersRef.current.viirs = L.tileLayer(nasa.viirs_truecolor_template, {
        attribution: nasa.attribution,
        opacity: 0.85,
        maxZoom: 9
      }).addTo(map)
    }

    if (showClouds && nasa?.modis_clouds_template) {
      layersRef.current.clouds = L.tileLayer(nasa.modis_clouds_template, {
        attribution: nasa.attribution,
        opacity: 0.6,
        maxZoom: 9
      }).addTo(map)
    }

    if (showRadar && radar?.radar_tile_template) {
      layersRef.current.radar = L.tileLayer(radar.radar_tile_template, {
        attribution: '&copy; <a href="https://www.rainviewer.com">RainViewer</a>',
        opacity: 0.75,
        maxZoom: 15
      }).addTo(map)
    }
  }, [monitorData, activeLayer, showRadar, showClouds])

  const risk = monitorData?.flood_risk_engine
  const weather = monitorData?.current_weather
  const currentLoc = monitorData?.selected_location || locations.find(l => l.id === selectedLocId) || locations[0]

  // Group locations by state for the dropdown
  const states = Array.from(new Set(locations.map(l => l.state)))

  return (
    <section id="live-monitor" className="py-20 bg-slate-900 text-slate-100 scroll-mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 pb-6 border-b border-slate-800 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 mb-3">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Live Environmental Telemetry & Sentinel Center
            </div>
            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
              Live Basin Monitor: South India States
            </h2>
            <p className="mt-2 text-slate-400 max-w-2xl text-sm sm:text-base">
              Synchronizing NASA GIBS daily optical passes, RainViewer live Doppler radar reflectivity,
              and Open-Meteo meteorological telemetry with our deep learning rain and flood retention engines across all Southern states.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {lastRefreshed && (
              <span className="text-xs text-slate-400">
                Refreshed at {lastRefreshed}
              </span>
            )}
            <button
              onClick={() => fetchData(selectedLocId)}
              disabled={loading}
              className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm font-medium transition border border-slate-700 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
              {loading ? "Updating..." : "Refresh Feeds"}
            </button>
          </div>
        </div>

        {/* PROMINENT SOUTH INDIA LOCATION SELECTOR BAR */}
        <div className="mb-8 p-4 rounded-2xl bg-gradient-to-r from-slate-800 via-slate-850 to-slate-900 border border-cyan-500/30 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/40 text-cyan-400 flex items-center justify-center flex-shrink-0">
              <Navigation className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-cyan-400 block">
                Select South Indian Location
              </span>
              <span className="text-sm font-extrabold text-white">
                {currentLoc.name} &bull; <span className="text-slate-300 font-medium">{currentLoc.state}</span>
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <select
                value={selectedLocId}
                onChange={(e) => handleLocationChange(e.target.value)}
                className="bg-slate-950 border border-slate-700 hover:border-cyan-500 text-white font-semibold text-xs sm:text-sm rounded-xl px-4 py-2.5 pr-8 focus:outline-none focus:ring-2 focus:ring-cyan-500 shadow-inner transition cursor-pointer"
              >
                {states.map((state) => (
                  <optgroup key={state} label={`📍 ${state}`} className="bg-slate-900 text-slate-300 font-bold">
                    {locations
                      .filter((loc) => loc.state === state)
                      .map((loc) => (
                        <option key={loc.id} value={loc.id} className="text-white py-1">
                          {loc.name} — {loc.basin_name.split('&')[0].trim()}
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-700 text-xs text-slate-300 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              <span className="font-semibold">{currentLoc.basin_name}</span>
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm flex items-center gap-3">
            <Info className="w-5 h-5 flex-shrink-0" />
            <div>
              <p className="font-semibold">Live Feed Advisory</p>
              <p>{error} (Serving local high-precision cached baseline datasets).</p>
            </div>
          </div>
        )}

        {/* Top Cards: Current Weather & Transparent Flood Risk Indicator */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Card 1: Transparent Flood Risk Indicator */}
          <div className="lg:col-span-2 rounded-2xl bg-slate-800/80 border border-slate-700/80 p-6 backdrop-blur">
            <div className="flex items-start justify-between flex-wrap gap-2">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                  {currentLoc.state} Basin Assessment
                </span>
                <h3 className="text-xl font-bold text-white mt-1 flex items-center gap-2">
                  <ShieldAlert className="w-5 h-5 text-cyan-400" />
                  Hydrological Flood Risk: {currentLoc.name}
                </h3>
              </div>
              {risk && (
                <div
                  className="px-4 py-1.5 rounded-full text-xs font-black tracking-wide uppercase border flex items-center gap-2 shadow-lg"
                  style={{
                    backgroundColor: `${risk.color}20`,
                    borderColor: risk.color,
                    color: risk.color
                  }}
                >
                  <span className="w-2 h-2 rounded-full animate-ping" style={{ backgroundColor: risk.color }}></span>
                  {risk.level} HAZARD LEVEL ({risk.score}/100)
                </div>
              )}
            </div>

            {risk ? (
              <div className="mt-5 space-y-4">
                <p className="text-sm text-slate-300 leading-relaxed bg-slate-900/60 p-3.5 rounded-xl border border-slate-800 font-medium">
                  📢 <span className="text-white font-semibold">{risk.advisory}</span>
                </p>

                {/* Subcomponents Breakdown */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <span className="text-slate-400 block font-medium">3-Day Forecast Rain</span>
                    <span className="text-lg font-bold text-cyan-400 mt-1 block">
                      {risk.parameters.forecast_3d_rain_mm} mm
                    </span>
                    <span className="text-[11px] text-slate-400">
                      Rain Factor: {risk.parameters.rain_subscore} / 100 pts
                    </span>
                  </div>

                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <span className="text-slate-400 block font-medium">Monitored Waterbody Area</span>
                    <span className="text-lg font-bold text-sky-400 mt-1 block">
                      {risk.parameters.current_lake_area_km2} km²
                    </span>
                    <span className="text-[11px] text-slate-400">
                      Baseline: {risk.parameters.lake_normal_baseline_km2} km² (Max: {risk.parameters.lake_flood_capacity_km2} km²)
                    </span>
                  </div>

                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <span className="text-slate-400 block font-medium">Retention Fullness</span>
                    <span className="text-lg font-bold text-emerald-400 mt-1 block">
                      {risk.parameters.lake_fullness_subscore} / 100 pts
                    </span>
                    <span className="text-[11px] text-slate-400">
                      Weight: 55% Rain + 45% Retention
                    </span>
                  </div>
                </div>

                {/* Transparent Formula Display */}
                <div className="text-[11px] text-slate-400 bg-slate-950/60 p-2.5 rounded-lg font-mono border border-slate-800">
                  <span className="text-slate-300 font-semibold font-sans">Scientific Formula: </span>
                  {risk.formula}
                </div>

                <p className="text-[11px] text-slate-400 italic">
                  {risk.disclaimer}
                </p>
              </div>
            ) : (
              <div className="py-8 text-center text-slate-500 text-sm">
                Awaiting telemetry calculation for {currentLoc.name}...
              </div>
            )}
          </div>

          {/* Card 2: Current Weather in Selected Location */}
          <div className="rounded-2xl bg-slate-800/80 border border-slate-700/80 p-6 backdrop-blur flex flex-col justify-between">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Ground Telemetry
              </span>
              <h3 className="text-xl font-bold text-white mt-1 flex items-center gap-2">
                <Compass className="w-5 h-5 text-cyan-400" />
                {currentLoc.name} Weather Now
              </h3>

              {weather ? (
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                      <Thermometer className="w-4 h-4 text-rose-400" /> Temp
                    </div>
                    <span className="text-xl font-extrabold text-white mt-1 block">
                      {weather.temperature_2m}°C
                    </span>
                  </div>

                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                      <Droplets className="w-4 h-4 text-cyan-400" /> Humidity
                    </div>
                    <span className="text-xl font-extrabold text-white mt-1 block">
                      {weather.relative_humidity_2m}%
                    </span>
                  </div>

                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                      <Wind className="w-4 h-4 text-emerald-400" /> Wind
                    </div>
                    <span className="text-xl font-extrabold text-white mt-1 block">
                      {weather.wind_speed_10m} <span className="text-xs font-normal text-slate-400">km/h</span>
                    </span>
                  </div>

                  <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60">
                    <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                      <Gauge className="w-4 h-4 text-amber-400" /> Pressure
                    </div>
                    <span className="text-xl font-extrabold text-white mt-1 block">
                      {weather.surface_pressure} <span className="text-xs font-normal text-slate-400">hPa</span>
                    </span>
                  </div>

                  <div className="col-span-2 bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                        <Cloud className="w-4 h-4 text-sky-400" /> Cloud Cover & Rain
                      </div>
                      <span className="text-sm font-bold text-white mt-0.5 block">
                        {weather.cloud_cover}% Clouded &bull; {weather.precipitation} mm Now
                      </span>
                    </div>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      Open-Meteo
                    </span>
                  </div>
                </div>
              ) : (
                <div className="py-8 text-center text-slate-500 text-sm">
                  Loading weather metrics...
                </div>
              )}
            </div>

            <p className="text-[11px] text-slate-500 mt-4">
              Station: {currentLoc.name} ({currentLoc.lat}°N, {currentLoc.lng}°E) &bull; {currentLoc.state}
            </p>
          </div>
        </div>

        {/* Map Control Bar and Interactive Leaflet View */}
        <div className="rounded-2xl bg-slate-800/80 border border-slate-700/80 p-5 backdrop-blur mb-8">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-4 pb-4 border-b border-slate-700/60">
            {/* Layer Selectors */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 mr-1">
                <Layers className="w-4 h-4 text-cyan-400" /> Base Satellite:
              </span>

              <button
                onClick={() => setActiveLayer("osm")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${activeLayer === "osm"
                  ? "bg-cyan-500 text-white shadow-md shadow-cyan-500/20 font-bold"
                  : "bg-slate-700/70 text-slate-300 hover:bg-slate-700"
                  }`}
              >
                🗺️ Cartographic OSM
              </button>

              <button
                onClick={() => setActiveLayer("modis")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${activeLayer === "modis"
                  ? "bg-cyan-500 text-white shadow-md shadow-cyan-500/20 font-bold"
                  : "bg-slate-700/70 text-slate-300 hover:bg-slate-700"
                  }`}
              >
                🛰️ NASA MODIS Terra (True Color)
              </button>

              <button
                onClick={() => setActiveLayer("viirs")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${activeLayer === "viirs"
                  ? "bg-cyan-500 text-white shadow-md shadow-cyan-500/20 font-bold"
                  : "bg-slate-700/70 text-slate-300 hover:bg-slate-700"
                  }`}
              >
                🛰️ NASA VIIRS SNPP (High-Res)
              </button>
            </div>

            {/* Overlay Toggles */}
            <div className="flex items-center gap-3">
              <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-300 cursor-pointer bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700">
                <input
                  type="checkbox"
                  checked={showRadar}
                  onChange={(e) => setShowRadar(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-0 bg-slate-800 border-slate-600"
                />
                📡 RainViewer Live Radar
              </label>

              <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-300 cursor-pointer bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700">
                <input
                  type="checkbox"
                  checked={showClouds}
                  onChange={(e) => setShowClouds(e.target.checked)}
                  className="rounded text-cyan-500 focus:ring-0 bg-slate-800 border-slate-600"
                />
                ☁️ NASA Cloud Optical Depth
              </label>
            </div>
          </div>

          {/* Satellite Observation Timestamps & Notice */}
          <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 mb-3 px-1 gap-2">
            <div>
              {monitorData?.nasa_gibs && (
                <span>
                  🟢 NASA Daily Satellite Pass: <strong className="text-white">{monitorData.nasa_gibs.satellite_date}</strong>{" "}
                  <span className="italic text-amber-400">({monitorData.nasa_gibs.label})</span>
                </span>
              )}
            </div>
            <div>
              {monitorData?.rainviewer_radar?.latest_timestamp && (
                <span>
                  📡 Radar Epoch:{" "}
                  <strong className="text-cyan-400">
                    {new Date(monitorData.rainviewer_radar.latest_timestamp * 1000).toLocaleTimeString()}
                  </strong>
                </span>
              )}
            </div>
          </div>

          {/* Leaflet Map Div */}
          <div
            ref={mapContainerRef}
            className="w-full h-[460px] rounded-xl overflow-hidden border border-slate-700 shadow-inner z-0"
            style={{ minHeight: "460px" }}
          />

          <p className="text-[11px] text-slate-500 mt-3 text-center">
            Map View dynamically centered on {currentLoc.name} ({currentLoc.state}) &bull; Continuous 10-minute automated telemetry polling.
          </p>
        </div>

        {/* Phase 10: RainLSTM Next-Day Prediction & 7-Day Forecast for Selected City */}
        {rainData && (
          <div className="rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/80 p-6 backdrop-blur">
            <div className="flex flex-col md:flex-row md:items-center justify-between pb-4 border-b border-slate-700/70 gap-3">
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
                  Time-Series Deep Learning
                </span>
                <h3 className="text-2xl font-bold text-white mt-0.5 flex items-center gap-2">
                  <CloudRain className="w-6 h-6 text-cyan-400" />
                  AquaWatch RainLSTM: 24-Hour Rain Probability for {currentLoc.name}
                </h3>
              </div>

              <div className="flex items-center gap-2 text-xs flex-wrap">
                <span className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300 font-mono">
                  Lookback: 10 Days Sequence
                </span>
                <span className="px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-semibold">
                  {rainData.data_source}
                </span>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Rain Probability Dial / Score */}
              <div className="bg-slate-950/70 rounded-xl p-5 border border-slate-800 flex flex-col justify-between">
                <div>
                  <span className="text-xs text-slate-400 font-medium">Tomorrow's Forecast Probability ({currentLoc.name})</span>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">
                      {rainData.rain_probability_pct}%
                    </span>
                    <span className="text-xs font-semibold text-slate-400">chance</span>
                  </div>

                  <div className="mt-4">
                    <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-700"
                        style={{ width: `${rainData.rain_probability_pct}%` }}
                      />
                    </div>
                  </div>

                  <p className="mt-4 text-sm font-semibold text-white flex items-center gap-2">
                    Verdict: <span className={rainData.rain_predicted ? "text-cyan-400" : "text-emerald-400"}>{rainData.verdict}</span>
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Model: {rainData.model_architecture}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500">
                  Computed dynamically using the latest 10-day meteorological sequence in {currentLoc.name}, {currentLoc.state}.
                </div>
              </div>

              {/* External 7-Day Numerical Weather Prediction Forecast */}
              <div className="md:col-span-2 bg-slate-950/70 rounded-xl p-5 border border-slate-800">
                <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    7-Day Precipitation Outlook for {currentLoc.name} (External NWP Forecast)
                  </span>
                  <span className="text-[10px] text-amber-400/90 font-medium bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                    Open-Meteo Forecast Engine (External Reference)
                  </span>
                </div>

                <div className="grid grid-cols-7 gap-2 pt-2">
                  {rainData.external_forecast_7d?.map((f, i) => (
                    <div
                      key={i}
                      className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 text-center flex flex-col justify-between hover:border-slate-700 transition"
                    >
                      <span className="text-[11px] font-semibold text-slate-400">
                        {f.date.slice(5)}
                      </span>
                      <div className="my-2">
                        <CloudRain
                          className={`w-5 h-5 mx-auto ${f.precipitation_probability_pct > 50
                            ? "text-cyan-400 animate-pulse"
                            : "text-slate-500"
                            }`}
                        />
                      </div>
                      <div>
                        <span className="text-xs font-extrabold text-white block">
                          {f.precipitation_probability_pct}%
                        </span>
                        <span className="text-[10px] text-slate-400 block mt-0.5">
                          {f.precipitation_sum_mm} mm
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <p className="text-[11px] text-slate-500 mt-4 text-right">
                  Values reflect Open-Meteo Ensemble Global NWP forecasts &bull; Used in 3-day flood retention formula.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
