import requests

# 1. Metrics
r_met = requests.get('http://127.0.0.1:8000/api/weather/metrics')
print('Weather Metrics HTTP:', r_met.status_code)
print('Headline:', r_met.json().get('headline_stats'))

# 2. Samples
r_samp = requests.get('http://127.0.0.1:8000/api/weather/samples')
print('Weather Samples HTTP:', r_samp.status_code)
print('Classes:', [s['class'] for s in r_samp.json()])

# 3. Classify with a test rain sample
test_img = r'aquawatch/results/weather_samples/rain_sample.jpg'
with open(test_img, 'rb') as f:
    r_cls = requests.post('http://127.0.0.1:8000/api/weather/classify', files={'file': ('rain_sample.jpg', f, 'image/jpeg')})
print('Classify Status:', r_cls.status_code)
d = r_cls.json()
print('Top Prediction:', d.get('top_prediction'), f"{d.get('confidence')}%")
print('Top 3 Predictions:', d.get('predictions'))
print('Grad-CAM Heatmap Base64 Present:', bool(d.get('gradcam_base64')), f"({len(d.get('gradcam_base64', ''))} chars)")
