import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Chembarambakkam Lake coordinates: [79.98, 12.98, 80.08, 13.06] (lon_min, lat_min, lon_max, lat_max)
bbox = [79.98, 12.98, 80.08, 13.06]

# Earth Search AWS STAC API (Sentinel-2 L2A open data on AWS, no auth)
url = "https://earth-search.aws.element84.com/v1/search"
payload = {
    "collections": ["sentinel-2-l2a"],
    "bbox": bbox,
    "datetime": "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z",
    "query": {"eo:cloud_cover": {"lt": 25}},
    "limit": 50,
    "sortby": [{"field": "properties.datetime", "direction": "asc"}]
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "User-Agent": "AquaWatch/1.0"}
)

try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        features = data.get("features", [])
        print(f"SUCCESS: Found {len(features)} real Sentinel-2 scenes over Chembarambakkam Lake in 2023!")
        for f in features[:10]:
            fid = f["id"]
            dt = f["properties"]["datetime"][:10]
            cc = f["properties"]["eo:cloud_cover"]
            print(f"  Date: {dt} | Cloud: {cc:.1f}% | ID: {fid}")
            # check asset keys
            assets = list(f.get("assets", {}).keys())
            if "red" in assets or "B04" in assets:
                print(f"    Available band assets: {assets[:8]}")
except Exception as e:
    print("STAC search error:", e)
