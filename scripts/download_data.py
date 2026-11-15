"""Download the ZAPBench activity traces and neuron positions (public, CC-BY 4.0; ~2 GB).

    uv run python scripts/download_data.py

Source: gs://zapbench-release/volumes/20240930 (Lueckmann et al., ICLR 2025). Saves
  data/traces.zip      zarr v3 in a zip: 7,879 frames x 71,721 neurons, float32
  data/centroids.json  neuron centroids in um (used only for the component maps)
"""
import base64, hashlib
import requests
from common import DATA

BASE = "https://storage.googleapis.com/zapbench-release/volumes/20240930"
FILES = {"traces.zip": f"{BASE}/traces.zip", "centroids.json": f"{BASE}/segmentation/dataframe_centroids.json"}

DATA.mkdir(exist_ok=True)
for name, url in FILES.items():
    dest = DATA / name
    if dest.exists():
        print("exists:", dest); continue
    print("downloading", url)
    md5 = hashlib.md5()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        expected = r.headers.get("x-goog-hash", "")
        with open(dest.with_suffix(".part"), "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 22):
                f.write(chunk); md5.update(chunk)
    if "md5=" in expected and f"md5={base64.b64encode(md5.digest()).decode()}" not in expected:
        raise RuntimeError(f"checksum mismatch for {name}")
    dest.with_suffix(".part").rename(dest)
    print("saved", dest)
