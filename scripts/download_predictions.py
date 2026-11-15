"""Download released ZAPBench test-set forecasts of the published models (public bucket, CC-BY 4.0).

    uv run python -u scripts/download_predictions.py --name unet --exp 126495300 --seeds 1,2,3
    uv run python -u scripts/download_predictions.py --name tsmixer --exp 146317978 --seeds 4

Long-context (256-frame) experiments: unet 126495300 (seeds 1-3), tsmixer 146317978, time-mix 146318174,
tide 146462053, linear 146855456 (seeds 4-6). Each seed is ~17 GB: per condition a zarr v2 array
[windows, 32 steps, 71,721 neurons] float32, uncompressed. Saved to data/preds/<name>/seed<k>/<condition>/;
<condition>/.complete marks a finished condition, so the download can be resumed.
"""
import base64, hashlib, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from common import DATA, ZARR_NAME

arg = lambda k: sys.argv[sys.argv.index(k) + 1]
NAME, EXP, SEEDS = arg("--name"), arg("--exp"), [int(s) for s in arg("--seeds").split(",")]
OUT = DATA / "preds" / NAME; OUT.mkdir(parents=True, exist_ok=True)
API = "https://storage.googleapis.com/storage/v1/b/zapbench-release/o"
BASE = "https://storage.googleapis.com/zapbench-release/"


def list_objects(prefix):
    items, token = [], None
    while True:
        url = f"{API}?prefix={prefix}&maxResults=1000&fields=items(name,size,md5Hash),nextPageToken" + (f"&pageToken={token}" if token else "")
        d = json.loads(urllib.request.urlopen(url, timeout=60).read())
        items += d.get("items", []); token = d.get("nextPageToken")
        if not token:
            return items


def fetch(item, dest_dir, prefix):
    dest = dest_dir / item["name"][len(prefix):]
    if dest.exists() and dest.stat().st_size == int(item["size"]):
        return 0
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(BASE + item["name"], timeout=120).read()
            if base64.b64encode(hashlib.md5(data).digest()).decode() != item["md5Hash"]:
                raise IOError("md5 mismatch")
            tmp = dest.with_name(dest.name + ".part"); tmp.write_bytes(data); tmp.rename(dest)
            return len(data)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


for seed in SEEDS:
    for cond in ZARR_NAME.values():
        dest_dir = OUT / f"seed{seed}" / cond
        if (dest_dir / ".complete").exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"inference/20250131/{EXP}/{seed}/predictions/{cond}/"
        items = list_objects(prefix); t = time.time()
        with ThreadPoolExecutor(24) as ex:
            nbytes = sum(ex.map(lambda it: fetch(it, dest_dir, prefix), items))
        (dest_dir / ".complete").write_text(json.dumps({"files": len(items)}))
        print(f"{NAME} seed {seed} {cond:26s} {len(items):5d} files, {nbytes / 1e9:5.2f} GB in {time.time() - t:4.0f}s", flush=True)
