"""
Build the app's visual category check: DINOv2 descriptors of recent tutti.ch listings in every category.

The phone compares the object it looks at with these listings (CategoryRefs.kt). A category whose listings
look like the object gets a high probability, which is mixed with JEV's. It catches the objects the camera
names wrongly (YOLOE calls a GameCube an "arcade machine", so JEV picks the wrong category from the name).

    python tools/build_category_refs.py                       # 60 listings per category
    python tools/build_category_refs.py --check photo.jpg     # also print the categories for some photos

Writes android/app/src/main/assets/category_refs.bin:
    "CREF", version 1, n, dim, categories, then the category keys (uint16 length + UTF-8),
    then n uint16 category indices, then n x dim float16 descriptors.

Descriptors are computed like the app's embedder (tools/export_embedder.py, DINOv2-small): the thumbnail
letterboxed on gray to 168 x 168, normalise(normalise(CLS) + normalise(mean of patch tokens)).
Needs torch, transformers, pillow, numpy, requests (the venv from tools/export_jev.py works).
"""
import argparse
import io
import json
import math
import struct
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_prompt  # noqa: E402

API = "https://www.tutti.ch/api/v10/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    "Accept-Language": "de-CH",
    "X-Tutti-Source": "web LIVE",
    "X-Tutti-Client-Identifier": "web/1.0.0+env-live.git-0000000",
}
QUERY = """
query Search($q: String, $category: ID, $constraints: ListingSearchConstraints,
  $first: Int!, $offset: Int!, $sort: ListingSortMode!, $direction: SortDirection!) {
  searchListingsByQuery(query: $q, category: $category, constraints: $constraints) {
    listings(first: $first, offset: $offset, sort: $sort, direction: $direction) {
      totalCount
      edges { node { listingID title thumbnail { normalRendition: rendition(width: 235, height: 167) { src } } } }
    }
  }
}
"""
SIZE = 168
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TOP, TEMPERATURE = 3, 0.03  # as in CategoryRefs.kt


def newest(session, category, n):
    out, offset = [], 0
    while len(out) < n:
        body = {"query": QUERY, "variables": {"q": None, "category": category, "constraints": None,
                                              "first": 100, "offset": offset, "sort": "TIMESTAMP", "direction": "DESCENDING"}}
        r = session.post(API, json=body, headers={**HEADERS, "X-Tutti-Hash": str(uuid.uuid4())}, timeout=30)
        r.raise_for_status()
        edges = r.json()["data"]["searchListingsByQuery"]["listings"]["edges"]
        if not edges:
            break
        for e in edges:
            src = ((e["node"].get("thumbnail") or {}).get("normalRendition") or {}).get("src")
            if src:
                out.append((e["node"]["listingID"], e["node"]["title"], src))
        offset += len(edges)
        time.sleep(0.3)
    return out[:n]


def letterbox(img):
    img = img.convert("RGB")
    s = SIZE / max(img.size)
    w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(img.resize((w, h), Image.BILINEAR), ((SIZE - w) // 2, (SIZE - h) // 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def embed(dino, images, batch=32):
    out = []
    for i in range(0, len(images), batch):
        x = (torch.stack(images[i:i + batch]) - MEAN) / STD
        h = dino(pixel_values=x).last_hidden_state
        e = F.normalize(h[:, 0], dim=-1) + F.normalize(h[:, 1:].mean(1), dim=-1)
        out.append(F.normalize(e, dim=-1))
    return torch.cat(out)


def probabilities(refs, cats, keys, e):
    sims = refs @ e
    score = {}
    for k, key in enumerate(keys):
        s = sims[cats == k]
        if len(s):
            score[key] = float(torch.topk(s, min(TOP, len(s))).values.mean())
    m = max(score.values())
    ex = {k: math.exp((v - m) / TEMPERATURE) for k, v in score.items()}
    z = sum(ex.values())
    return sorted(((k, v / z) for k, v in ex.items()), key=lambda t: -t[1])


def main():
    ap = argparse.ArgumentParser(description="Build the app's visual category references from tutti.ch")
    ap.add_argument("--per-category", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "android/app/src/main/assets/category_refs.bin"))
    ap.add_argument("--weights", default="facebook/dinov2-small")
    ap.add_argument("--check", nargs="*", default=[], help="photos (object crops) to classify with the result")
    args = ap.parse_args()

    from transformers import Dinov2Model

    cats = jev_prompt.load_categories()
    keys = [c["key"] for g in cats["groups"] for c in g["categories"]]
    session = requests.Session()
    rows = []
    for k, key in enumerate(keys):
        try:
            found = newest(session, key, args.per_category)
        except Exception as exc:  # a category that fails just has no references
            print(f"{key}: {exc}")
            continue
        rows += [(k, src) for _, _, src in found]
        print(f"{key}: {len(found)} listings")

    def fetch(src):
        try:
            return Image.open(io.BytesIO(session.get(src, timeout=20).content))
        except Exception:
            return None

    with ThreadPoolExecutor(8) as pool:
        images = list(pool.map(fetch, [src for _, src in rows]))
    kept = [(k, letterbox(img)) for (k, _), img in zip(rows, images) if img is not None]
    print(f"{len(kept)} thumbnails")

    dino = Dinov2Model.from_pretrained(args.weights).eval()
    refs = embed(dino, [t for _, t in kept])
    cat_idx = torch.tensor([k for k, _ in kept])

    out = Path(args.out)
    with open(out, "wb") as f:
        f.write(b"CREF")
        f.write(struct.pack("<4i", 1, len(kept), refs.shape[1], len(keys)))
        for key in keys:
            b = key.encode("utf-8")
            f.write(struct.pack("<H", len(b)) + b)
        f.write(cat_idx.numpy().astype("<u2").tobytes())
        f.write(refs.numpy().astype("<f2").tobytes())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    for path in args.check:
        e = embed(dino, [letterbox(Image.open(path))])[0]
        top = probabilities(refs, cat_idx, keys, e)[:4]
        print(f"{Path(path).name}: " + ", ".join(f"{k} {p:.2f}" for k, p in top))


if __name__ == "__main__":
    main()
