#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "torch",
#     "torchvision",
#     "pillow",
#     "numpy",
#     "requests",
#     "tutti-scraper>=0.3.3,<0.4",
# ]
# ///
"""
Search tutti.ch and find the listing that looks most like your photo, live.

Three things run at the same time:
  1. the scraper streams listings from tutti.ch page by page,
  2. a pool of threads downloads each listing's thumbnail into the output folder,
  3. DINOv2 embeds every thumbnail as it lands and compares it to your photo.

The best match so far is printed every time it changes, so you can stop with
Ctrl+C as soon as you see what you want. All scores and prices go to a CSV, and
the embeddings go into the same cache dino_compare.py uses, so running
dino_compare.py on the folder later with another photo needs no recomputation.

Needs dino_compare.py in the same folder.

Usage:
    uv run live_match.py photo.jpg "gamecube"
    uv run live_match.py photo.jpg "gamecube" -c consolesGames --stop-at 0.85
    uv run live_match.py photo.jpg "chaise" -c household --max-results 300 --top 15
"""

import argparse
import csv
import math
import queue
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from tutti_scraper import CATEGORY_GROUPS, Scraper, TuttiClient, flatten_listing, listing_url

import time

from dino_compare import (
    IMAGE_EXTS,
    MODELS,
    build_transform,
    embed,
    load_cache,
    load_image,
    pick_device,
    save_cache,
)

DONE = object()  # end-of-stream marker on the item queue


# ------------------------------------------------------------------ output
# A single status line lives at the bottom of the terminal (stderr) and every
# other message is printed above it, so threads can log without garbling it.

_print_lock = threading.Lock()
_status_text = ""
_tty = sys.stderr.isatty()


def _emit(stream, text):
    with _print_lock:
        if _tty and _status_text:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        stream.write(text + "\n")
        stream.flush()
        if _tty and _status_text:
            sys.stderr.write(_status_text)
            sys.stderr.flush()


def log(msg):
    _emit(sys.stderr, msg)


def say(msg):
    _emit(sys.stdout, msg)


def status(text):
    global _status_text
    if not _tty:
        return
    with _print_lock:
        _status_text = text
        sys.stderr.write("\r\033[K" + text)
        sys.stderr.flush()


def clear_status():
    global _status_text
    with _print_lock:
        if _tty and _status_text:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        _status_text = ""


# ------------------------------------------------------------------ helpers

def parse_args():
    ap = argparse.ArgumentParser(description="Live DINOv2 matching of a photo against tutti.ch listings.")
    ap.add_argument("query_image", help="photo of the object you want to price")
    ap.add_argument("search", help='text to search on tutti.ch, e.g. "gamecube"')
    ap.add_argument("-c", "--category", action="append",
                    help="tutti category or group, repeatable (e.g. consolesGames, household)")
    ap.add_argument("--lang", default="de", choices=["de", "fr", "it"], help="listing URL language (default: de)")
    ap.add_argument("--max-results", type=int, help="stop scraping after this many listings")
    ap.add_argument("--stop-at", type=float, help="stop everything once a match scores at least this")
    ap.add_argument("--out", help="folder for the thumbnails (default: img/<search>)")
    ap.add_argument("--csv", help="where to write every score (default: <out>/matches.csv)")
    ap.add_argument("--top", type=int, default=10, help="results to show at the end (default: 10)")
    ap.add_argument("--workers", type=int, default=8, help="parallel thumbnail downloads (default: 8)")
    ap.add_argument("--delay", type=float, default=1, help="seconds between tutti.ch API calls (default: 0.25)")
    ap.add_argument("--timeout", type=float, default=10.0, help="thumbnail download timeout in s (default: 10)")
    ap.add_argument("--model", default="vits14", choices=MODELS, help="DINOv2 backbone (default: vits14)")
    ap.add_argument("--size", type=int, default=224, help="input size in px, multiple of 14 (default: 224)")
    ap.add_argument("--no-crop", action="store_true", help="squash images to a square instead of center crop")
    ap.add_argument("--batch-size", type=int, default=32, help="max images per forward pass (default: 32)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps (default: auto)")
    ap.add_argument("--headstart", type=int, default=300, help="max listings the scraper may be ahead of the comparison (default: 300)")
    return ap.parse_args()


def expand_categories(names):
    """None means 'search everything'. Group names expand to their sub-categories."""
    if not names:
        return [None]
    out = []
    for name in names:
        out.extend(CATEGORY_GROUPS.get(name, (name,)))
    return list(dict.fromkeys(out))


def slugify(text):
    return re.sub(r"[^\w-]+", "_", text.strip().lower()).strip("_") or "search"


def numeric_price(row):
    try:
        p = float(row.get("price"))
    except (TypeError, ValueError):
        return None
    return p if math.isfinite(p) else None


def chf(value):
    return f"CHF {value:,.0f}".replace(",", "'")


def price_label(row):
    p = numeric_price(row)
    if p is not None:
        return chf(p)
    return str(row.get("formattedPrice") or "?").strip() or "?"


def short(text, n):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


#overwriting scraper to not do the ascending sweep 
class NoSweepScraper(Scraper):
    """Scraper without the final ascending sweep."""

    def run(self):
        allow_split = self.seed_category is None
        yield from self._scrape(
            category=self.seed_category,
            constraints=self.seed_constraints,
            allow_category_split=allow_split,
        )


# ------------------------------------------------------------------ pipeline

class Pipeline:
    """Scraper thread -> download pool -> bounded queue -> main thread (DINOv2)."""

    def __init__(self, args, out_dir, cache, transform, categories):
        self.args = args
        self.out_dir = out_dir
        self.cache = cache  # read-only here, the main thread builds the updated copy
        self.transform = transform
        self.categories = categories
        # Bounded, so downloads pause if embedding falls behind (caps memory).
        self.items = queue.Queue(maxsize=max(4 * args.batch_size, 64))
        self.stop = threading.Event()
        self.scrape_done = threading.Event()
        self.found = 0
        self.downloaded = 0
        self.failed = 0
        self._count_lock = threading.Lock()
        self._local = threading.local()
        self.processed = 0

    def _count(self, name):
        with self._count_lock:
            setattr(self, name, getattr(self, name) + 1)

    def _session(self):
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
        return s

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.items.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def _limit_reached(self):
        return self.args.max_results is not None and self.found >= self.args.max_results

    def _wait_for_room(self):
        """Block the scraper while it is more than --headstart listings ahead."""
        while not self.stop.is_set():
            backlog = self.found - self.failed - self.processed
            if backlog < self.args.headstart:
                return
            time.sleep(0.1)

    def fetch(self, row):
        """Download one thumbnail (unless already on disk) and preprocess it."""
        if self.stop.is_set():
            return
        url = row["thumbnailURL"]
        ext = Path(urlparse(url).path).suffix.lower()
        if ext not in IMAGE_EXTS:
            ext = ".jpg"
        name = re.sub(r"[^\w-]", "_", str(row["listingID"]))
        path = self.out_dir / f"{name}{ext}"
        try:
            if not path.exists():
                r = self._session().get(url, timeout=self.args.timeout)
                r.raise_for_status()
                part = path.with_name(path.name + ".part")
                part.write_bytes(r.content)
                part.replace(path)
            self._count("downloaded")
            mtime = path.stat().st_mtime_ns
            hit = self.cache.get(path.name)
            if hit is not None and hit[0] == mtime:
                self._put((path.name, mtime, row, None, hit[1]))
            else:
                self._put((path.name, mtime, row, self.transform(load_image(path)), None))
        except Exception as exc:
            self._count("failed")
            log(f"Skipping listing {row.get('listingID')}: {exc}")

    def scrape(self):
        """Runs in its own thread: stream listings and hand thumbnails to the pool."""
        client = TuttiClient(lang=self.args.lang, delay=self.args.delay)
        seen = set()
        pool = ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix="download")
        try:
            for category in self.categories:
                if self.stop.is_set() or self._limit_reached():
                    break
                try:
                    scraper = NoSweepScraper(client, self.args.search, sort="RELEVANCE", seed_category=category)
                    for node in scraper.run():
                        if self.stop.is_set() or self._limit_reached():
                            break
                        lid = node.get("listingID")
                        if lid is None or lid in seen:
                            continue
                        seen.add(lid)
                        node["url"] = listing_url(node, self.args.lang)
                        row = flatten_listing(node, self.args.lang)
                        if not row.get("thumbnailURL"):
                            continue
                        self._count("found")
                        self._wait_for_room()
                        pool.submit(self.fetch, row)
                except Exception as exc:
                    log(f"Scraping failed for category {category or 'all'}: {exc}")
        finally:
            pool.shutdown(wait=True, cancel_futures=self.stop.is_set())
            self.scrape_done.set()
            self._put(DONE)


def batches(items, size):
    """Yield whatever is ready, up to `size` items. An empty list means 'nothing yet'."""
    while True:
        try:
            item = items.get(timeout=0.3)
        except queue.Empty:
            yield []
            continue
        if item is DONE:
            return
        batch = [item]
        while len(batch) < size:
            try:
                item = items.get_nowait()
            except queue.Empty:
                break
            if item is DONE:
                yield batch
                return
            batch.append(item)
        yield batch


# ------------------------------------------------------------------ main

def main():
    args = parse_args()

    query_path = Path(args.query_image).expanduser().resolve()
    if not query_path.is_file():
        sys.exit(f"Query image not found: {query_path}")
    if args.size % 14 != 0:
        sys.exit(f"--size must be a multiple of 14 (DINOv2 uses 14px patches), got {args.size}")

    out_dir = Path(args.out or Path("img") / slugify(args.search)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).expanduser() if args.csv else out_dir / "matches.csv"

    crop = not args.no_crop
    transform = build_transform(args.size, crop)
    cache_path = out_dir / f".dinov2_{args.model}_{args.size}_{'crop' if crop else 'full'}.npz"
    cache = load_cache(cache_path)

    pipe = Pipeline(args, out_dir, cache, transform, expand_categories(args.category))

    # Start scraping first, so listings and thumbnails pile up while the model loads.
    threading.Thread(target=pipe.scrape, name="scraper", daemon=True).start()

    device = pick_device(args.device)
    log(f'Searching tutti.ch for "{args.search}" into {out_dir}')
    log(f"Loading dinov2_{args.model} on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{args.model}",trust_repo=True, skip_validation=True)
    model = model.to(device).eval()
    q = embed(model, [transform(load_image(query_path))], device)[0]
    log("Model ready, matching as thumbnails arrive. Ctrl+C stops and shows the results so far.\n")

    new_cache = dict(cache)
    results = []  # (score, image name, row)
    best = None

    def refresh():
        state = "scraping" if not pipe.scrape_done.is_set() else "finishing"
        best_txt = f"best {best[0]:.3f}" if best else "no match yet"
        status(f"  {state}: {pipe.found} listings, {pipe.downloaded} downloaded, "
               f"{len(results)} compared, {best_txt}")

    try:
        for batch in batches(pipe.items, args.batch_size):
            if batch:
                fresh = [it for it in batch if it[4] is None]
                vecs = embed(model, [it[3] for it in fresh], device) if fresh else []
                computed = {it[0]: v for it, v in zip(fresh, vecs)}
                for name, mtime, row, _, cached in batch:
                    vec = cached if cached is not None else computed[name]
                    new_cache[name] = (mtime, vec)
                    score = float(vec @ q)
                    results.append((score, name, row))
                    if best is None or score > best[0]:
                        best = (score, name, row)
                        say(f"new best  {score:.3f}  {price_label(row):>12}  "
                            f"{short(row.get('title'), 50)}  {row.get('url', '')}")
            refresh()
            if args.stop_at is not None and best and best[0] >= args.stop_at:
                log(f"Reached --stop-at {args.stop_at}, stopping.")
                break
    except KeyboardInterrupt:
        log("\nStopped, here is what was found so far.")
    finally:
        pipe.stop.set()
        clear_status()

    if new_cache:
        save_cache(cache_path, new_cache)

    if not results:
        sys.exit("No listing could be compared.")

    results.sort(key=lambda r: -r[0])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "price", "formattedPrice", "title", "url", "image", "listingID"])
        for rank, (score, name, row) in enumerate(results, 1):
            p = numeric_price(row)
            w.writerow([rank, f"{score:.4f}", "" if p is None else p, row.get("formattedPrice", ""),
                        row.get("title", ""), row.get("url", ""), name, row.get("listingID", "")])

    top = results[: max(args.top, 1)]
    print(f'\nTop {len(top)} of {len(results)} listings for "{args.search}" '
          f"(dinov2_{args.model} @ {args.size}px)\n")
    print(f"{'rank':>4}  {'score':>5}  {'price':>12}  {'title':<50}  link")
    for rank, (score, name, row) in enumerate(top, 1):
        print(f"{rank:>4}  {score:.3f}  {price_label(row):>12}  "
              f"{short(row.get('title'), 50):<50}  {row.get('url', '')}")

    prices = [p for p in (numeric_price(r) for _, _, r in top) if p is not None]
    if prices:
        print(f"\nMedian price of these {len(top)} closest listings "
              f"({len(prices)} with a price): {chf(statistics.median(prices))}")
    print(f"All {len(results)} scores saved to {csv_path}")
    if pipe.failed:
        print(f"{pipe.failed} thumbnail(s) could not be downloaded or read.")

    pipe.processed = len(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
