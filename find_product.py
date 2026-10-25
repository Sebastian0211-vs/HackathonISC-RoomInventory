import argparse
import csv
import math
import queue
import re
import statistics
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from tutti_scraper import MAX_OFFSET, PAGE_SIZE, TuttiClient, flatten_listing, listing_url

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
from live_match import (
    clear_status,
    chf,
    expand_categories,
    log,
    numeric_price,
    price_label,
    say,
    short,
    slugify,
    status,
)

# Words that never start, end or sit inside a product name (de / fr / it / en,
# accents stripped, plus marketplace filler).
STOPWORDS = set("""
a e i o l d x
der die das den dem des ein eine einen einem einer und oder mit ohne fur von vom zu zum zur
im in am an auf aus bei bis neu neue neuer neues neuwertig gut gute guter gutes sehr top
zustand verkaufe verkaufen verkauf inkl inklusive original originale originalverpackt ovp
set paket stk stuck gebraucht funktioniert funktionsfahig komplett voll
le la les un une des du de et ou avec sans pour en au aux sur tres bon bonne etat neuf neuve
occasion vendre vend vends lot complet complete fonctionne comme parfait
il lo gli una con senza per di da del della nuovo nuova usato usata ottimo ottima stato vendo completo
the an and or with without for of on new used good great condition mint boxed sale sell selling
chf fr franken preis prix prezzo vb nur abholung livraison versand
ungetestet getestet bastler defekt kaputt startet nicht ersatzteile teile funktionstuchtig
""".split())
QUANTITY = re.compile(r"^(\d+x|x\d+)$")  # "2x", "x3"

# Single words too broad to be worth a search of their own. Colours are fine
# inside a longer name ("gamecube lila") but not as a search by themselves.
GENERIC = set("""
konsole konsolen spielkonsole spielkonsolen console consoles videogame videogames videospiel videospiele
game games spiel spiele jeu jeux gioco giochi zubehor accessoires accessori accessories
controller controllers manette manettes kabel cable cables retro vintage
schwarz weiss rot blau grun gelb lila violett pink rosa grau silber gold orange braun beige turkis
noir blanc rouge bleu vert jaune violet gris argent rose nero bianco rosso blu verde giallo viola grigio
argento black white red blue green yellow purple grey gray silver
slim pro lite mini max plus edition limited special bundle collection sammlung
""".split())

# Explore: listings within EXPLORE_MARGIN of the best score are candidates, a title's
# product name starts within its first HEAD_WINDOW words, and a single word found in
# more than BROAD_SHARE of all titles is a brand, not a product.
EXPLORE_MARGIN = 0.15
HEAD_WINDOW = 4
BROAD_SHARE = 0.10

# Refine: a listing's weight is exp((score - best score) / FOCUS), so the very
# closest listings decide which extra word narrows a name.
FOCUS = 0.05

# Ranking: mean score of a search's TOP_N closest listings (missing ones count as
# the overall median). Searches within TIE of the best are a tie, which goes to
# the more specific name, then to the higher strong-match rate.
TOP_N = 5
TIE = 0.02

# A "strong match" scores above what only the top BASE_RATE of the seed listings
# reach. The rate is pulled toward BASE_RATE with HIT_PRIOR pseudo-results.
BASE_RATE = 0.02
HIT_PRIOR = 10


# ------------------------------------------------------------------ text mining

def tokens(text):
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def phrases(title, max_n=3):
    """Every 1 to 3 word span of a title that could plausibly be a product name."""
    toks = tokens(title)
    out = set()
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            span = toks[i:i + n]
            if span[0].isdigit() or any(t in STOPWORDS or QUANTITY.match(t) for t in span):
                continue
            if n == 1 and (len(span[0]) < 3 or span[0] in GENERIC):
                continue
            out.add(" ".join(span))
    return out


# ------------------------------------------------------------------ tutti paging

def iter_listings(client, text, category, sort):
    """Plain linear paging, up to tutti's offset cap. Enough for capped searches,
    and unlike Scraper it never bisects by price, so a capped sample isn't biased
    toward the cheapest listings."""
    offset = 0
    while offset <= MAX_OFFSET:
        result = client.search(text, category=category, offset=offset, first=PAGE_SIZE, sort=sort)
        listings = result["listings"]
        edges = listings["edges"]
        if not edges:
            return
        for edge in edges:
            yield edge["node"]
        offset += PAGE_SIZE
        if offset >= listings["totalCount"]:
            return


# ------------------------------------------------------------------ pipeline

@dataclass(eq=False)
class Search:
    text: str | None
    categories: list
    limit: int
    is_seed: bool = False
    why: str = ""
    members: set = field(default_factory=set)
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None

    @property
    def label(self):
        if self.text:
            return self.text
        return ", ".join(c for c in self.categories if c) or "everything"


class Explorer:
    """Search threads -> shared download pool -> bounded queue -> main thread (DINOv2).
    The main thread also mines names and launches new searches."""

    def __init__(self, args, out_dir, cache, transform, categories):
        self.args = args
        self.out_dir = out_dir
        self.cache = cache
        self.transform = transform
        self.categories = categories
        self.items = queue.Queue(maxsize=max(4 * args.batch_size, 64))
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.pending = 0  # thumbnails submitted but not yet queued or failed
        self.failed = 0
        self.seen = set()
        self.searches = []
        self.searched_keys = set()
        self.pool = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="download")
        self._local = threading.local()
        # Owned by the main thread.
        self.scores = {}
        self.rows = {}
        self.images = {}
        self.phrases = {}
        self.df = Counter()

    # -- searches -------------------------------------------------------

    def launch(self, text, why="", is_seed=False):
        cats = self.categories if (is_seed or not self.args.refine_everywhere) else [None]
        s = Search(text=text, categories=cats, is_seed=is_seed, why=why,
                   limit=self.args.seed_limit if is_seed else self.args.per_search)
        self.searches.append(s)
        if text:
            self.searched_keys.add(frozenset(tokens(text)))
        threading.Thread(target=self._run_search, args=(s,), name=f"search-{len(self.searches)}",
                         daemon=True).start()
        return s

    def refined(self):
        return [s for s in self.searches if not s.is_seed]

    def can_launch(self):
        refined = self.refined()
        running = sum(not s.done.is_set() for s in refined)
        return len(refined) < self.args.max_searches and running < self.args.parallel

    def idle(self):
        if any(not s.done.is_set() for s in self.searches):
            return False
        with self.lock:
            if self.pending:
                return False
        return self.items.empty()

    def _run_search(self, s):
        client = TuttiClient(lang=self.args.lang, delay=self.args.delay)
        sort = "RELEVANCE" if s.text else "TIMESTAMP"
        per_category = math.ceil(s.limit / len(s.categories))
        try:
            for category in s.categories:
                if self.stop.is_set():
                    break
                taken = 0
                try:
                    for node in iter_listings(client, s.text, category, sort):
                        if self.stop.is_set() or taken >= per_category:
                            break
                        lid = node.get("listingID")
                        if lid is None:
                            continue
                        with self.lock:
                            if lid in s.members:
                                continue
                            s.members.add(lid)
                            is_new = lid not in self.seen
                            self.seen.add(lid)
                        taken += 1
                        if not is_new:
                            continue  # already downloaded or queued by another search
                        node["url"] = listing_url(node, self.args.lang)
                        row = flatten_listing(node, self.args.lang)
                        if not row.get("thumbnailURL"):
                            continue
                        row["_found_by"] = s.label
                        with self.lock:
                            self.pending += 1
                        try:
                            self.pool.submit(self._fetch, row)
                        except RuntimeError:  # pool already shut down
                            with self.lock:
                                self.pending -= 1
                            return
                except Exception as exc:
                    s.error = str(exc)
                    log(f'Search "{s.label}" failed in {category or "all categories"}: {exc}')
        finally:
            s.done.set()

    # -- downloads ------------------------------------------------------

    def _session(self):
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = self._local.session = requests.Session()
        return sess

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.items.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def _fetch(self, row):
        try:
            if self.stop.is_set():
                return
            url = row["thumbnailURL"]
            ext = Path(urlparse(url).path).suffix.lower()
            if ext not in IMAGE_EXTS:
                ext = ".jpg"
            name = re.sub(r"[^\w-]", "_", str(row["listingID"]))
            path = self.out_dir / f"{name}{ext}"
            if not path.exists():
                r = self._session().get(url, timeout=self.args.timeout)
                r.raise_for_status()
                part = path.with_name(path.name + ".part")
                part.write_bytes(r.content)
                part.replace(path)
            mtime = path.stat().st_mtime_ns
            hit = self.cache.get(path.name)
            if hit is not None and hit[0] == mtime:
                self._put((path.name, mtime, row, None, hit[1]))
            else:
                self._put((path.name, mtime, row, self.transform(load_image(path)), None))
        except Exception as exc:
            with self.lock:
                self.failed += 1
            log(f"Skipping listing {row.get('listingID')}: {exc}")
        finally:
            with self.lock:
                self.pending -= 1

    # -- results and mining (main thread) -------------------------------

    def add_result(self, lid, score, row, image):
        self.scores[lid] = score
        self.rows[lid] = row
        self.images[lid] = image
        ph = phrases(row.get("title"))
        self.phrases[lid] = ph
        self.df.update(ph)

    def _closest(self):
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[: self.args.top_k]

    def head_name(self, lid):
        """The product name a title starts with: the longest phrase (up to 3 words)
        beginning in its first few words that also appears in another title, or,
        failing that, the longest early phrase of up to 2 words."""
        toks = tokens(self.rows[lid].get("title"))
        ph = self.phrases[lid]
        n_titles = len(self.scores)
        best, fallback = None, None
        for start in range(min(len(toks), HEAD_WINDOW)):
            for n in (3, 2, 1):
                p = " ".join(toks[start:start + n])
                if len(toks[start:start + n]) < n or p not in ph:
                    continue
                if n == 1 and self.df[p] > BROAD_SHARE * n_titles:
                    continue  # a brand like "nintendo", not a product
                if self.df[p] >= 2 and (best is None or (n, -start) > best[0]):
                    best = ((n, -start), p)
                if n <= 2 and (fallback is None or (n, -start) > fallback[0]):
                    fallback = ((n, -start), p)
        pick = best or fallback
        return pick[1] if pick else None

    def _family_searched(self, key):
        """True if this name or a close relative was already searched: a broader
        version of it ("xbox one" after "xbox one s"), or one sharing at least two
        words ("gamecube konsole lila" after "nintendo gamecube lila"). A searched
        brand alone ("nintendo") doesn't block "nintendo gamecube", and "xbox 360"
        stays separate from "xbox one"."""
        return any(key <= base or len(key & base) >= 2 for base in self.searched_keys)

    def explore_name(self):
        """Product name of the closest listing that no name search has covered yet,
        looking past covered ones down to EXPLORE_MARGIN below the best score."""
        if len(self.scores) < 10:
            return None
        with self.lock:
            covered = set().union(*(s.members for s in self.refined()))
        ranked = sorted(self.scores.items(), key=lambda kv: -kv[1])
        floor = ranked[0][1] - EXPLORE_MARGIN
        for lid, score in ranked:
            if score < floor:
                break
            if lid in covered:
                continue
            name = self.head_name(lid)
            if name and not self._family_searched(frozenset(name.split())):
                return name, score, lid
        return None

    def refine_names(self):
        """Narrower versions of names already searched ("gamecube" -> "gamecube
        indigo"), ranked by how much they appear in the closest titles (the very
        closest dominate, see FOCUS) and how rare they are overall."""
        if len(self.scores) < 10:
            return []
        closest = self._closest()
        top_score = closest[0][1]
        weight, support, best = Counter(), Counter(), {}
        for lid, score in closest:
            w = math.exp((score - top_score) / FOCUS)
            for p in self.phrases[lid]:
                weight[p] += w
                support[p] += 1
                best[p] = max(best.get(p, score), score)
        n = len(self.scores)
        candidates = []
        for p, w in weight.items():
            key = frozenset(p.split())
            if support[p] < self.args.min_support or key in self.searched_keys:
                continue
            if not any(base < key for base in self.searched_keys):
                continue  # only narrow an existing name, never start a new one here
            candidates.append((w * math.log(1 + n / self.df[p]), support[p], best[p], p))
        candidates.sort(reverse=True)
        return candidates

    def search_stats(self, s, threshold, prior):
        with self.lock:
            members = list(s.members)
        ranked = sorted((lid for lid in members if lid in self.scores), key=lambda lid: -self.scores[lid])
        if not ranked:
            return None
        vals = [self.scores[lid] for lid in ranked]
        top = vals[:TOP_N]
        hits = sum(v >= threshold for v in vals)
        return {
            "search": s,
            "n": len(vals),
            "top": (sum(top) + (TOP_N - len(top)) * prior) / TOP_N,
            "hits": hits,
            "rate": (hits + HIT_PRIOR * BASE_RATE) / (len(vals) + HIT_PRIOR),
            "best": vals[0],
            "ranked": ranked,
        }


def drain(items, size, timeout=0.3):
    try:
        batch = [items.get(timeout=timeout)]
    except queue.Empty:
        return []
    while len(batch) < size:
        try:
            batch.append(items.get_nowait())
        except queue.Empty:
            break
    return batch


# ------------------------------------------------------------------ main

def parse_args():
    ap = argparse.ArgumentParser(description="Identify the object in a photo from tutti.ch listings.")
    ap.add_argument("query_image", help="photo of the object")
    ap.add_argument("-c", "--category", action="append",
                    help="tutti category or group, repeatable (e.g. consolesGames)")
    ap.add_argument("--seed", help="optional rough word to narrow the first search")
    ap.add_argument("--seed-limit", type=int, default=800, help="listings from the first search (default: 800)")
    ap.add_argument("--per-search", type=int, default=200, help="listings per name search (default: 200)")
    ap.add_argument("--max-searches", type=int, default=8, help="names to try in total (default: 8)")
    ap.add_argument("--parallel", type=int, default=3, help="name searches running at once (default: 3)")
    ap.add_argument("--top-k", type=int, default=30, help="closest titles mined for names (default: 30)")
    ap.add_argument("--min-support", type=int, default=2,
                    help="refine: a narrower name must appear in this many of those titles (default: 2)")
    ap.add_argument("--confident", type=float, default=0.5,
                    help="warn when the best listing of the winning name scores below this (default: 0.5)")
    ap.add_argument("--min-compared", type=int, default=150,
                    help="listings to compare before the first name search (default: 150)")
    ap.add_argument("--mine-every", type=float, default=3.0, help="seconds between name searches (default: 3)")
    ap.add_argument("--refine-everywhere", action="store_true",
                    help="run name searches in all categories, not just the given ones")
    ap.add_argument("--lang", default="de", choices=["de", "fr", "it"], help="listing URL language (default: de)")
    ap.add_argument("--out", help="folder for thumbnails and CSVs (default: img/explore_<category>)")
    ap.add_argument("--top", type=int, default=10, help="listings to show for the winning name (default: 10)")
    ap.add_argument("--workers", type=int, default=12, help="parallel thumbnail downloads (default: 12)")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between API calls per search (default: 0.3)")
    ap.add_argument("--timeout", type=float, default=10.0, help="thumbnail download timeout in s (default: 10)")
    ap.add_argument("--model", default="vits14", choices=MODELS, help="DINOv2 backbone (default: vits14)")
    ap.add_argument("--size", type=int, default=224, help="input size in px, multiple of 14 (default: 224)")
    ap.add_argument("--no-crop", action="store_true", help="squash images to a square instead of center crop")
    ap.add_argument("--batch-size", type=int, default=32, help="max images per forward pass (default: 32)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps (default: auto)")
    return ap.parse_args()


def launch_next(ex):
    explore = ex.explore_name()
    if explore:
        name, score, lid = explore
        why = f'explore: from "{short(ex.rows[lid].get("title"), 40)}", {score:.3f}'
    else:
        candidates = ex.refine_names()
        if not candidates:
            return False
        _, support, best, name = candidates[0]
        why = f"refine: in {support} of the {ex.args.top_k} closest titles, best {best:.3f}"
    ex.launch(name, why=why)
    say(f'searching "{name}" ({why})')
    return True


def main():
    args = parse_args()

    query_path = Path(args.query_image).expanduser().resolve()
    if not query_path.is_file():
        sys.exit(f"Query image not found: {query_path}")
    if args.size % 14 != 0:
        sys.exit(f"--size must be a multiple of 14 (DINOv2 uses 14px patches), got {args.size}")
    if not args.category and not args.seed:
        sys.exit("Give at least a category (-c consolesGames) or a seed word (--seed konsole).")

    categories = expand_categories(args.category)
    default_name = "explore_" + slugify(args.seed or "_".join(args.category))
    out_dir = Path(args.out or Path("img") / default_name).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = not args.no_crop
    transform = build_transform(args.size, crop)
    cache_path = out_dir / f".dinov2_{args.model}_{args.size}_{'crop' if crop else 'full'}.npz"
    cache = load_cache(cache_path)

    ex = Explorer(args, out_dir, cache, transform, categories)
    seed = ex.launch(args.seed, is_seed=True)  # starts while the model loads

    device = pick_device(args.device)
    log(f"Seed search: {seed.label}{' / ' + ', '.join(c for c in categories if c) if args.seed and args.category else ''}")
    log(f"Loading dinov2_{args.model} on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{args.model}", trust_repo=True)
    model = model.to(device).eval()
    q = embed(model, [transform(load_image(query_path))], device)[0]
    log("Model ready. Ctrl+C stops and shows the results so far.\n")

    new_cache = dict(cache)
    best = None
    last_mine = 0.0

    try:
        while True:
            batch = drain(ex.items, args.batch_size)
            if batch:
                fresh = [it for it in batch if it[4] is None]
                vecs = embed(model, [it[3] for it in fresh], device) if fresh else []
                computed = {it[0]: v for it, v in zip(fresh, vecs)}
                for name, mtime, row, _, cached in batch:
                    vec = cached if cached is not None else computed[name]
                    new_cache[name] = (mtime, vec)
                    score = float(vec @ q)
                    lid = row["listingID"]
                    ex.add_result(lid, score, row, name)
                    if best is None or score > best[0]:
                        best = (score, lid)
                        say(f"new best  {score:.3f}  {price_label(row):>12}  {short(row.get('title'), 45)}"
                            f"  [{row['_found_by']}]  {row.get('url', '')}")

            ready = len(ex.scores) >= args.min_compared or seed.done.is_set()
            now = time.monotonic()
            if ready and now - last_mine >= args.mine_every and ex.can_launch():
                last_mine = now
                launch_next(ex)

            running = sum(not s.done.is_set() for s in ex.searches)
            status(f"  {len(ex.scores)} compared, {running} search(es) running, "
                   f"{len(ex.refined())}/{args.max_searches} names tried"
                   + (f", best {best[0]:.3f}" if best else ""))

            if ex.idle():
                if ex.can_launch() and launch_next(ex):
                    continue
                break
    except KeyboardInterrupt:
        log("\nStopped, here is what was found so far.")
    finally:
        ex.stop.set()
        clear_status()
        ex.pool.shutdown(wait=False, cancel_futures=True)

    if new_cache:
        save_cache(cache_path, new_cache)
    if not ex.scores:
        sys.exit("No listing could be compared. If the seed search found nothing, "
                 "try adding a rough word with --seed.")

    # ---------------------------------------------------------- report
    with ex.lock:
        seed_members = list(seed.members)
    background = [ex.scores[lid] for lid in seed_members if lid in ex.scores]
    if len(background) < 50:
        background = list(ex.scores.values())
    threshold = float(np.quantile(background, 1 - BASE_RATE))
    prior = statistics.median(ex.scores.values())

    stats = [st for s in ex.searches if (st := ex.search_stats(s, threshold, prior))]
    stats.sort(key=lambda st: (-st["top"], -st["rate"]))

    print(f"\nSearches ranked by the mean score of their {TOP_N} closest listings. 'strong' counts results"
          f" scoring {threshold:.3f} or more, a level only the top {BASE_RATE:.0%} of the seed reaches.\n")
    print(f"{'rank':>4}  {'top' + str(TOP_N):>5}  {'best':>5}  {'strong':>9}  search")
    for i, st in enumerate(stats, 1):
        s = st["search"]
        strong = f"{st['hits']}/{st['n']}"
        print(f"{i:>4}  {st['top']:.3f}  {st['best']:.3f}  {strong:>9}  {s.label}"
              + ("  (seed)" if s.is_seed else ""))

    refined = [st for st in stats if not st["search"].is_seed]
    if refined:
        contenders = [st for st in refined if st["top"] >= refined[0]["top"] - TIE]
        winner = max(contenders, key=lambda st: (len(tokens(st["search"].text)), st["rate"]))
        print(f'\nMost likely product: "{winner["search"].label}"')
        if winner["best"] < args.confident:
            print(f"Low confidence: its closest listing only scores {winner['best']:.3f}. Listings showing "
                  f"the same object usually score well above {args.confident} (rough rule of thumb), so "
                  "the item itself was probably never among the listings compared. Try a larger "
                  "--seed-limit or a rough --seed word.")
    else:
        winner = max(stats, key=lambda st: st["best"])
        print("\nNo name stood out. Try a larger --seed-limit or a rough --seed word.")

    top = winner["ranked"][: max(args.top, 1)]
    print(f"\nClosest listings from that search:\n")
    print(f"{'rank':>4}  {'score':>5}  {'price':>12}  {'title':<50}  link")
    for i, lid in enumerate(top, 1):
        row = ex.rows[lid]
        print(f"{i:>4}  {ex.scores[lid]:.3f}  {price_label(row):>12}  "
              f"{short(row.get('title'), 50):<50}  {row.get('url', '')}")
    prices = [p for p in (numeric_price(ex.rows[lid]) for lid in top) if p is not None]
    if prices:
        print(f"\nMedian price of these {len(top)} ({len(prices)} with a price): {chf(statistics.median(prices))}")

    # ---------------------------------------------------------- CSVs
    found_in = defaultdict(list)
    for s in ex.searches:
        with ex.lock:
            members = list(s.members)
        for lid in members:
            found_in[lid].append(s.label)

    with open(out_dir / "searches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "search", "seed", f"top{TOP_N}_mean", "best_score", "strong", "n", "strong_rate",
                    "strong_threshold", "why"])
        for i, st in enumerate(stats, 1):
            s = st["search"]
            w.writerow([i, s.label, s.is_seed, f"{st['top']:.4f}", f"{st['best']:.4f}", st["hits"], st["n"],
                        f"{st['rate']:.4f}", f"{threshold:.4f}", s.why])

    with open(out_dir / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "score", "price", "formattedPrice", "title", "url", "image", "listingID", "searches"])
        ranked = sorted(ex.scores, key=lambda lid: -ex.scores[lid])
        for i, lid in enumerate(ranked, 1):
            row = ex.rows[lid]
            p = numeric_price(row)
            w.writerow([i, f"{ex.scores[lid]:.4f}", "" if p is None else p, row.get("formattedPrice", ""),
                        row.get("title", ""), row.get("url", ""), ex.images[lid], lid,
                        "; ".join(found_in[lid])])

    print(f"\nSaved {out_dir / 'searches.csv'} and {out_dir / 'matches.csv'}")
    if ex.failed:
        print(f"{ex.failed} thumbnail(s) could not be downloaded or read.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)