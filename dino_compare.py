import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MODELS = [
    "vits14", "vitb14", "vitl14", "vitg14",
    "vits14_reg", "vitb14_reg", "vitl14_reg", "vitg14_reg",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Rank a folder of images by DINOv2 similarity to a query image.")
    ap.add_argument("query", help="query image")
    ap.add_argument("folder", help="folder of images to compare against")
    ap.add_argument("--model", default="vits14", choices=MODELS, help="DINOv2 backbone (default: vits14)")
    ap.add_argument("--size", type=int, default=224, help="input size in px, multiple of 14 (default: 224)")
    ap.add_argument("--no-crop", action="store_true",
                    help="squash the whole image to a square instead of resize + center crop")
    ap.add_argument("--top", type=int, default=10, help="how many results to print, 0 for all (default: 10)")
    ap.add_argument("--threshold", type=float, default=None, help="only print results with score >= this")
    ap.add_argument("--csv", help="write every score to this CSV file")
    ap.add_argument("-r", "--recursive", action="store_true", help="include images in subfolders")
    ap.add_argument("--batch-size", type=int, default=32, help="images per forward pass (default: 32)")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps (default: auto)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the embedding cache")
    return ap.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(size, crop):
    bicubic = transforms.InterpolationMode.BICUBIC
    if crop:
        resize = [transforms.Resize(size, interpolation=bicubic), transforms.CenterCrop(size)]
    else:
        resize = [transforms.Resize((size, size), interpolation=bicubic)]
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return transforms.Compose(resize + [transforms.ToTensor(), normalize])


def load_image(path, background=(255, 255, 255), margin=0.08):
    """Open an image upright as RGB.

    A cutout with transparency (e.g. a segmentation mask saved as PNG) is cropped
    to the object, padded to a square and put on a white background. Without this,
    PIL fills the transparent area with whatever colour hides under the alpha,
    usually black, which looks nothing like a listing photo.
    """
    img = ImageOps.exif_transpose(Image.open(path))  # respect phone camera rotation
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        img = img.convert("RGBA")
        alpha = img.getchannel("A")
        if alpha.getextrema()[0] < 255:
            box = alpha.point(lambda a: 255 if a > 16 else 0).getbbox()
            if box:
                img = img.crop(box)
            side = int(max(img.size) * (1 + 2 * margin))
            canvas = Image.new("RGBA", (side, side), (*background, 255))
            canvas.alpha_composite(img, ((side - img.width) // 2, (side - img.height) // 2))
            return canvas.convert("RGB")
    return img.convert("RGB")


@torch.inference_mode()
def embed(model, tensors, device):
    x = torch.stack(tensors).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        feats = model(x)
    return F.normalize(feats.float(), dim=-1).cpu().numpy()


def list_images(folder, recursive):
    files = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_cache(path):
    if not path.exists():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        return {
            p: (m, e)
            for p, m, e in zip(data["paths"].tolist(), data["mtimes"].tolist(), data["embs"])
        }
    except Exception as exc:
        print(f"Ignoring unreadable cache {path.name}: {exc}", file=sys.stderr)
        return {}


def save_cache(path, entries):
    if not entries:
        return
    paths = sorted(entries)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(
                f,
                paths=np.array(paths),
                mtimes=np.array([entries[p][0] for p in paths], dtype=np.int64),
                embs=np.stack([entries[p][1] for p in paths]).astype(np.float32),
            )
        tmp.replace(path)
    except OSError as exc:
        print(f"Could not write cache ({exc}), continuing without it.", file=sys.stderr)


def main():
    args = parse_args()

    query = Path(args.query).expanduser().resolve()
    folder = Path(args.folder).expanduser().resolve()
    if not query.is_file():
        sys.exit(f"Query image not found: {query}")
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    if args.size % 14 != 0:
        sys.exit(f"--size must be a multiple of 14 (DINOv2 uses 14px patches), got {args.size}")

    gallery = [p for p in list_images(folder, args.recursive) if p.resolve() != query]
    if not gallery:
        sys.exit(f"No images found in {folder}")

    device = pick_device(args.device)
    print(f"Loading dinov2_{args.model} on {device}...", file=sys.stderr)
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{args.model}", trust_repo=True)
    model = model.to(device).eval()

    crop = not args.no_crop
    transform = build_transform(args.size, crop)

    cache_path = folder / f".dinov2_{args.model}_{args.size}_{'crop' if crop else 'full'}.npz"
    cache = {} if args.no_cache else load_cache(cache_path)

    # Reuse cached embeddings for unchanged files, queue the rest.
    entries = {}
    todo = []
    for p in gallery:
        rel = str(p.relative_to(folder))
        mtime = p.stat().st_mtime_ns
        hit = cache.get(rel)
        if hit is not None and hit[0] == mtime:
            entries[rel] = hit
        else:
            todo.append((rel, p, mtime))

    if todo:
        print(f"Embedding {len(todo)} image(s), {len(entries)} from cache.", file=sys.stderr)
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            tensors, kept = [], []
            for rel, p, mtime in chunk:
                try:
                    tensors.append(transform(load_image(p)))
                    kept.append((rel, mtime))
                except Exception as exc:
                    print(f"\nSkipping {rel}: {exc}", file=sys.stderr)
            if tensors:
                for (rel, mtime), e in zip(kept, embed(model, tensors, device)):
                    entries[rel] = (mtime, e)
            done = min(start + args.batch_size, len(todo))
            print(f"\r  {done}/{len(todo)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if not args.no_cache:
            save_cache(cache_path, entries)
    else:
        print(f"All {len(entries)} embeddings loaded from cache.", file=sys.stderr)

    if not entries:
        sys.exit("No gallery image could be read.")

    q = embed(model, [transform(load_image(query))], device)[0]
    names = list(entries)
    gallery_embs = np.stack([entries[n][1] for n in names])
    scores = gallery_embs @ q  # cosine similarity, since everything is L2-normalized
    order = np.argsort(-scores)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "score", "image"])
            for rank, i in enumerate(order, 1):
                w.writerow([rank, f"{scores[i]:.4f}", names[i]])
        print(f"Wrote {len(order)} scores to {args.csv}", file=sys.stderr)

    shown = [i for i in order if args.threshold is None or scores[i] >= args.threshold]
    if args.top > 0:
        shown = shown[:args.top]

    print(f"\nQuery: {query.name}   gallery: {len(names)} images   model: dinov2_{args.model} @ {args.size}px\n")
    print(f"{'rank':>4}  {'score':>6}  image")
    for rank, i in enumerate(shown, 1):
        print(f"{rank:>4}  {scores[i]:>6.3f}  {names[i]}")
    if not shown:
        print("  (no results above the threshold)")


if __name__ == "__main__":
    main()