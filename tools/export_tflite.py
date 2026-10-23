"""
Export YOLOE as an object detector (LiteRT .tflite) for the Android app, plus what the app needs to name
the object you aim at.

The 4585-class vocabulary and the mask branch are not run in the model. What's left is the YOLOE backbone,
the box head, one "is there an object here" score per anchor, and the anchor's class embedding:

    output 0  [1, anchors, 5]    x0, y0, x1, y1 normalized to the model input (0..1), objectness (0..1)
    output 1  [1, anchors, 128]  class embedding (YOLOE's cls features, the input of its vocabulary layer)

The app runs class-agnostic NMS on output 0. For the box under the reticle it reads output 1 at that box's
anchor and scores the vocabulary itself (sigmoid(W e + b), 4585 x 128 per level), which costs nothing
next to running the vocabulary for all anchors in the model:

    yoloe_vocab.bin    "YVOC", version, levels, classes, dim, then per level W [classes, dim] and b [classes]
                       as little-endian float16, then the anchor count of each level (int32)
    yoloe_names.txt    the class names, one per line, in vocabulary order

By default the vocabulary is tools/product_vocab.txt: product types people sell, turned into vocabulary rows by
YOLOE's own text encoder (the text-prompt checkpoint yoloe-26s-seg.pt with MobileCLIP2, which reproduces the
prompt-free model's built-in rows exactly and shares its backbone). YOLOE's 4585 built-in tags include many
scene and abstract words ("bank vault", "hurricane") that make poor product names; --vocab builtin keeps them.

Objectness comes from:
  --score pf       YOLOE's prompt-free proposal filter (the LRPC "pf" conv): one 1x1 conv, nearly free.
                   Good on yoloe-26s/m/l. Default when the checkpoint has a usable one.
  --score classes  max over the 4585 class scores: about 50% slower, needed for yoloe-26n
                   (its proposal filter is not usable as a score).

Usage (from the repo root, on Linux or macOS: LiteRT has no Windows wheels):
    pip install "ultralytics>=8.4" litert-torch ai-edge-litert ai-edge-quantizer git+https://github.com/ultralytics/CLIP.git
    python tools/export_tflite.py                          # yoloe-26s-seg-pf, 640
    python tools/export_tflite.py --imgsz 480              # faster, less detail
    python tools/export_tflite.py --check                  # compare TFLite with PyTorch on test images

Writes android/app/src/main/assets/yoloe.tflite, yoloe_vocab.bin and yoloe_names.txt.
"""
import argparse
import struct
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def objectness_head(self, x):
    """Replacement for YOLOEDetect.forward_lrpc: decoded boxes, one objectness per anchor, class embeddings."""
    x = x[: self.nl]  # the head also receives extra inputs, only the feature maps matter here
    bs = x[0].shape[0]
    boxes, scores, feats = [], [], []
    for i in range(self.nl):
        cls_feat = self.cv3[i](x[i])
        boxes.append(self.lrpc[i].loc(self.cv2[i](x[i])).view(bs, 4 * self.reg_max, -1))
        if self.score_mode == "pf":
            s = self.lrpc[i].pf(cls_feat)[:, :1]                     # B, 1, H, W   proposal logit
        else:
            s = self.vocab_convs[i](cls_feat).amax(1, keepdim=True)  # B, 1, H, W   best class logit
        scores.append(s.view(bs, 1, -1))
        feats.append(cls_feat.view(bs, cls_feat.shape[1], -1))
    dbox = self._get_decode_boxes({"boxes": torch.cat(boxes, 2), "feats": x, "index": None})  # B, 4, A pixels xyxy
    out = torch.cat([dbox / self.input_wh, torch.cat(scores, 2).sigmoid()], 1)  # B, 5, A
    return out.permute(0, 2, 1), torch.cat(feats, 2).permute(0, 2, 1)            # B, A, 5 and B, A, C


class NHWC(nn.Module):
    """The app feeds NHWC float images, litert_torch traces NCHW: put the transpose inside the model."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x.permute(0, 3, 1, 2))


def vocab_weights(head):
    """Per level: W [classes, dim] and b [classes] of the vocabulary layer, the same math as LRPCHead."""
    out = []
    for lr in head.lrpc:
        w = lr.vocab.weight.data.reshape(lr.vocab.weight.shape[0], -1).float()
        out.append((w, lr.vocab.bias.data.float()))
    return out


def build(weights, imgsz, score="auto"):
    from ultralytics import YOLOE
    from ultralytics.nn.modules.head import YOLOEDetect

    yoloe = YOLOE(weights)
    net = deepcopy(yoloe.model).float().eval()
    for p in net.parameters():
        p.requires_grad = False

    head = net.model[-1]
    assert hasattr(head, "lrpc"), "needs a prompt-free checkpoint (*-pf.pt)"
    vocab = vocab_weights(head)
    if score == "auto":  # yoloe-26n's proposal filter has 512 channels and does not rank objects
        score = "pf" if head.lrpc[0].pf.out_channels == 1 else "classes"
    head.__class__ = YOLOEDetect  # drop the mask branch: no prototypes, no mask coefficients
    head.export, head.dynamic, head.xyxy, head.format = True, False, True, "litert"
    head.score_mode = score
    head.register_buffer("input_wh", torch.tensor([imgsz] * 4, dtype=torch.float32).view(1, 4, 1))
    if score == "classes":  # vocabulary as a 1x1 conv (it is a Linear on some levels)
        head.vocab_convs = nn.ModuleList()
        for w, b in vocab:
            c = nn.Conv2d(w.shape[1], w.shape[0], 1)
            c.weight.data = w.clone().view(w.shape[0], w.shape[1], 1, 1)
            c.bias.data = b.clone()
            head.vocab_convs.append(c)
    head.forward = types.MethodType(objectness_head, head)
    with torch.no_grad():
        net(torch.zeros(1, 3, imgsz, imgsz))  # caches anchors and strides before tracing
    levels = [imgsz // int(s) for s in head.stride]
    counts = [n * n for n in levels]
    names = [yoloe.names[i] for i in range(len(yoloe.names))]
    return NHWC(net).eval(), score, vocab, counts, names


def text_vocab(weights, names):
    """Vocabulary rows for [names] from YOLOE's text encoder (downloads the encoder on first use)."""
    from ultralytics import YOLOE

    tp = YOLOE(weights)
    vocab = tp.model.get_vocab(names)
    return [(v.weight.data.reshape(len(names), -1).float(), v.bias.data.float()) for v in vocab]


def read_names(path):
    lines = (line.split("#")[0].strip() for line in Path(path).read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(n for n in lines if n))


def write_vocab(path, vocab, counts):
    import numpy as np

    with open(path, "wb") as f:
        f.write(b"YVOC")
        f.write(struct.pack("<4i", 1, len(vocab), vocab[0][0].shape[0], vocab[0][0].shape[1]))
        for w, b in vocab:
            f.write(w.numpy().astype("<f2").tobytes())
            f.write(b.numpy().astype("<f2").tobytes())
        f.write(np.asarray(counts, dtype="<i4").tobytes())


def to_fp16(tfl):
    """Store weights as FP16 (half the size). Only the legacy GPU delegate accepts the DEQUANTIZE ops this adds."""
    from ai_edge_quantizer import quantizer, recipe, recipe_manager

    rm = recipe_manager.RecipeManager()
    rm.add_weight_only_config(
        regex=".*", operation_name=recipe.TFLOperationName.ALL_SUPPORTED, num_bits=16,
        algorithm_key=recipe.AlgorithmName.FLOAT_CASTING,
    )
    qt = quantizer.Quantizer(str(tfl))
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(str(tfl), overwrite=True)


def letterbox(img_bgr, imgsz):
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    s = min(imgsz / w, imgsz / h)
    nw, nh = round(w * s), round(h * s)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    dx, dy = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas[..., ::-1].astype(np.float32) / 255.0  # RGB, 0..1, HWC


def nms(rows, min_score=0.4, iou=0.5):
    """What the app does: threshold, sort, class-agnostic NMS. rows: [A, 5]. Returns kept anchor indices."""
    import torchvision

    rows = torch.as_tensor(rows)
    idx = torch.nonzero(rows[:, 4] >= min_score).flatten()
    keep = torchvision.ops.nms(rows[idx, :4], rows[idx, 4], iou)
    return idx[keep]


def name_box(feats, anchor, vocab, counts, names, k=3):
    """What the app does for the box under the reticle: score the vocabulary at the box's anchor."""
    level, start = 0, 0
    while anchor >= start + counts[level]:
        start += counts[level]
        level += 1
    w, b = vocab[level]
    p = torch.sigmoid(w @ torch.as_tensor(feats[anchor]).float() + b)
    top = torch.topk(p, k)
    return [(names[i], float(v)) for v, i in zip(top.values, top.indices)]


def check(model, tfl, imgsz, vocab, counts, names, weights, images):
    import cv2
    import numpy as np
    import torchvision
    from ai_edge_litert.interpreter import Interpreter
    from ultralytics import YOLOE
    from ultralytics.utils import ASSETS

    ref_model = YOLOE(weights)
    it = Interpreter(model_path=str(tfl))
    it.allocate_tensors()
    outs = it.get_output_details()
    for path in images or [ASSETS / "bus.jpg", ASSETS / "zidane.jpg"]:
        img = cv2.imread(str(path))
        x = letterbox(img, imgsz)[None]
        with torch.no_grad():
            pt_boxes, pt_feats = (t[0].numpy() for t in model(torch.from_numpy(x.copy())))
        it.set_tensor(it.get_input_details()[0]["index"], x)
        it.invoke()
        tf = [it.get_tensor(o["index"])[0] for o in outs]
        tf_boxes = next(t for t in tf if t.shape[-1] == 5)
        tf_feats = next(t for t in tf if t.shape[-1] != 5)
        a, b = nms(pt_boxes).numpy(), nms(tf_boxes).numpy()
        same = (torchvision.ops.box_iou(torch.as_tensor(pt_boxes[a, :4]), torch.as_tensor(tf_boxes[b, :4])).max(1).values > 0.9
                if len(a) and len(b) else torch.zeros(1))
        print(f"{Path(path).name}: max |PyTorch - TFLite| score {np.abs(pt_boxes[:, 4] - tf_boxes[:, 4]).max():.4f}, "
              f"features {np.abs(pt_feats - tf_feats).max():.4f}, {len(a)} boxes in PyTorch, {len(b)} in TFLite, "
              f"{float(same.float().mean()):.0%} identical")
        ref = ref_model.predict(str(path), imgsz=imgsz, conf=0.25, verbose=False)[0]
        ref_names = sorted({ref.names[int(c)] for c in ref.boxes.cls})
        print(f"   Ultralytics names (built-in tags): {', '.join(ref_names[:12])}")
        for i in b[:8]:
            i = int(i)
            x0, y0, x1, y1 = (tf_boxes[i, :4] * imgsz).round().astype(int).tolist()
            top = ", ".join(f"{n} {p:.2f}" for n, p in name_box(tf_feats, i, vocab, counts, names))
            print(f"   ({x0:3d},{y0:3d})-({x1:3d},{y1:3d})  {tf_boxes[i, 4]:.2f}  {top}")


def main():
    ap = argparse.ArgumentParser(description="Export YOLOE as a LiteRT detector for the Android app")
    ap.add_argument("--model", default="yoloe-26s-seg-pf.pt", help="prompt-free YOLOE checkpoint")
    ap.add_argument("--imgsz", type=int, default=640, help="square input size, multiple of 32")
    ap.add_argument("--score", choices=["auto", "pf", "classes"], default="auto", help="objectness source, see top")
    ap.add_argument("--fp16", action="store_true", help="FP16 weights: half the size, legacy GPU delegate only")
    ap.add_argument("--out", default=str(ROOT / "android/app/src/main/assets"), help="output folder")
    ap.add_argument("--check", action="store_true", help="compare PyTorch and TFLite on test images")
    ap.add_argument("--images", nargs="*", help="images for --check (default: Ultralytics' bus and zidane)")
    ap.add_argument("--skip-export", action="store_true", help="only run --check on the files already in --out")
    ap.add_argument("--vocab", default=str(ROOT / "tools/product_vocab.txt"),
                    help="names to recognise, one per line, or 'builtin' for YOLOE's 4585 tags")
    ap.add_argument("--text-model", default="yoloe-26s-seg.pt", help="text-prompt checkpoint matching --model")
    args = ap.parse_args()
    assert args.imgsz % 32 == 0, "imgsz must be a multiple of 32"

    model, score, vocab, counts, names = build(args.model, args.imgsz, args.score)
    if args.vocab != "builtin":
        names = read_names(args.vocab)
        vocab = text_vocab(args.text_model, names)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tfl = out / "yoloe.tflite"
    write_vocab(out / "yoloe_vocab.bin", vocab, counts)
    (out / "yoloe_names.txt").write_text("\n".join(n.replace("\n", " ") for n in names) + "\n", encoding="utf-8")
    if args.skip_export:  # the vocabulary files are rewritten, the detector is kept
        check(model, tfl, args.imgsz, vocab, counts, names, args.model, args.images)
        return
    import litert_torch

    litert_torch.convert(model, (torch.zeros(1, args.imgsz, args.imgsz, 3),)).export(str(tfl))
    if args.fp16:
        to_fp16(tfl)
    print(f"wrote {tfl} ({tfl.stat().st_size / 1e6:.1f} MB), {args.model} at {args.imgsz}, objectness from {score}, "
          f"anchors per level {counts}; vocabulary {len(names)} x {vocab[0][0].shape[1]}")
    if args.check:
        check(model, tfl, args.imgsz, vocab, counts, names, args.model, args.images)


if __name__ == "__main__":
    main()
