"""
Export DINOv2-small as the app's re-identification model (LiteRT .tflite).

The app embeds the crop of every box it needs to recognise: the box you tap (a saved item) and
new boxes that appear later. If a new box looks like a saved item, it gets that item's number
and "saved" mark back, even after the object left the frame.

    input  [1, S, S, 3] RGB 0..1, the box letterboxed to S x S (ImageNet normalisation is inside)
    output [1, 384]     L2-normalised descriptor: normalise(normalise(CLS) + normalise(mean of patch tokens))

Cosine similarity between two descriptors is just their dot product. On crops saved from the app,
the same object seen again scores about 0.85 to 0.97 and different objects about 0.3 on average
(two open laptops still reach about 0.85, hence the margin rule in the app).

The graph is DINOv2 rewritten for mobile GPUs (dinov2_gpu.py): with the stock graph, the GPU of LiteRT 2's
CompiledModel returned almost the same descriptor for every input on a Galaxy Z Fold4 (two unrelated test
patterns scored 0.94 instead of 0.45), while the CPU was fine. The app still checks this at start-up
(Embedder.selfCheck) and falls back to the CPU if the GPU gets it wrong.

Usage (from the repo root, on Linux or macOS: LiteRT has no Windows wheels):
    pip install transformers litert-torch ai-edge-litert ai-edge-quantizer
    python tools/export_embedder.py                       # downloads facebook/dinov2-small
    python tools/export_embedder.py --weights path/to/dinov2-small --check

Writes android/app/src/main/assets/embedder.tflite.
"""
import argparse
from pathlib import Path

import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov2_gpu import Dinov2Gpu  # noqa: E402


def reference(dino, x):
    """The stock Hugging Face forward, to check the rewrite above against."""
    m = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    sd = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    h = dino(pixel_values=((x - m) / sd).permute(0, 3, 1, 2)).last_hidden_state
    e = F.normalize(h[:, 0], dim=-1) + F.normalize(h[:, 1:].mean(1), dim=-1)
    return F.normalize(e, dim=-1)


def main():
    ap = argparse.ArgumentParser(description="Export DINOv2-small as the app's re-identification model")
    ap.add_argument("--weights", default="facebook/dinov2-small", help="Hugging Face id or local folder")
    ap.add_argument("--size", type=int, default=168, help="input size, multiple of 14 (168 = 12 x 12 patches)")
    ap.add_argument("--fp16", action="store_true", help="FP16 weights (44 MB): broken on LiteRT 2's GPU, see top")
    ap.add_argument("--out", default=str(ROOT / "android/app/src/main/assets"), help="output folder")
    ap.add_argument("--check", action="store_true", help="compare PyTorch and TFLite on random crops")
    args = ap.parse_args()
    assert args.size % 14 == 0, "size must be a multiple of 14"

    import litert_torch
    from transformers import Dinov2Model

    dino = Dinov2Model.from_pretrained(args.weights, attn_implementation="eager").eval()
    model = Dinov2Gpu(dino, args.size).eval()
    sample = torch.rand(1, args.size, args.size, 3)
    with torch.no_grad():
        print(f"cosine(rewrite, Hugging Face) = {float(model(sample)[0] @ reference(dino, sample)[0]):.6f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tfl = out / "embedder.tflite"
    litert_torch.convert(model, (sample,)).export(str(tfl))
    if args.fp16:
        from export_tflite import to_fp16

        to_fp16(tfl)
    print(f"wrote {tfl} ({tfl.stat().st_size / 1e6:.1f} MB), input {args.size}")

    if args.check:
        from ai_edge_litert.interpreter import Interpreter

        it = Interpreter(model_path=str(tfl))
        it.allocate_tensors()
        for _ in range(3):
            x = torch.rand(1, args.size, args.size, 3)
            with torch.no_grad():
                ref = model(x)[0].numpy()
            it.set_tensor(it.get_input_details()[0]["index"], x.numpy())
            it.invoke()
            got = it.get_tensor(it.get_output_details()[0]["index"])[0]
            print(f"cosine(PyTorch, TFLite) = {float(ref @ got):.5f}")


if __name__ == "__main__":
    main()
