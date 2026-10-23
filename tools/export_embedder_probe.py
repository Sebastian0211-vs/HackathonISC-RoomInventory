"""
Debug models for the embedder on the phone's GPU: the CLS token and the patch-token mean after the patch
embedding and after every encoder layer, as separate outputs (2 x (layers + 1) x [1, 384]).

Push them, start the app, and logcat shows for every output how close the GPU result is to the CPU one
(EmbedderProbe.kt), which tells which layer the GPU gets wrong:

    python tools/export_embedder_probe.py --weights path/to/dinov2-small --out /tmp
    adb push /tmp/embedder_probe_conv.tflite /sdcard/Android/data/ch.hevs.roominventory/files/
    adb logcat -s RoomInventory | grep probe
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov2_gpu import Dinov2Gpu  # noqa: E402


class Probe(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        outs = []
        for i, h in enumerate(self.m.hidden(x)):
            if i == len(self.m.layers):
                h = self.m.norm(h)
            outs += list(self.m.pool(h))
        return tuple(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="facebook/dinov2-small")
    ap.add_argument("--size", type=int, default=168)
    ap.add_argument("--out", default=".")
    ap.add_argument("--layers", type=int, default=12, help="keep only the first N encoder layers (smaller file)")
    args = ap.parse_args()
    import litert_torch
    from transformers import Dinov2Model

    dino = Dinov2Model.from_pretrained(args.weights, attn_implementation="eager").eval()
    dino.encoder.layer = dino.encoder.layer[: args.layers]
    sample = torch.rand(1, args.size, args.size, 3)
    probe = Probe(Dinov2Gpu(dino, args.size)).eval()
    out = Path(args.out) / "embedder_probe_conv.tflite"
    litert_torch.convert(probe, (sample,)).export(str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
