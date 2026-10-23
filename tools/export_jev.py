"""
Export JEV (Tev1-0.8B-experimental, Together AI) as a GGUF file for the app's llama.cpp runtime.

    python tools/export_jev.py                # download, convert, write android/app/src/main/assets/jev.gguf
    python tools/export_jev.py --check        # also compare the GGUF (through tools/jev_native_test) with
                                              # the original model in transformers, on the app's prompts

Steps: download the checkpoint from Hugging Face into tools/.work/tev1-0.8b, check out llama.cpp at the
same tag as the app (android/app/src/main/cpp/CMakeLists.txt) into tools/.work/llama.cpp, then run its
convert_hf_to_gguf.py (text model only, the vision tower is dropped) with Q8_0 weights (about 0.8 GB).

Needs git and a Python with torch, transformers (with Qwen3.5), numpy, sentencepiece, safetensors, e.g.
    uv venv tools/.work/venv --python 3.12
    uv pip install --python tools/.work/venv torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python tools/.work/venv numpy sentencepiece transformers safetensors "protobuf<5" jinja2
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tools/.work"
REPO = "togethercomputer/Tev1-0.8B-experimental"

CHECK_LABELS = [
    ("video game console", ["game controller"]), ("chair", []), ("headphones", ["earphone"]),
    ("laptop", []), ("table lamp", []), ("bicycle", []), ("sneakers", []), ("guitar", []),
    ("television", []), ("smartphone", []), ("teddy bear", []), ("coffee machine", []),
    ("wristwatch", []), ("backpack", []), ("camera", ["lens"]), ("book", []),
]


def llama_tag():
    cm = (ROOT / "android/app/src/main/cpp/CMakeLists.txt").read_text()
    return re.search(r"set\(LLAMA_TAG (\S+)\)", cm).group(1)


def fetch_model(dest):
    """Hugging Face checkpoint, through huggingface_hub when installed, else git (with git-lfs)."""
    if (dest / "config.json").exists():
        return dest
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, local_dir=str(dest))
    except ImportError:
        subprocess.check_call(["git", "clone", "--depth", "1", f"https://huggingface.co/{REPO}", str(dest)])
    return dest


def checkout_llama(dest, tag):
    if not (dest / "convert_hf_to_gguf.py").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", tag,
                               "https://github.com/ggml-org/llama.cpp", str(dest)])
    return dest


def convert(model_dir, llama_dir, out, outtype):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(model_dir),
                           "--outfile", str(out), "--outtype", outtype])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def reference(model_dir, prompts):
    """Letter probabilities from the original checkpoint in transformers (float32, CPU)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32).eval()
    # The app's prompt builder must match the checkpoint's own chat template exactly.
    for p in prompts:
        msgs = [{"role": "system", "content": jev_prompt.SYSTEM}, {"role": "user", "content": p["user"]}]
        expected = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert expected == p["prompt"], "prompt differs from the chat template"
    out = []
    for p in prompts:
        ids = tok(p["prompt"], return_tensors="pt", add_special_tokens=False).input_ids
        letter_ids = [tok.encode(x, add_special_tokens=False) for x in jev_prompt.LETTERS[:len(p["keys"])]]
        assert all(len(i) == 1 for i in letter_ids)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1]
        out.append(softmax([float(logits[i[0]]) for i in letter_ids]))
    return out


def native(binary, gguf, prompts):
    """Letter probabilities from the GGUF through the app's own C++ code (tools/jev_native_test)."""
    req = WORK / "jev_check_prompts.json"
    req.write_text(json.dumps([{"prompt": p["prompt"], "n": len(p["keys"])} for p in prompts]), encoding="utf-8")
    res = subprocess.run([str(binary), str(gguf), str(req)], check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
    return [softmax(r["logits"]) for r in rows], [r["ms"] for r in rows], [r["tokens"] for r in rows]


def check(model_dir, gguf, binary):
    cats = jev_prompt.load_categories()
    prompts = []
    for label, alts in CHECK_LABELS:
        state = jev_prompt.state_for(label, alts)
        text, opts = jev_prompt.group_prompt(cats, state)
        user = jev_prompt.user_json(state, cats["question_group"], opts)
        prompts.append({"label": label, "prompt": text, "user": user, "keys": [k for k, _ in opts]})
    ref = reference(model_dir, prompts)
    nat = native(binary, gguf, prompts) if binary else None
    worst = 0.0
    for i, p in enumerate(prompts):
        r = ref[i]
        top = max(range(len(r)), key=r.__getitem__)
        line = f"{p['label']:<20} {p['keys'][top]:<22} {r[top]:.3f}"
        if nat:
            n = nat[0][i]
            ntop = max(range(len(n)), key=n.__getitem__)
            diff = max(abs(a - b) for a, b in zip(r, n))
            worst = max(worst, diff)
            line += f"   gguf: {p['keys'][ntop]:<22} {n[ntop]:.3f}  max |dp| {diff:.3f}  {nat[2][i]} tok {nat[1][i]:.0f} ms"
            if ntop != top:
                line += "  <-- different answer"
        print(line)
    if nat:
        print(f"worst probability difference: {worst:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Export JEV (Tev1-0.8B) as GGUF for the Android app")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "bf16", "f16", "f32"])
    ap.add_argument("--out", default=str(ROOT / "android/app/src/main/assets/jev.gguf"))
    ap.add_argument("--check", action="store_true", help="compare with transformers on the app's prompts")
    ap.add_argument("--native", default=None, help="jev_test binary for --check (default: tools/jev_native_test/build)")
    ap.add_argument("--skip-convert", action="store_true", help="only run --check on an existing GGUF")
    args = ap.parse_args()

    model_dir = WORK / "tev1-0.8b"
    out = Path(args.out)
    if not args.skip_convert:
        fetch_model(model_dir)
        llama_dir = checkout_llama(WORK / "llama.cpp", llama_tag())
        convert(model_dir, llama_dir, out, args.outtype)
    if args.check:
        binary = args.native
        if binary is None:
            build = ROOT / "tools/jev_native_test/build"
            found = [f for f in build.rglob("jev_test*") if f.is_file() and f.suffix in ("", ".exe")] if build.exists() else []
            binary = str(found[0]) if found else None
            if binary is None:
                print("no tools/jev_native_test build: checking the prompt format and the reference only")
        check(model_dir, out, binary)


if __name__ == "__main__":
    main()
