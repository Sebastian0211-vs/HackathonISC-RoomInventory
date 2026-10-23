# Room Inventory, Android

Point the phone at an object and the card at the bottom of the screen tells you what it is and what it sells
for on [tutti.ch](https://www.tutti.ch), refined live for as long as you keep aiming at it. Everything runs on
the phone: object detection, naming, categorisation and image matching. The only network traffic is tutti's
public search API and its listing thumbnails. There is no API key and no cloud service.

## Using it

- **Aim**: the box under the reticle (in the middle) becomes the item in focus after a few frames, shown
  with corner marks. The card shows its name, the price (median of the closest listings) and how sure the
  app is (*weak*, *likely* or *strong match*), the tutti category, and on the right the picture of the closest
  listing. The longer you aim at it, the better the answer gets; looking away pauses the search for that item.
- **Tap the listing picture** to open that listing on tutti.ch (long-pressing the card does the same).
- **Pause** (the ❚❚ button next to ☰) stops everything: the camera (the last frame stays on screen), the
  models, JEV (cut off mid-question) and every tutti search and download. ▶ carries on where it stopped.
  Leaving the app pauses the searches too.
- **Tap a box** to lock the focus on it (tap it again to release). This also saves a full-resolution photo of
  it in `Pictures/RoomInventory/`.
- **Tap the card** to change the category when the app picked the wrong one; the search restarts there.
  **Long-press the card** to open the closest listing on tutti.ch.
- **☰** (top left): every item seen with its guess, price and closest listing (tap the picture to open it),
  the state of the on-device models, the detection threshold, and buttons to clear the tutti cache or
  forget all items.

Items are remembered by appearance between launches. Aim at something you scanned before and its answer is
back at once, and the search carries on from where it stopped.

## How it works

    camera frame -> YOLOE boxes -> box in focus -> names (YOLOE vocabulary)
                 -> JEV picks the tutti categories -> newest listings of those categories
                 -> DINOv2 compares their thumbnails with the object -> product names from the closest titles
                 -> those names are searched in turn -> best name + median price of its closest listings

- **Boxes**: `yoloe-26s-seg-pf` at 640 px on the GPU (LiteRT 2 `CompiledModel`, CPU fallback), exported by
  `tools/export_tflite.py`. About 22 ms per frame on a Galaxy Z Fold4. A small IoU tracker keeps boxes steady.
- **Names**: the model also outputs YOLOE's class embedding for every anchor. For the box in focus the app
  scores its own vocabulary of product types (`tools/product_vocab.txt`, 328 names, turned into YOLOE rows by
  its MobileCLIP2 text encoder) and keeps the best names over several frames (`YoloeVocab.kt`, `ItemMemory.kt`).
- **Category**: JEV, a decision model (Together AI's Tev1-0.8B, a Qwen3.5 fine-tune), runs on the phone's
  CPU through llama.cpp (`src/main/cpp`). It answers two multiple-choice questions, the group (15 options) and
  then the category inside the likeliest groups (63 tutti categories, `assets/jev_categories.json`). The
  probabilities are the softmax of the option letters' logits after one forward pass, exactly as in
  `jev_classifier.py`. About 15 to 30 s per new set of names on a Fold4; the search starts with the first
  group's answer. Answers are cached for good: the same names (in any order) never run JEV twice, and an
  object whose main name was already answered with at least 80% reuses that answer at once. If `assets/category_refs.bin` exists (`tools/build_category_refs.py`), a visual check against
  example listings of every category is mixed in, which catches objects the camera names wrongly.
- **Matching**: DINOv2-small (`tools/export_embedder.py`) turns the object and every listing thumbnail into
  a 384-number descriptor; similarity is their dot product. It runs on the GPU, about 20 ms per picture on a
  Fold4 (90 ms on its CPU); at start-up the app checks the GPU against the CPU and falls back to the CPU if
  they disagree. The search (`FinderCore.kt`, a port of
  `find_product.py`) browses the newest listings of the likeliest categories, extracts product names from the
  titles of the closest ones and searches them, each search sharpening the next. A name wins on the mean
  similarity of its 5 closest listings. The price is the median of the 10 closest listings of the winning
  search. Free listings ("Gratis", "Regalo"...) count neither as a product name nor as a price.
- **Caches** (in the app's private storage): `tutti.db` (listings, thumbnail descriptors, searches for 6
  hours), `jev_cache.json` (JEV answers per set of names), `items.json` (items, their views and answers),
  and the pictures of the closest listings in the cache folder.
  Descriptors are tied to the embedder file, so they are recomputed after a model change.

## Build

The models are not in git. Export them once (Linux or macOS for the LiteRT ones: LiteRT has no Windows
wheels), then build:

    pip install "ultralytics>=8.4" litert-torch ai-edge-litert ai-edge-quantizer transformers git+https://github.com/ultralytics/CLIP.git
    python tools/export_tflite.py --check      # yoloe.tflite (38 MB), yoloe_vocab.bin, yoloe_names.txt
    python tools/export_embedder.py --check    # embedder.tflite (87 MB)
    python tools/export_jev.py --check         # jev.gguf (0.8 GB), see the script for its venv
    python tools/build_category_refs.py        # optional: category_refs.bin (3 MB)
    cd android
    gradlew assembleDebug                      # or open the android/ folder in Android Studio
    adb install -r app/build/outputs/apk/debug/app-debug.apk

The app includes llama.cpp, built from source with the NDK (28.2) and CMake (3.22.1), both installed from
Android Studio's SDK manager. CMake downloads llama.cpp at the tag in `src/main/cpp/CMakeLists.txt` unless
you point it at a checkout (`tools/export_jev.py` makes one):

    gradlew assembleDebug -PllamaSource=C:/path/to/repo/tools/.work/llama.cpp

The full APK is about 1 GB, mostly JEV. On first launch the app unpacks JEV to its storage (a few seconds).
For quick reinstalls while developing, `-PnoJev` builds an APK without it; the phone keeps the copy it
unpacked from an earlier full install:

    gradlew installDebug -PnoJev

## Try another model without rebuilding

Files pushed to the app's folder win over the bundled ones:

    adb push yoloe.tflite /sdcard/Android/data/ch.hevs.roominventory/files/
    adb push jev.gguf /sdcard/Android/data/ch.hevs.roominventory/files/
    adb shell rm /sdcard/Android/data/ch.hevs.roominventory/files/yoloe.tflite   # back to the bundled one

## Debugging

    adb logcat -s RoomInventory

shows the backends chosen, JEV's timings and answers, and the search's progress.

The embedder logs its check at start-up (`embedder on GPU FP16: 19.7 ms, agrees with the CPU 0.9900...`).
The stock DINOv2 graph gave nearly the same descriptor for every picture on the Fold4's GPU: the token
layout ([1, tokens, channels] with fully connected layers) was the problem, not FP16. `tools/dinov2_gpu.py`
lays the tokens out like an image row with 1x1 convolutions, which the GPU gets right; the export uses it.
To find which layer a GPU gets wrong, export probe models and push them; at start-up the app logs, for every
intermediate output, how close the GPU is to the CPU:

    python tools/export_embedder_probe.py --weights path/to/dinov2-small --layers 2 --out /tmp
    adb push /tmp/embedder_probe_conv.tflite /sdcard/Android/data/ch.hevs.roominventory/files/
    adb logcat -s RoomInventory | grep probe

Unit tests (prompt rendering against `tools/jev_prompt.py`, title parsing, the search logic, tutti parsing)
run on the computer: `gradlew testDebugUnitTest`.

Photos of tapped boxes: `adb pull /sdcard/Pictures/RoomInventory/`.
