# Grad-CAM Heatmap Fix Report

## Symptom

User-reported: right after uploading an image and the inference result appearing, the
Grad-CAM heatmap was "mathematically correct, but the appearance is not" — i.e. the
underlying activation computation was sound, but the rendered overlay looked wrong.

## What was actually there

The Grad-CAM heatmap (`app/diagnostics/grad_cam.py`) and a companion bounding-box
overlay (`app/diagnostics/bounding_box.py`) had been **deleted entirely** in commit
`c36878a` ("remove visual evidence feature (bounding boxes and grad-cam)"), one commit
before the most recent UI overhaul. So the feature wasn't buggy in the running app —
it was gone. Before removing it, the code even shipped with a caption admitting the
problem: *"Note: Heatmap resolution is coarse due to 32x32 ResNet stem input."*

This report restores the feature (`app/diagnostics/grad_cam.py`, wired back into the
"Plain-English Verdict" tab in `app/app.py`, same location as before) and fixes the
three concrete bugs that caused the bad appearance, instead of removing it again.

## Root causes and fixes

### 1. Heatmap always explained "FAKE", never the actual verdict
`app/app.py` called `run_grad_cam(model, image, target_class=0)` — hardcoded to class 0
(FAKE) on every single request, regardless of what the model actually predicted. When
an image was classified REAL, the panel still rendered "why is this FAKE" activations,
which don't correspond to the badge shown one screen up. That's not a rendering bug,
but it is exactly the kind of "the math is right, the picture is wrong" mismatch a user
would report.

**Fix**: `run_grad_cam(model, image, target_class=None)` now defaults to `None`, which
`GradCAM.generate_heatmap()` already supported — it explains whatever class the
forward pass itself predicts. The heatmap is now always self-consistent with the
verdict it's rendered next to.

### 2. Blocky, pixelated overlay from naive upsampling of a tiny activation grid
Grad-CAM reads activations off a conv layer, which is tiny for a model whose stem is
built for 32×32 CIFAR images. The old code used `model.layer3[-1]` (an **8×8** grid)
and upsampled it to the full uploaded-image resolution (which can be 1000px+) with
OpenCV's default resize interpolation. Stretching an 8×8 grid ~100–200× with basic
interpolation produces visible hard-edged squares — a "blocky" heatmap that looks wrong
even though every value in it is a correct Grad-CAM weight.

**Fix**, in `app/diagnostics/grad_cam.py`:
- Target layer switched from `layer3` (8×8) to `layer2` (**16×16**) — verified via a
  live forward pass against the trained checkpoint (`(16, 16)` vs `(8, 8)`), doubling
  linear resolution while still being deep enough in the network to carry meaningful
  class-discriminative signal. A fallback chain (`layer2 → layer3 → layer4 → layer1`)
  keeps this robust if the model architecture ever changes.
- Resize interpolation switched from OpenCV's default to `cv2.INTER_CUBIC` for a
  smooth upsample instead of a blocky one.
- A `cv2.GaussianBlur`, sized relative to one cell of the original coarse grid, softens
  the residual grid pattern cubic interpolation alone leaves behind at large upscale
  factors.

### 3. Jet colormap creates false "hotspots"
The overlay used OpenCV's classic `COLORMAP_JET`. Jet's luminance ramp is not
perceptually uniform — it has a well-documented failure mode (used across the
visualization community as the canonical "why does my heatmap look wrong" example)
where the sharp red/yellow transition reads as a false hotspot independent of the
actual underlying value, and the color ordering doesn't track magnitude intuitively.

**Fix**: default colormap switched to `cv2.COLORMAP_TURBO`, a drop-in OpenCV colormap
designed specifically to fix Jet's perceptual-uniformity problems while remaining just
as visually vivid.

### 4. Fragile manual alpha blend
`heatmap * 0.4 + org_im_cv * 0.6` relied on the two weights summing to 1 to implicitly
stay in the `0–255` range before the final `np.uint8()` cast — correct today, but a
silent overflow risk for any future edit to the blend weights.

**Fix**: replaced with `cv2.addWeighted(...)`, which saturates/clips to `[0, 255]`
explicitly instead of depending on the weights summing to 1.

## Files changed

| File | Change |
| :--- | :--- |
| `app/diagnostics/grad_cam.py` | Restored (was deleted); target layer, interpolation, blur, colormap, and blend fixed as above; `target_class` defaults to auto-detected predicted class. |
| `app/app.py` | Restored the Grad-CAM block in the "Plain-English Verdict" tab; call site no longer forces `target_class=0`; caption updated to reflect it explains the actual predicted class. |

`bounding_box.py` (the other half of the deleted "Visual Evidence" feature) was **not**
restored — out of scope, the user only asked about the heatmap.

## Verification

- `python -m py_compile` on both changed files — no syntax errors.
- `pytest tests/ -q` — all 39 existing tests still pass (no regression).
- Live smoke test against the real trained checkpoint: confirmed `layer2` yields a
  `(16, 16)` activation grid vs `layer3`'s `(8, 8)`, and `run_grad_cam()` returns a
  valid overlay image end-to-end.
- Ran the Streamlit app, uploaded a test image, and visually confirmed the rendered
  heatmap is smooth (no blocky grid squares) and uses the Turbo colormap.
