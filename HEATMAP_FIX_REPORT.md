# Grad-CAM Heatmap Fix Report

## Update: follow-up pass (same day)

After the initial visual fix and push, went back through `grad_cam.py` and its
`app.py` wiring end-to-end looking for anything else related to the heatmap. Found
and fixed three more bugs — none affected the pixels of a single, isolated heatmap
render (why they weren't visible in the first pass's before/after screenshot), but
all three affect correctness or usability in the way the app actually runs. See
"Follow-up fixes" below for details; the original writeup follows unchanged.

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

---

## Follow-up fixes (second pass)

Asked to keep looking for anything else wrong with the heatmap. These three don't
show up in a single before/after screenshot — they only surface once the app is
actually used the way Streamlit apps get used (multiple reruns per session, more than
one concurrent user) — but they're real bugs in the restored feature.

### 5. Grad-CAM recomputed on every Streamlit rerun, not just on a new image
`app.py` called `run_grad_cam()` directly inside the `d_tab_verdict` render block, with
no caching. Streamlit reruns the **entire script top to bottom** on any widget
interaction anywhere on the page — clicking "Run Live JPEG Compression Test" in a
different tab, adjusting a slider, anything. Every one of those unrelated reruns was
silently regenerating the same heatmap from scratch (forward pass + backward pass +
resize + blur + colormap), each time flashing the "Generating Grad-CAM..." spinner
next to a result that hadn't changed. Every other per-image diagnostic in this same
tab (`explanation_data`, `attribution`, `_regions`) was already cached in
`st.session_state` and gated behind the `is_fresh_run` check computed earlier in
`app.py` — Grad-CAM was the one diagnostic that didn't follow that pattern.

**Fix**: `run_grad_cam()` now runs once inside the existing `if is_fresh_run:` block,
right alongside the explanation/attribution calls, and is cached to
`st.session_state["cached_cam_image"]`. The render block just displays the cached
image (with a manual "🔄 Retry Grad-CAM" button on failure, matching the retry buttons
already used for the Gemini explanation and attribution calls). Verified by triggering
an unrelated rerun (the JPEG compression test button, which doesn't change the
analysis cache key) and confirming the cached heatmap re-renders instantly with no
spinner instead of recomputing.

### 6. Gradient memory left dangling on the shared cached model
`run_grad_cam()` temporarily flips `requires_grad=True` on every model parameter and
runs `target.backward()` to get Grad-CAM's gradients, then restores the original
`requires_grad` flags — but never cleared the `.grad` tensors that `backward()` leaves
behind. Since `app/model_loader.py` caches the model process-wide via
`@st.cache_resource`, those full-size gradient tensors (roughly the same footprint as
the model's own ~90MB of weights) stayed pinned in memory for the entire lifetime of
the cached model after the *first* time anyone opened the Grad-CAM panel, not just
transiently during the computation.

**Fix**: added `model.zero_grad(set_to_none=True)` at the end of `run_grad_cam()`'s
`finally` block, so the gradients are released immediately after use instead of
sitting there until some later, unrelated Grad-CAM call happens to overwrite them.
Verified with a direct check that `any(p.grad is not None for p in model.parameters())`
is `False` both before and after a `run_grad_cam()` call.

### 7. No thread-safety around mutating the shared cached model
This is the one that can actually produce a **wrong** heatmap, not just a slow one.
`@st.cache_resource` shares one model instance across every user session a Streamlit
server is handling. `run_grad_cam()` mutates that shared object — toggling
`requires_grad` on every parameter and running `backward()`, which **accumulates**
gradients into `.grad` — with no locking. If two Grad-CAM requests land close together
(two browser tabs, two people using the same deployment at once, or even two quick
reruns racing), their `backward()` calls interleave on the same parameters and each
one's gradients get mixed into the other's, corrupting both resulting heatmaps. That
failure mode is a textbook match for "the math is right in isolation, the picture
comes out wrong" — it just needed more than one request in flight to trigger.

**Fix**: added a module-level `threading.Lock()` around the hook-registration →
forward → backward → hook-removal critical section in `run_grad_cam()`, the same
pattern already used in `app/api/gemini_gateway.py` for the same class of shared-state
problem. Verified by hammering `run_grad_cam()` from 4 concurrent threads against the
same shared model (simulating concurrent users) — zero errors, and confirmed no
leftover gradients afterward either.

### Additional files changed

| File | Change |
| :--- | :--- |
| `app/diagnostics/grad_cam.py` | Added a `threading.Lock` around the shared-model mutation, and `model.zero_grad(set_to_none=True)` after use. |
| `app/app.py` | Grad-CAM now computed once per fresh analysis (inside the existing `is_fresh_run` block) and cached in `st.session_state`, instead of recomputing on every rerun; added a retry button on failure. |

### Additional verification

- `pytest tests/ -q` — still 39/39 passing.
- Direct check: `any(p.grad is not None for p in model.parameters())` is `False` both
  immediately before and immediately after a `run_grad_cam()` call.
- 4 concurrent threads each calling `run_grad_cam()` 3× against the same shared model
  object: zero exceptions, no leftover gradients.
- Browser test: uploaded an image, ran analysis, confirmed the heatmap renders; then
  triggered an unrelated rerun (the Live Robustness Check button, which doesn't touch
  the analysis cache key) and confirmed the cached heatmap reappears instantly with no
  "Generating Grad-CAM..." spinner.
