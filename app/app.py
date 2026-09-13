"""
SignalScope - AI-Generated Image Detection Application
Streamlit Interface for ResNet-50 Model Inference & Diagnostics
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root directory is first in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
import streamlit as st
import torch
import torch.nn.functional as F

# Custom imports from app package
try:
    from app.config import (
        BENCHMARK_METRICS,
        DISCLAIMER_TEXT,
        CONFIDENCE_DISCLAIMER,
        DEFAULT_INFERENCE_MODE,
        PATCH_N,
        PATCH_AGGREGATION_DEFAULT,
        PATCH_AGGREGATION_METHODS,
        INFERENCE_MODES
    )
    from app.model_loader import load_model
    from app.predictor import (
        predict_image,
        predict_image_patch_vote,
        predict_image_tta,
        predict_image_hybrid,
        predict_image_auto,
        predict_batch,
        preprocess_image
    )
except (ModuleNotFoundError, ImportError):
    from config import (
        BENCHMARK_METRICS,
        DISCLAIMER_TEXT,
        CONFIDENCE_DISCLAIMER,
        DEFAULT_INFERENCE_MODE,
        PATCH_N,
        PATCH_AGGREGATION_DEFAULT,
        PATCH_AGGREGATION_METHODS,
        INFERENCE_MODES
    )
    from model_loader import load_model
    from predictor import (
        predict_image,
        predict_image_patch_vote,
        predict_image_tta,
        predict_image_hybrid,
        predict_image_auto,
        predict_batch,
        preprocess_image
    )

# Page configuration
st.set_page_config(
    page_title="SignalScope - AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 0;
    }
    
    .result-box-fake {
        background-color: #450a0a;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.5rem;
        color: #fecdd3;
        text-align: center;
        margin-top: 1rem;
    }
    
    .result-box-real {
        background-color: #064e3b;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.5rem;
        color: #a7f3d0;
        text-align: center;
        margin-top: 1rem;
    }
    
    .result-box-warning {
        background-color: #451a03;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        padding: 1.5rem;
        color: #fef3c7;
        text-align: center;
        margin-top: 1rem;
    }
    
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .result-subtext {
        font-size: 0.95rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Application Header
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🔍 SignalScope</div>
        <div class="header-subtitle">AI-Generated Image Detection & Diagnostics System</div>
        <p style="margin-top: 0.8rem; color: #cbd5e1; font-size: 0.95rem;">
            Screen images using the trained ResNet-50 model with multi-strategy inference (Native Patch Voting, Hybrid, TTA) 
            to classify whether an image is <strong>AI-Generated (FAKE)</strong> or an <strong>Authentic Photograph (REAL)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Model Setup & Inference Controls
    with st.sidebar:
        st.header("⚡ System Status")
        
        try:
            model, device = load_model()
            device_label = "🔥 GPU (CUDA)" if device.type == "cuda" else "💻 CPU"
            st.success(f"ResNet-50 Active ({device_label})")
        except Exception as err:
            st.error("Failed to load model checkpoint.")
            st.error(f"Details: {str(err)}")
            st.stop()

        st.markdown("---")
        st.header("⚙️ Inference Controls")
        
        from app.strategies.strategy_registry import list_strategies
        registered_strats = list_strategies()
        strategy_options = [s["key"] for s in registered_strats]
        strategy_labels = {s["key"]: s["display_name"] for s in registered_strats}


        selected_mode_key = st.selectbox(
            "Inference Strategy",
            options=strategy_options,
            index=strategy_options.index(DEFAULT_INFERENCE_MODE) if DEFAULT_INFERENCE_MODE in strategy_options else 0,
            format_func=lambda x: strategy_labels.get(x, x),
            help="Select how the image is presented to the trained model."
        ) or "auto"



        with st.expander("🛠️ Advanced Settings", expanded=False):
            patch_n_val = st.slider(
                "Number of Patches (Patch/Hybrid)",
                min_value=8,
                max_value=64,
                value=PATCH_N,
                step=4,
                help="Number of native 32x32 crops extracted across the image."
            )
            
            aggregation_val = st.selectbox(
                "Patch Aggregation",
                options=PATCH_AGGREGATION_METHODS,
                index=PATCH_AGGREGATION_METHODS.index(PATCH_AGGREGATION_DEFAULT),
                help="Strategy to aggregate patch-level predictions."
            )
            
            seed_val = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                help="Ensures deterministic patch crop locations."
            )

        st.markdown("---")
        st.subheader("📊 CIFAKE Test Benchmarks")
        for key, val in BENCHMARK_METRICS.items():
            st.markdown(f"**{key}:** `{val}`")

        st.markdown("---")
        st.info(CONFIDENCE_DISCLAIMER)

    # Navigation Tabs
    tab_single, tab_batch, tab_about = st.tabs([
        "🖼️ Single Image Analysis", 
        "📁 Batch Processing", 
        "ℹ️ Technical Details & Benchmark"
    ])

    # ------------------------------------------------------------------
    # TAB 1: Single Image Inference
    # ------------------------------------------------------------------
    with tab_single:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a JPG, JPEG, or PNG image",
                type=["jpg", "jpeg", "png"],
                help="Upload an image to run live model inference."
            )
            
            caption_input = st.text_area(
                "Optional Caption / Claim",
                help="If the image has a caption or claim (e.g. 'Handmade ceramic mug'), enter it here to test Multimodal Image+Text consistency."
            )

            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    raw_bytes = uploaded_file.read()
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    image.load()
                except Exception as img_err:
                    st.error(f"Unable to read uploaded image file: {str(img_err)}")
                    image = None
                    raw_bytes = b""

                if image is not None:
                    width, height = image.size
                    
                    # Stage 1: Pre-Screening Metadata & C2PA Provenance Inspection
                    from app.diagnostics.metadata_inspector import inspect_image_metadata
                    pre_meta = inspect_image_metadata(image, raw_bytes=raw_bytes, filename=uploaded_file.name)

                    
                    if pre_meta["provenance_verdict"] == "AI_GENERATED":
                        st.error(f"🤖 **Stage 1 Metadata Pre-Screening:** {pre_meta['status_message']}")
                    elif pre_meta["provenance_verdict"] == "CAMERA_REAL":
                        st.success(f"📸 **Stage 1 Metadata Pre-Screening:** {pre_meta['status_message']}")
                    else:
                        st.info("📜 **Stage 1 Metadata Pre-Screening:** No C2PA or AI metadata tags detected → Passing image to ResNet-50 PyTorch Pipeline.")

                    # Create 32x32 model input representation using bicubic interpolation matching PyTorch pipeline
                    from app.strategies.patch.patch_extractor import prepare_image
                    clean_image_input = prepare_image(image)
                    image_32 = clean_image_input.resize((32, 32), Image.Resampling.BICUBIC)

                    v_tab1, v_tab2 = st.tabs(["🖼️ Original Preview", "🔬 ResNet-50 Input (32×32 px)"])
                    with v_tab1:
                        st.image(image, caption=f"Original High-Res Preview: {uploaded_file.name}", width="stretch")
                        st.caption(f"**Filename:** `{uploaded_file.name}` | **Resolution:** {width} × {height} px | **Mode:** {image.mode}")
                    with v_tab2:
                        st.image(image_32, caption="ResNet-50 Stem Input (Exact 32×32 px Bicubic Downscale)", width="stretch")
                        st.info("💡 **Neural Network Perspective:** This 32×32 pixel image is the exact bicubic downscaled input fed into the baseline ResNet-50 model stem. Notice how fine pixel textures are compressed.")

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔎 Analyze Image", type="primary", width="stretch"):
                        st.session_state.analyze_clicked = True
                else:
                    st.session_state.analyze_clicked = False
            else:
                st.info("Upload an image on the left to begin analysis.")
                st.session_state.analyze_clicked = False

        with col_output:
            st.subheader("🎯 Inference & Diagnostic Output")

            if uploaded_file is not None and st.session_state.get("analyze_clicked", False):
                mode_str = str(selected_mode_key or "auto").upper()
                with st.spinner(f"Executing PyTorch inference ({mode_str} mode)..."):
                    start_t = time.time()
                    from app.predictor import predict_image_auto

                    res = predict_image_auto(
                        image,
                        model=model,
                        device=device,
                        mode=selected_mode_key or "auto",
                        n_patches=patch_n_val,
                        seed=int(seed_val),
                        aggregation=aggregation_val,
                        precomputed_metadata=pre_meta
                    )
                    elapsed_ms = (time.time() - start_t) * 1000

                # Attach metadata diagnostic if present
                if "metadata_diagnostic" not in res:
                    res["metadata_diagnostic"] = pre_meta


                label = res["label"]
                conf = res["confidence"]
                fake_prob = res["fake_probability"]
                real_prob = res["real_probability"]
                active_mode = res.get("inference_mode", selected_mode_key)

                # Status Box Rendering
                if active_mode == "metadata_provenance":
                    box_class = "result-box-fake"
                    badge_text = "🤖 AI-GENERATED (C2PA / METADATA VERIFIED)"
                elif conf < 0.70:
                    box_class = "result-box-warning"
                    badge_text = "⚠️ LOW CONFIDENCE - REVIEW RECOMMENDED"
                elif label == "FAKE":
                    box_class = "result-box-fake"
                    badge_text = "🤖 AI-GENERATED (FAKE)"
                else:
                    box_class = "result-box-real"
                    badge_text = "📸 REAL PHOTOGRAPH"

                st.markdown(f"""
                <div class="{box_class}">
                    <div class="result-label">{badge_text}</div>
                    <div class="result-subtext">Confidence Score: {conf * 100:.2f}% | Mode: {active_mode.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

                if active_mode == "metadata_provenance":
                    st.info(f"⚡ **Short-Circuit Notice:** Verified AI digital metadata detected (`{res['agreement']}`). Deep learning neural network forward pass bypassed.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📈 Classification Probabilities")
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Fake Probability", f"{fake_prob * 100:.2f}%")
                    st.progress(fake_prob)
                with col_p2:
                    st.metric("Real Probability", f"{real_prob * 100:.2f}%")
                    st.progress(real_prob)

                # --------------------------------------------------
                # Module B: Generator Attribution
                # --------------------------------------------------
                if label == "FAKE" or fake_prob > 0.5:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🕵️‍♂️ Generator Attribution")
                    with st.spinner("Analyzing artifacts to determine generator family..."):
                        try:
                            from model.generator_attribution import predict_generator_attribution
                            attribution = predict_generator_attribution(image)
                            
                            a_col1, a_col2 = st.columns(2)
                            with a_col1:
                                st.metric("Likely Generator Family", attribution.get("family", "Unknown"))
                            with a_col2:
                                st.metric("Specific Model", attribution.get("specific_model", "Unknown"))
                            
                            st.caption(f"**Attribution Note:** {attribution.get('note', '')}")
                        except ImportError:
                            st.warning("Generator attribution module not found.")


                # Disclaimer
                st.caption("🛡️ Confidence reflects model certainty under the selected strategy, not an absolute guarantee.")

                # --------------------------------------------------
                # Diagnostics Expander
                # --------------------------------------------------
                with st.expander("🔬 Comprehensive Diagnostics & Stability Analysis", expanded=True):
                    d_tab1, d_tab2, d_tab3, d_tab4, d_tab5, d_tab6, d_tab7, d_tab8 = st.tabs([
                        "📊 Output & Entropy", "🧩 Patch Stability", "⚖️ Hybrid Comparison", "🌀 FFT Diagnostic", "📜 C2PA & Metadata", "🎯 Inference Regions", "🧠 AI Explanation", "🛡️ Live Degradation Test"
                    ])

                    # Sub-Tab 1: Output & Entropy
                    with d_tab1:
                        st.markdown(f"**Inference Timing:** `{elapsed_ms:.2f} ms` | **Device:** `{device.type.upper()}` | **Resolution:** `{res.get('image_dimensions')}`")
                        st.markdown(f"**Normalized Shannon Entropy:** `{res.get('normalized_entropy', 0.0):.4f}` (Raw: `{res.get('entropy', 0.0):.4f}`)")
                        st.info(f"**Uncertainty Level:** {res.get('uncertainty_level')}\n\n{res.get('uncertainty_note')}")

                    # Sub-Tab 2: Patch Stability & Binned Histogram
                    with d_tab2:
                        if "stability" in res:
                            stab = res["stability"]
                            s_col1, s_col2, s_col3 = st.columns(3)
                            with s_col1:
                                st.metric("Mean Patch Fake Prob", f"{stab['mean_fake_probability']*100:.1f}%")
                                st.metric("Median Patch Fake Prob", f"{stab['median_fake_probability']*100:.1f}%")
                            with s_col2:
                                st.metric("Std Dev", f"{stab['std_fake_probability']:.4f}")
                                st.metric("Range (Max - Min)", f"{stab['range_fake_probability']:.4f}")
                            with s_col3:
                                st.metric("Patch Agreement", f"{stab['patch_agreement_pct']:.1f}%")
                                st.metric("Votes (Fake / Real)", f"{stab['fake_patch_count']} / {stab['real_patch_count']}")

                            # Binned Probability Histogram Visualization
                            st.markdown("---")
                            st.subheader("📊 Patch Probability Distribution Histogram")
                            patch_probs = res.get("patch_fake_probs", [])
                            if patch_probs:
                                bins = np.linspace(0.0, 1.0, 11)
                                counts, _ = np.histogram(patch_probs, bins=bins)
                                bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(10)]
                                df_hist = pd.DataFrame({"Patch Count": counts}, index=bin_labels)
                                st.bar_chart(df_hist)
                        else:
                            st.info("Patch stability diagnostics are active when in Native Patch or Hybrid inference modes.")

                    # Sub-Tab 3: Hybrid Comparison
                    with d_tab3:
                        if "resize_prediction" in res and "patch_prediction" in res:
                            h_resize = res["resize_prediction"]
                            h_patch = res["patch_prediction"]
                            h_top_k = res.get("top_k_patch_prediction", {
                                "label": "FAKE" if h_patch.get("top_k_patch_fake_prob", 0) >= 0.5 else "REAL",
                                "fake_probability": h_patch.get("top_k_patch_fake_prob", h_patch["fake_probability"]),
                                "real_probability": 1.0 - h_patch.get("top_k_patch_fake_prob", h_patch["fake_probability"])
                            })
                            
                            st.markdown(f"**Hybrid Status:** `{res.get('agreement')}` | **Prob Difference (Resize vs Patch):** `{res.get('prediction_difference', 0.0):.4f}`")
                            
                            comp_data = {
                                "Strategy": [
                                    "Baseline Resize (32x32)",
                                    "Native Patch Voting (Mean)",
                                    "Native Patch Voting (Top-K Artifacts)",
                                    "🎯 Final Hybrid Consensus"
                                ],
                                "Predicted Label": [
                                    h_resize["label"],
                                    h_patch["label"],
                                    h_top_k["label"],
                                    res["label"]
                                ],
                                "Fake Probability": [
                                    f"{h_resize['fake_probability']*100:.2f}%",
                                    f"{h_patch['fake_probability']*100:.2f}%",
                                    f"{h_top_k['fake_probability']*100:.2f}%",
                                    f"{res['fake_probability']*100:.2f}%"
                                ],
                                "Real Probability": [
                                    f"{h_resize['real_probability']*100:.2f}%",
                                    f"{h_patch['real_probability']*100:.2f}%",
                                    f"{h_top_k['real_probability']*100:.2f}%",
                                    f"{res['real_probability']*100:.2f}%"
                                ]
                            }
                            st.dataframe(pd.DataFrame(comp_data), width="stretch")
                            st.caption(
                                "💡 **Understanding Hybrid Consensus:** High-resolution AI images often contain smooth background regions "
                                "(sky, plain walls) alongside localized AI artifacts in detailed areas. The Hybrid Consensus engine evaluates both mean "
                                "and peak localized patch activations to prevent false-negative classifications when baseline downscaling obscures artifacts."
                            )
                        else:
                            st.info("Hybrid comparison is available when running in 'Hybrid' mode.")

                    # Sub-Tab 4: Experimental FFT Diagnostic
                    with d_tab4:
                        fft_data = res.get("fft_diagnostic", {})
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            st.metric("Spectral Irregularity Score", f"{fft_data.get('spectral_score', 0.0):.4f}")
                            st.write(f"**Diagnostic Label:** `{fft_data.get('diagnostic_label')}`")
                        with f_col2:
                            st.metric("High/Low Energy Ratio", f"{fft_data.get('high_to_low_ratio', 0.0):.4f}")
                            st.metric("Spectral Slope", f"{fft_data.get('spectral_slope', 0.0):.4f}")

                        st.warning(f"⚠️ **Experimental Diagnostic Note:** {fft_data.get('interpretation')}")

                    # Sub-Tab 5: Stage 1 Metadata & C2PA Provenance
                    with d_tab5:
                        meta_data = res.get("metadata_diagnostic", {})
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.write(f"**Provenance Verdict:** `{meta_data.get('provenance_verdict')}`")
                            st.write(f"**Source Identified:** `{meta_data.get('source_identified') or 'None'}`")
                        with m_col2:
                            st.write(f"**C2PA Manifest Header:** `{'Detected' if meta_data.get('c2pa_manifest_detected') else 'Not Detected'}`")
                            st.write(f"**Camera Hardware:** `{meta_data.get('camera_matched') or 'None Detected'}`")

                        st.markdown("---")
                        st.write("📜 **Extracted EXIF / PNG Header Summary:**")
                        m_summary = meta_data.get("metadata_summary", {})
                        if m_summary:
                            for key, val in m_summary.items():
                                st.markdown(f"- **{key}**: `{val}`")
                        else:
                            st.info("No EXIF or PNG metadata headers detected in file (metadata unpopulated or stripped).")

                    # Sub-Tab 6: High-Scoring Inference Regions (Bounding Box Overlay)
                    with d_tab6:
                        st.caption("🎯 **Model Inference Highlights:** Draws bounding boxes around patches with high AI-fake scores.")
                        analysis_data = res.get("analysis", {})
                        regions = analysis_data.get("highlighted_regions", [])

                        if not regions and "patch_prediction" in res:
                            patch_p = res["patch_prediction"]
                            coords_list = patch_p.get("patch_coordinates", [])
                            probs_list = patch_p.get("patch_fake_probs", [])
                            regions = [
                                {"x": c[0], "y": c[1], "width": c[2]-c[0], "height": c[3]-c[1], "fake_probability": p, "source": "patch_vote"}
                                for c, p in zip(coords_list, probs_list) if p >= 0.50
                            ]

                        if regions:
                            from app.diagnostics.bounding_box import render_highlighted_regions
                            boxed_image = render_highlighted_regions(image, regions)
                            
                            st.write("📋 **Highlighted Region Data:**")
                            st.dataframe(pd.DataFrame(regions), width="stretch")
                            
                            st.markdown("---")
                            st.caption("🔥 **ResNet-50 Grad-CAM Heatmap:** Visualizes network activation hotspots for the FAKE class.")
                            from app.diagnostics.grad_cam import run_grad_cam
                            with st.spinner("Generating Grad-CAM..."):
                                try:
                                    cam_image = run_grad_cam(model, image, target_class=0)
                                except Exception as e:
                                    st.error(f"Grad-CAM generation failed: {e}")
                                    cam_image = None
                            
                            col_b1, col_b2 = st.columns(2)
                            with col_b1:
                                st.image(boxed_image, caption="Bounding Box Localization", use_container_width=True)
                            with col_b2:
                                if cam_image:
                                    st.image(cam_image, caption="Grad-CAM Activation", use_container_width=True)
                        else:
                            st.success("✅ No localized suspicious AI patch regions detected above 50% fake threshold.")

                    # Sub-Tab 7: Faithful Explanation (Gemini API)
                    with d_tab7:
                        st.subheader("🤖 Faithful Explanation (Gemini Vision)")
                        st.write("Generating a human-readable explanation for the visual cues behind the verdict...")
                        
                        from app.diagnostics.explainer import generate_faithful_explanation
                        with st.spinner("Analyzing visual cues and multimodal consistency..."):
                            explanation_data = generate_faithful_explanation(
                                image=image,
                                prediction_label=label,
                                regions=regions,
                                caption=caption_input if caption_input else None,
                                diagnostic_context=res
                            )
                        
                        st.markdown(f"**Explanation:**\n> {explanation_data.get('explanation')}")
                        
                        if caption_input:
                            st.markdown("---")
                            st.write("📝 **Multimodal Image-Text Consistency**")
                            c_score = explanation_data.get('consistency_score')
                            if c_score is not None:
                                st.metric("Consistency Score (0 to 1)", f"{c_score:.2f}")
                            st.write(f"**Note:** {explanation_data.get('consistency_note')}")
                            
                    # Sub-Tab 8: Live Degradation Test
                    with d_tab8:
                        st.subheader("🛡️ Live Robustness Check")
                        st.write("Test if the current verdict holds up against severe JPEG compression (Quality: 30) on-the-fly.")
                        
                        if st.button("Run Live JPEG Compression Test"):
                            from io import BytesIO
                            with st.spinner("Compressing and re-evaluating..."):
                                buf = BytesIO()
                                image.convert("RGB").save(buf, format="JPEG", quality=30)
                                buf.seek(0)
                                comp_img = Image.open(buf).convert("RGB")
                                comp_res = predict_image_auto(
                                    comp_img,
                                    model=model,
                                    device=device,
                                    mode=selected_mode_key or "auto",
                                    n_patches=patch_n_val,
                                    seed=int(seed_val),
                                    aggregation=aggregation_val
                                )
                                comp_prob = comp_res.get("fake_probability", 0.0)
                                orig_prob = res.get("fake_probability", 0.0)
                                
                                st.markdown("### Robustness Results")
                                r_col1, r_col2 = st.columns(2)
                                with r_col1:
                                    st.metric("Original Fake Prob", f"{orig_prob*100:.2f}%")
                                with r_col2:
                                    delta = (comp_prob - orig_prob) * 100
                                    st.metric("Compressed Fake Prob", f"{comp_prob*100:.2f}%", delta=f"{delta:.2f}%")
                                
                                if (orig_prob >= 0.5 and comp_prob >= 0.5) or (orig_prob < 0.5 and comp_prob < 0.5):
                                    st.success("✅ Verdict is stable under severe degradation.")
                                else:
                                    st.error("❌ Verdict flipped under degradation. Model is sensitive.")


            elif uploaded_file is not None:
                st.info("Click **Analyze Image** above to run the PyTorch inference & diagnostics engine.")

    # ------------------------------------------------------------------
    # TAB 2: Batch Processing
    # ------------------------------------------------------------------
    with tab_batch:
        st.subheader("📁 Batch Image Analysis")
        st.write("Upload multiple images to execute bulk inference and export results as CSV.")

        batch_files = st.file_uploader(
            "Upload multiple images (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if batch_files:
            st.write(f"Total images uploaded: **{len(batch_files)}**")
            
            if st.button("🚀 Process Batch Predictions", type="primary"):
                images_dict = {}
                for bf in batch_files:
                    try:
                        bf.seek(0)
                        img = Image.open(bf)
                        img.load()
                        images_dict[bf.name] = img
                    except Exception:
                        pass
                
                with st.spinner(f"Running inference on {len(images_dict)} images using '{selected_mode_key.upper()}' strategy..."):
                    df_results = predict_batch(
                        images_dict,
                        model=model,
                        device=device,
                        mode=selected_mode_key,
                        n_patches=patch_n_val,
                        aggregation=aggregation_val
                    )

                st.success("Batch processing complete!")
                display_df = df_results.drop(columns=["Raw Confidence"], errors="ignore")
                st.dataframe(display_df, width="stretch")

                # CSV Download
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Results CSV",
                    data=csv_data,
                    file_name="signalscope_batch_predictions.csv",
                    mime="text/csv"
                )

    # ------------------------------------------------------------------
    # TAB 3: Technical Details & Benchmark Output
    # ------------------------------------------------------------------
    with tab_about:
        st.subheader("🔬 Architecture & Multi-Strategy Inference Overview")
        
        st.markdown("""
        ### SignalScope Pipeline Capabilities
        SignalScope uses a fine-tuned **ResNet-50** neural network trained on the **CIFAKE** dataset.
        
        - **Supported Inference Strategies:**
          - **Resize (Baseline):** Resizes full image to $32 \\times 32$ bicubic (fast, baseline).
          - **Native Patch Voting:** Extracts $N$ native $32 \\times 32$ crops from high-res images to preserve pixel-level texture.
          - **Hybrid:** Combines Resize and Patch predictions, calculating strategy agreement.
          - **Test-Time Augmentation (TTA):** Evaluates predictions across multiple geometric transformations.
        - **Normalization:** ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
        - **Diagnostics:** Normalized Shannon Entropy $H(p)$, Patch Disagreement Variance, and Experimental 2D FFT Spectral Analysis.
        """)

        st.markdown("---")
        st.subheader("📈 Benchmark Performance Metrics")

        m_keys = list(BENCHMARK_METRICS.keys())
        if len(m_keys) >= 3:
            cols = st.columns(min(len(m_keys), 3))
            for i, (k, v) in enumerate(BENCHMARK_METRICS.items()):
                col_idx = i % len(cols)
                cols[col_idx].markdown(f"**{k}:** `{v}`")
        else:
            for k, v in BENCHMARK_METRICS.items():
                st.markdown(f"**{k}:** `{v}`")


        outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
        if outputs_dir.exists():
            st.markdown("---")
            st.subheader("🖼️ Benchmark Visualizations")
            
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                cm_img = outputs_dir / "confusion_matrix.png"
                if cm_img.exists():
                    st.image(str(cm_img), caption="Confusion Matrix (CIFAKE Test Set)")
                
                roc_img = outputs_dir / "roc_curve.png"
                if roc_img.exists():
                    st.image(str(roc_img), caption="ROC Curve")

            with p_col2:
                acc_img = outputs_dir / "training_validation_accuracy.png"
                if acc_img.exists():
                    st.image(str(acc_img), caption="Training & Validation Accuracy")
                    
                pr_img = outputs_dir / "precision_recall_curve.png"
                if pr_img.exists():
                    st.image(str(pr_img), caption="Precision-Recall Curve")

    # Global Footer Disclaimer
    st.markdown("---")
    st.caption(f"🛡️ **Disclaimer:** {DISCLAIMER_TEXT}")


if __name__ == "__main__":
    main()

