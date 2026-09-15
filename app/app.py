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
        PATCH_AGGREGATION_METHODS
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
        PATCH_AGGREGATION_METHODS
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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Glassmorphism Header */
    .header-card {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        color: #ffffff;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .header-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.15);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* Result Boxes with Glow and Entrance Animation */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-box-fake, .result-box-real, .result-box-warning {
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }

    .result-box-fake::before, .result-box-real::before, .result-box-warning::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: -1;
        opacity: 0.15;
    }
    
    .result-box-fake {
        background: rgba(69, 10, 10, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fecdd3;
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.25);
    }
    .result-box-fake::before { background: radial-gradient(circle at top right, #ef4444, transparent 70%); }
    
    .result-box-real {
        background: rgba(6, 78, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #a7f3d0;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.25);
    }
    .result-box-real::before { background: radial-gradient(circle at top right, #10b981, transparent 70%); }
    
    .result-box-warning {
        background: rgba(69, 26, 3, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fef3c7;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.25);
    }
    .result-box-warning::before { background: radial-gradient(circle at top right, #f59e0b, transparent 70%); }
    
    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }
    
    .result-subtext {
        font-size: 1.05rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Streamlit Button Overrides */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        color: white;
        border: none;
    }

    div[data-testid="stButton"] > button:active {
        transform: translateY(0px);
    }

    /* File uploader hover */
    div[data-testid="stFileUploader"] section {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #38bdf8;
        background: rgba(56, 189, 248, 0.05);
    }
    
    /* Expander UI Fix */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def main():
    if "selected_mode_key" not in st.session_state:
        st.session_state.selected_mode_key = DEFAULT_INFERENCE_MODE
    if "patch_n_val" not in st.session_state:
        st.session_state.patch_n_val = PATCH_N
    if "aggregation_val" not in st.session_state:
        st.session_state.aggregation_val = PATCH_AGGREGATION_DEFAULT
    if "seed_val" not in st.session_state:
        st.session_state.seed_val = 42

    selected_mode_key = st.session_state.selected_mode_key
    patch_n_val = st.session_state.patch_n_val
    aggregation_val = st.session_state.aggregation_val
    seed_val = st.session_state.seed_val

    # Application Header
    st.markdown("""
    <div class="header-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="header-title">🔍 SignalScope</div>
                <div class="header-subtitle">AI-Generated Image Detection & Diagnostics System</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Model Setup & Inference Controls
    with st.sidebar:
        st.header("🧠 Model Information")
        st.markdown("""
        **Architecture**: ResNet-50 (PyTorch)
        
        **Training Dataset**: CIFAKE (120,000 images)
        - **Authentic**: 60,000 real CIFAR-10 photos
        - **AI-Generated**: 60,000 Latent Diffusion images
        
        **Model Strengths**:
        - Highly accurate at detecting diffusion artifacts
        - Extremely robust to compression and resizing
        
        **Model Limitations**:
        - May struggle with older non-diffusion GANs
        - Potential for false positives on heavily-filtered real photographs
        
        **Confusion Matrix Insights**:
        - **High Recall on Fake**: The model rarely misses AI-generated images.
        - **Precision on Real**: It errs on the side of caution when classifying real photos.
        """)
        
        st.markdown("---")
        st.header("⚡ System Status")
        
        try:
            model, device = load_model()
            device_label = "🔥 GPU (CUDA)" if device.type == "cuda" else "💻 CPU"
            st.success(f"ResNet-50 Active ({device_label})")
        except Exception as err:
            st.error("Failed to load model checkpoint.")
            st.error(f"Details: {str(err)}")
            st.stop()

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
                pre_meta = {}
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

                    with st.expander("⚙️ Inference Engine Controls", expanded=False):
                        from app.strategies.strategy_registry import list_strategies
                        registered_strats = list_strategies()
                        strategy_options = [s["key"] for s in registered_strats]
                        strategy_labels = {s["key"]: s["display_name"] for s in registered_strats}
                
                        st.selectbox(
                            "Inference Strategy",
                            options=strategy_options,
                            key="selected_mode_key",
                            format_func=lambda x: strategy_labels.get(x, x),
                            help="Select how the image is presented to the trained model."
                        )
                
                        st.slider(
                            "Number of Patches (Patch/Hybrid)",
                            min_value=8,
                            max_value=64,
                            step=4,
                            key="patch_n_val",
                            help="Number of native 32x32 crops extracted across the image. Higher takes longer but is more accurate."
                        )
                        
                        st.selectbox(
                            "Patch Aggregation",
                            options=PATCH_AGGREGATION_METHODS,
                            key="aggregation_val",
                            help="Strategy to aggregate patch-level predictions. Mean is balanced, Max is highly sensitive."
                        )
                        
                        st.number_input(
                            "Random Seed",
                            min_value=0,
                            max_value=9999,
                            step=1,
                            key="seed_val",
                            help="Ensures deterministic patch crop locations for reproducibility."
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔎 Analyze Image", type="primary", use_container_width=True):
                        st.session_state.analyze_clicked = True
                        st.session_state.force_reanalyze = True
                else:
                    st.session_state.analyze_clicked = False
            else:
                st.info("Upload an image on the left to begin analysis.")
                st.session_state.analyze_clicked = False

        with col_output:
            st.subheader("🎯 Inference & Diagnostic Output")

            if uploaded_file is not None and st.session_state.get("analyze_clicked", False):
                analysis_cache_key = f"{uploaded_file.name}_{uploaded_file.size}_{selected_mode_key}_{patch_n_val}_{seed_val}_{aggregation_val}_{caption_input}"
                is_fresh_run = (st.session_state.get("current_cache_key") != analysis_cache_key) or st.session_state.get("force_reanalyze", False)

                if is_fresh_run:
                    mode_str = str(selected_mode_key or "auto").upper()
                    with st.spinner(f"Executing PyTorch inference ({mode_str} mode)..."):
                        start_t = time.time()
                        res = predict_image_auto(
                            image,
                            model=model,
                            device=device,
                            mode=selected_mode_key or "auto",
                            n_patches=patch_n_val,
                            seed=int(seed_val),
                            aggregation=aggregation_val,
                            precomputed_metadata=pre_meta,
                            raw_bytes=raw_bytes,
                        )
                        elapsed_ms = (time.time() - start_t) * 1000

                    if "metadata_diagnostic" not in res:
                        res["metadata_diagnostic"] = pre_meta

                    # Pre-compute regions for the explainer
                    _analysis_data = res.get("analysis", {})
                    _regions = _analysis_data.get("highlighted_regions", [])
                    if not _regions and "patch_prediction" in res:
                        _patch_p = res["patch_prediction"]
                        _coords_list = _patch_p.get("patch_coordinates", [])
                        _probs_list = _patch_p.get("patch_fake_probs", [])
                        _regions = [
                            {"x": c[0], "y": c[1], "width": c[2]-c[0], "height": c[3]-c[1], "fake_probability": p, "source": "patch_vote"}
                            for c, p in zip(_coords_list, _probs_list) if p >= 0.50
                        ]

                    _gemini_img = image.copy()
                    _gemini_img.thumbnail((512, 512))

                    attribution = {"family": "Unknown", "specific_model": "Unknown", "confidence": 0.0, "note": "N/A"}
                    if res["label"] == "FAKE" or res["fake_probability"] > 0.5:
                        with st.spinner("Analyzing generator artifacts via Gemini Vision..."):
                            try:
                                from model.generator_attribution import predict_generator_attribution
                                attribution = predict_generator_attribution(_gemini_img)
                            except Exception:
                                pass

                    with st.spinner("Generating faithful explanation with Gemini Vision..."):
                        from app.diagnostics.explainer import generate_faithful_explanation
                        try:
                            explanation_data = generate_faithful_explanation(
                                image=_gemini_img,
                                prediction_label=res["label"],
                                regions=_regions,
                                caption=caption_input if caption_input else None,
                                diagnostic_context=res
                            )
                        except Exception:
                            explanation_data = {"explanation": "Gemini explainer encountered an error.", "consistency_score": None, "consistency_note": "N/A"}

                    st.session_state["cached_res"] = res
                    st.session_state["cached_elapsed_ms"] = elapsed_ms
                    st.session_state["cached_attribution"] = attribution
                    st.session_state["cached_explanation"] = explanation_data
                    st.session_state["cached_regions"] = _regions
                    st.session_state["current_cache_key"] = analysis_cache_key
                    st.session_state["force_reanalyze"] = False
                else:
                    res = st.session_state["cached_res"]
                    elapsed_ms = st.session_state["cached_elapsed_ms"]
                    attribution = st.session_state["cached_attribution"]
                    explanation_data = st.session_state["cached_explanation"]
                    _regions = st.session_state.get("cached_regions", [])

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
                    st.info(f"⚡ **Metadata Override Notice:** Verified AI digital metadata detected (`{res['agreement']}`). Deep learning model was executed for analysis, but the final verdict is locked by cryptographic provenance.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📈 Classification Probabilities")
                
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("Fake Probability", f"{fake_prob * 100:.2f}%")
                    st.progress(fake_prob)
                with col_p2:
                    st.metric("Real Probability", f"{real_prob * 100:.2f}%")
                    st.progress(real_prob)

                # When C2PA metadata override is active, also surface the raw ResNet-50
                # model probabilities so users can see what the visual model alone found.
                if active_mode == "metadata_provenance" and "model_fake_probability" in res:
                    model_fake = res["model_fake_probability"]
                    model_real = res["model_real_probability"]
                    st.caption(
                        f"🔬 **ResNet-50 Visual Model Output (pre-override):** "
                        f"Fake `{model_fake * 100:.1f}%` / Real `{model_real * 100:.1f}%` — "
                        f"Final verdict locked by C2PA cryptographic provenance."
                    )

                # --------------------------------------------------
                # Module B: Generator Attribution
                # --------------------------------------------------
                if label == "FAKE" or fake_prob > 0.5:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🕵️‍♂️ Generator Attribution")
                    a_col1, a_col2 = st.columns(2)
                    with a_col1:
                        st.metric("Likely Generator Family", attribution.get("family", "Unknown"))
                    with a_col2:
                        st.metric("Specific Model", attribution.get("specific_model", "Unknown"))
                    
                    st.caption(f"**Attribution Note:** {attribution.get('note', '')}")
                    if "rate limit" in str(attribution.get("note", "")).lower() or "unavailable" in str(attribution.get("note", "")).lower():
                        if st.button("🔄 Retry Generator Attribution", key="retry_attr_btn"):
                            with st.spinner("Retrying generator attribution via Gemini..."):
                                from model.generator_attribution import predict_generator_attribution
                                _g_img = image.copy()
                                _g_img.thumbnail((512, 512))
                                attribution = predict_generator_attribution(_g_img)
                                st.session_state["cached_attribution"] = attribution
                                st.rerun()


                # Disclaimer
                st.caption("🛡️ Confidence reflects model certainty under the selected strategy, not an absolute guarantee.")

                # --------------------------------------------------
                # Diagnostics Expander
                # --------------------------------------------------
                with st.expander("🔬 Comprehensive Diagnostics & Stability Analysis", expanded=True):
                    d_tab_verdict, d_tab_robust, d_tab_expert = st.tabs([
                        "💡 Plain-English Verdict", 
                        "🛡️ Live Robustness Check",
                        "🔬 Deep Diagnostics (For Experts)"
                    ])

                    # Sub-Tab 1: Output & Entropy
                    with d_tab_expert:
                        st.subheader('📊 Output & Entropy')
                        st.markdown(f"**Inference Timing:** `{elapsed_ms:.2f} ms` | **Device:** `{device.type.upper()}` | **Resolution:** `{res.get('image_dimensions')}`")
                        st.markdown(f"**Normalized Shannon Entropy:** `{res.get('normalized_entropy', 0.0):.4f}` (Raw: `{res.get('entropy', 0.0):.4f}`)")
                        st.info(f"**Uncertainty Level:** {res.get('uncertainty_level')}\n\n{res.get('uncertainty_note')}")

                    # Sub-Tab 2: Patch Stability & Binned Histogram
                    with d_tab_expert:
                        st.markdown('---')
                        st.subheader('🧩 Patch Stability')
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
                    with d_tab_expert:
                        st.markdown('---')
                        st.subheader('⚖️ Hybrid Comparison')
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
                    with d_tab_expert:
                        st.markdown('---')
                        st.subheader('🌀 FFT Diagnostic')
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
                    with d_tab_expert:
                        st.markdown('---')
                        st.subheader('📜 Stage 1 Metadata & C2PA Provenance')
                        meta_data = res.get("metadata_diagnostic", {})
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.write(f"**Provenance Verdict:** `{meta_data.get('provenance_verdict')}`")
                            st.write(f"**Source Identified:** `{meta_data.get('source_identified') or 'None'}`")
                        with m_col2:
                            generator = meta_data.get('metadata_summary', {}).get('C2PA Generator')
                            if generator:
                                st.write(f"**C2PA Manifest Header:** `Detected ({generator})`")
                            else:
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


                    # Sub-Tab 7: Faithful Explanation (Gemini API)
                    with d_tab_verdict:
                        st.subheader("🤖 Faithful Explanation (Gemini Vision)")
                        st.markdown(f"**Explanation:**\n> {explanation_data.get('explanation')}")
                        if "rate limit" in str(explanation_data.get("explanation", "")).lower() or "failed" in str(explanation_data.get("explanation", "")).lower():
                            if st.button("🔄 Retry Gemini Explanation", key="retry_expl_btn"):
                                with st.spinner("Retrying explanation via Gemini Vision..."):
                                    from app.diagnostics.explainer import generate_faithful_explanation
                                    _g_img = image.copy()
                                    _g_img.thumbnail((512, 512))
                                    explanation_data = generate_faithful_explanation(
                                        image=_g_img,
                                        prediction_label=label,
                                        regions=_regions,
                                        caption=caption_input if caption_input else None,
                                        diagnostic_context=res
                                    )
                                    st.session_state["cached_explanation"] = explanation_data
                                    st.rerun()
                        

                        if caption_input:
                            st.markdown("---")
                            st.write("📝 **Multimodal Image-Text Consistency**")
                            c_score = explanation_data.get('consistency_score')
                            if c_score is not None:
                                st.metric("Consistency Score (0 to 1)", f"{c_score:.2f}")
                            st.write(f"**Note:** {explanation_data.get('consistency_note')}")
                            
                    # Sub-Tab 8: Live Degradation Test
                    with d_tab_robust:
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
            
            st.markdown("---")
            st.subheader("🛡️ Active Defence & Robustness Analysis")
            eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
            deg_img = eval_dir / "degradation_vs_accuracy.png"
            if deg_img.exists():
                st.image(str(deg_img), caption="Accuracy under JPEG Compression & Resizing (Active Defence)", width="stretch")

    # Global Footer Disclaimer
    st.markdown("---")
    st.caption(f"🛡️ **Disclaimer:** {DISCLAIMER_TEXT}")


if __name__ == "__main__":
    main()

