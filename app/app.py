"""
SignalScope - AI-Generated Image Detection Application
Streamlit Interface for ResNet-50 Model Inference
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root directory is first in sys.path and remove script directory to avoid name collision with app package
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

# Custom imports from app package (handles direct script execution vs package import)
try:
    from app.config import (
        BENCHMARK_METRICS,
        DISCLAIMER_TEXT,
        CONFIDENCE_DISCLAIMER
    )
    from app.model_loader import load_model
    from app.predictor import predict_image, predict_batch, preprocess_image
except (ModuleNotFoundError, ImportError):
    from config import (
        BENCHMARK_METRICS,
        DISCLAIMER_TEXT,
        CONFIDENCE_DISCLAIMER
    )
    from model_loader import load_model
    from predictor import predict_image, predict_batch, preprocess_image

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
    /* Global styles */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    /* Header card */
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
    
    /* Result Cards */
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
    
    .debug-box {
        background-color: #0f172a;
        border: 1px dashed #475569;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: #38bdf8;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Application Header
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🔍 SignalScope</div>
        <div class="header-subtitle">AI-Generated Image Detection System</div>
        <p style="margin-top: 0.8rem; color: #cbd5e1; font-size: 0.95rem;">
            Screen images using the trained ResNet-50 model to classify whether an image is 
            <strong>AI-Generated (FAKE)</strong> or an <strong>Authentic Photograph (REAL)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Model Setup & Benchmarks
    with st.sidebar:
        st.header("⚡ System Info")
        
        # Load model dynamically
        try:
            model, device = load_model()
            device_label = "🔥 GPU (CUDA)" if device.type == "cuda" else "💻 CPU"
            st.success(f"ResNet-50 Model Active ({device_label})")
        except Exception as err:
            st.error("Failed to load model checkpoint.")
            st.error(f"Details: {str(err)}")
            st.stop()

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
                help="Upload an image to run live PyTorch model inference."
            )

            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)
                    image.load()
                except Exception as img_err:
                    st.error(f"Unable to read uploaded image file: {str(img_err)}")
                    image = None

                if image is not None:
                    st.image(image, caption=f"Preview: {uploaded_file.name}", width="stretch")
                    width, height = image.size
                    st.caption(f"**Filename:** `{uploaded_file.name}` | **Size:** {width} × {height} px | **Color Mode:** {image.mode}")
                    
                    analyze_clicked = st.button("🔎 Analyze Image", type="primary", width="stretch")
                else:
                    analyze_clicked = False
            else:
                st.info("Upload an image on the left to begin analysis.")
                analyze_clicked = False

        with col_output:
            st.subheader("🎯 Live Inference Output")

            if uploaded_file is not None and analyze_clicked:
                with st.spinner("Executing PyTorch forward pass..."):
                    start_t = time.time()
                    
                    # Direct raw forward pass to obtain exact unrounded logits
                    input_tensor = preprocess_image(image).to(device)
                    with torch.no_grad():
                        raw_logits = model(input_tensor).squeeze(0)
                        raw_probs = F.softmax(raw_logits, dim=0)

                    elapsed_ms = (time.time() - start_t) * 1000

                    logit_fake = float(raw_logits[0].item())
                    logit_real = float(raw_logits[1].item())
                    fake_prob = float(raw_probs[0].item())
                    real_prob = float(raw_probs[1].item())

                    label = "FAKE" if fake_prob > real_prob else "REAL"
                    conf = fake_prob if label == "FAKE" else real_prob

                    # Status Box Rendering
                    if conf < 0.70:
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
                        <div class="result-subtext">Confidence Score: {conf * 100:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("📈 Computed Probabilities")
                    
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Fake Probability", f"{fake_prob * 100:.2f}%")
                        st.progress(fake_prob)
                    with col_p2:
                        st.metric("Real Probability", f"{real_prob * 100:.2f}%")
                        st.progress(real_prob)

                    # Live PyTorch Model Diagnostics Panel (Proves No Static Data)
                    with st.expander("🔬 Live PyTorch Diagnostic Logits (Proof of Dynamic Computation)", expanded=False):
                        st.markdown(f"""
                        **Forward Pass Timing:** `{elapsed_ms:.2f} ms`  
                        **Device Executed:** `{device.type.upper()}`  
                        
                        **Raw Unrounded Output Logits:**
                        - `Logit Class 0 (FAKE):` `{logit_fake:+.6f}`
                        - `Logit Class 1 (REAL):` `{logit_real:+.6f}`
                        
                        **Raw Softmax Probabilities:**
                        - `Prob Class 0 (FAKE):` `{fake_prob:.8f}`
                        - `Prob Class 1 (REAL):` `{real_prob:.8f}`
                        """)

                    # Domain Shift Notice
                    st.markdown("---")
                    st.subheader("💡 Analysis Explanation & Domain Limitation")
                    if label == "FAKE":
                        st.write(
                            "The model detected spatial artifact patterns associated with Class 0 (FAKE) in its training distribution."
                        )
                    else:
                        st.write(
                            "The model detected texture characteristics associated with Class 1 (REAL) in its training distribution."
                        )

                    st.warning(
                        "⚠️ **Important Note on Dataset Scope:** This ResNet-50 model was trained strictly on the **CIFAKE dataset** "
                        "(32×32 CIFAR-10 images upscaled to 224×224). High-resolution webcam photographs of human faces or modern digital camera photos "
                        "possess very different pixel statistics than CIFAR-10 real photos. As a result, the model may classify modern webcam photos as FAKE "
                        "due to dataset domain shift."
                    )

            elif uploaded_file is not None:
                st.info("Click **Analyze Image** above to run the live PyTorch inference engine.")

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
                
                with st.spinner(f"Running inference on {len(images_dict)} images..."):
                    df_results = predict_batch(images_dict, model=model, device=device)

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
        st.subheader("🔬 Architecture & Dataset Information")
        
        st.markdown("""
        ### SignalScope Pipeline Overview
        SignalScope uses a fine-tuned **ResNet-50** neural network trained on the **CIFAKE** dataset.
        
        - **Input Resolution:** $32 \\times 32$ RGB Image (adapted native stem)
        - **Normalization:** ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
        - **Optimizer:** AdamW ($1 \\times 10^{-4}$) with Cross-Entropy Loss
        """)

        st.markdown("---")
        st.subheader("📈 Benchmark Performance Metrics")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Test Accuracy", BENCHMARK_METRICS.get("Test Accuracy", "98.33%"))
            st.metric("Macro F1 Score", BENCHMARK_METRICS.get("Macro F1 Score", "0.9832"))
        with m_col2:
            st.metric("ROC-AUC", BENCHMARK_METRICS.get("ROC-AUC", "0.9987"))
            st.metric("PR-AUC", BENCHMARK_METRICS.get("PR-AUC", "0.9988"))
        with m_col3:
            st.metric("Sensitivity", "98.34%")
            st.metric("Specificity", "98.31%")

        # Display evaluation plots if available
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
