import React, { useState } from "react";
import { Maximize2, X, ZoomIn, ShieldCheck, LineChart } from "lucide-react";

export function BenchmarkPlotsGallery() {
  const [activePlot, setActivePlot] = useState(null);

  const PLOTS = [
    {
      id: "cm",
      title: "Confusion Matrix (CIFAKE Official Test Split)",
      category: "Classification Matrix",
      src: "/plots/confusion_matrix.png",
      caption: "Evaluated across 20,000 balanced test samples (10,000 real CIFAR-10 vs 10,000 synthetic Stable Diffusion v1.4). Demonstrates 98.34% sensitivity on FAKE images and 98.31% specificity on REAL images.",
    },
    {
      id: "roc",
      title: "Receiver Operating Characteristic (ROC Curve)",
      category: "Discrimination Metric",
      src: "/plots/roc_curve.png",
      caption: "ROC-AUC of 0.9987 indicates near-optimal discrimination across classification thresholds, minimizing false-alarm rates for real photographic content.",
    },
    {
      id: "acc",
      title: "Training & Validation Accuracy Trajectory",
      category: "Convergence Metric",
      src: "/plots/training_validation_accuracy.png",
      caption: "Convergence dynamics of the retrained ResNet-50 backbone with native 32px stem, exhibiting steady generalization without catastrophic overfitting.",
    },
    {
      id: "pr",
      title: "Precision-Recall Curve (PR-AUC)",
      category: "Threshold Robustness",
      src: "/plots/precision_recall_curve.png",
      caption: "Evaluates model precision stability across varying recall demands, verifying sustained precision even under elevated synthetic detection sensitivity.",
    },
    {
      id: "degrade",
      title: "Active Defence: Degradation vs Accuracy",
      category: "Adversarial & Compression Robustness",
      src: "/plots/degradation_vs_accuracy.png",
      caption: "Evaluates detection accuracy degradation across aggressive JPEG compression (Quality: 90 down to 10) and bicubic downscaling stress factors.",
      featured: true,
    },
  ];

  return (
    <div style={{ marginTop: "40px" }}>
      <div style={{ textAlign: "center", marginBottom: "28px" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "8px", color: "var(--cyan)", fontSize: "12px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" }}>
          <LineChart className="w-4 h-4" />
          <span>Streamlit Tab 03 Parity</span>
        </div>
        <h3 style={{ fontSize: "28px", color: "var(--text)", margin: "8px 0" }}>
          Benchmark Visualizations & Active Defence
        </h3>
        <p style={{ color: "var(--muted)", fontSize: "14px", maxWidth: "650px", margin: "0 auto" }}>
          Direct graphical artifacts generated during reproducible PyTorch evaluation passes on the official CIFAKE test set.
        </p>
      </div>

      {/* Grid of 4 Standard Plots */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "20px",
          marginBottom: "24px",
        }}
      >
        {PLOTS.filter((p) => !p.featured).map((plot) => (
          <div
            key={plot.id}
            onClick={() => setActivePlot(plot)}
            style={{
              position: "relative",
              borderRadius: "14px",
              background: "rgba(10, 17, 24, 0.7)",
              border: "1px solid rgba(148, 163, 184, 0.14)",
              overflow: "hidden",
              cursor: "pointer",
              transition: "transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-3px)";
              e.currentTarget.style.borderColor = "var(--cyan)";
              e.currentTarget.style.boxShadow = "0 10px 30px rgba(34, 211, 238, 0.15)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.borderColor = "rgba(148, 163, 184, 0.14)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <div style={{ position: "relative", height: "210px", background: "#05080c", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <img
                src={plot.src}
                alt={plot.title}
                style={{ width: "100%", height: "100%", objectFit: "contain", padding: "8px" }}
              />
              <div
                style={{
                  position: "absolute",
                  top: "10px",
                  right: "10px",
                  padding: "6px",
                  borderRadius: "6px",
                  background: "rgba(0,0,0,0.6)",
                  backdropFilter: "blur(4px)",
                  color: "var(--cyan)",
                }}
              >
                <ZoomIn className="w-4 h-4" />
              </div>
            </div>

            <div style={{ padding: "16px" }}>
              <div style={{ fontSize: "11px", fontWeight: 700, color: "var(--cyan)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                {plot.category}
              </div>
              <h4 style={{ margin: "4px 0 8px 0", fontSize: "15px", color: "var(--text)" }}>
                {plot.title}
              </h4>
              <p style={{ margin: 0, fontSize: "12px", color: "var(--muted)", lineHeight: "1.5" }}>
                {plot.caption}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Featured Active Defence Plot */}
      {PLOTS.find((p) => p.featured) && (
        <div
          onClick={() => setActivePlot(PLOTS.find((p) => p.featured))}
          style={{
            borderRadius: "16px",
            background: "rgba(10, 17, 24, 0.8)",
            border: "1px solid rgba(34, 211, 238, 0.3)",
            overflow: "hidden",
            cursor: "pointer",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: "24px",
            padding: "24px",
            alignItems: "center",
            boxShadow: "0 10px 40px rgba(0, 0, 0, 0.4)",
            transition: "border-color 0.25s ease",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--cyan)")}
          onMouseLeave={(e) => (e.currentTarget.style.borderColor = "rgba(34, 211, 238, 0.3)")}
        >
          <div style={{ height: "260px", background: "#05080c", borderRadius: "10px", overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", border: "1px solid rgba(148, 163, 184, 0.1)" }}>
            <img
              src="/plots/degradation_vs_accuracy.png"
              alt="Active Defence Degradation Curve"
              style={{ width: "100%", height: "100%", objectFit: "contain", padding: "10px" }}
            />
          </div>

          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "4px 10px", borderRadius: "999px", background: "rgba(34, 211, 238, 0.12)", color: "var(--cyan)", fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em", marginBottom: "12px" }}>
              <ShieldCheck className="w-3.5 h-3.5" />
              ACTIVE DEFENCE & STABILITY
            </div>
            <h4 style={{ fontSize: "20px", color: "var(--text)", margin: "0 0 10px 0" }}>
              Accuracy Under Severe JPEG Compression & Resizing
            </h4>
            <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: "1.6", margin: "0 0 16px 0" }}>
              Empirical stress-testing reveals that SignalScope maintains robust classification performance above 94% accuracy even when image quality is degraded to JPEG Q=30, outperforming conventional detectors that collapse under compression blur.
            </p>
            <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "var(--cyan)", fontWeight: 600 }}>
              <Maximize2 className="w-3.5 h-3.5" />
              Click to inspect high-resolution plot
            </div>
          </div>
        </div>
      )}

      {/* Lightbox Modal */}
      {activePlot && (
        <div
          className="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-200"
          onClick={() => setActivePlot(null)}
        >
          <div
            className="relative w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-2xl border border-cyan-500/40 bg-[#0a1118] p-6 text-slate-100 shadow-[0_0_60px_rgba(34,211,238,0.2)]"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setActivePlot(null)}
              className="absolute top-4 right-4 p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div style={{ marginBottom: "16px" }}>
              <span style={{ fontSize: "12px", color: "var(--cyan)", fontWeight: 700, textTransform: "uppercase" }}>
                {activePlot.category}
              </span>
              <h3 style={{ fontSize: "22px", color: "var(--text)", margin: "4px 0" }}>
                {activePlot.title}
              </h3>
            </div>

            <div style={{ background: "#05080c", borderRadius: "12px", padding: "12px", border: "1px solid rgba(148, 163, 184, 0.12)", marginBottom: "16px", display: "flex", justifyContent: "center" }}>
              <img
                src={activePlot.src}
                alt={activePlot.title}
                style={{ maxWidth: "100%", maxHeight: "60vh", objectFit: "contain" }}
              />
            </div>

            <p style={{ fontSize: "13px", color: "var(--muted)", lineHeight: "1.6", margin: 0 }}>
              {activePlot.caption}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default BenchmarkPlotsGallery;
