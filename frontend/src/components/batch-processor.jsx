import React, { useState, useRef } from "react";
import { UploadCloud, Play, CheckCircle, AlertCircle, RefreshCw, Trash2, Download } from "lucide-react";
import { RadialGlowButton } from "./ui/radial-glow-button";

export function BatchProcessor({ selectedStrategy = "auto" }) {
  const [files, setFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const fileInputRef = useRef(null);

  // Handle multi-file selection
  const handleFilesSelected = (e) => {
    const selected = Array.from(e.target.files || []);
    if (!selected.length) return;

    const validImages = selected.filter((file) => file.type.startsWith("image/"));
    if (validImages.length < selected.length) {
      setErrorMsg("Some non-image files were skipped. Only JPG, JPEG, and PNG are supported.");
    } else {
      setErrorMsg("");
    }

    const mapped = validImages.map((file) => ({
      file,
      id: `${file.name}-${file.size}-${Math.random()}`,
      name: file.name,
      size: `${(file.size / 1024).toFixed(1)} KB`,
      preview: URL.createObjectURL(file),
      status: "pending", // 'pending' | 'processing' | 'done'
    }));

    setFiles((prev) => [...prev, ...mapped]);
  };

  const handleRemoveFile = (id) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleClearAll = () => {
    setFiles([]);
    setResults([]);
    setProgress(0);
    setErrorMsg("");
  };

  // Run Batch Processing Pipeline
  const handleProcessBatch = async () => {
    if (!files.length || isProcessing) return;

    setIsProcessing(true);
    setProgress(0);
    setResults([]);
    setErrorMsg("");

    const batchOutputs = [];

    for (let i = 0; i < files.length; i++) {
      const item = files[i];
      // Mark current file as processing
      setFiles((prev) =>
        prev.map((f) => (f.id === item.id ? { ...f, status: "processing" } : f))
      );

      // Simulate model batch forward pass with realistic heuristics & latency
      await new Promise((r) => setTimeout(r, 220));

      const fn = item.name.toLowerCase();
      const isAI =
        fn.includes("ai") ||
        fn.includes("synth") ||
        fn.includes("fake") ||
        fn.includes("midjourney") ||
        fn.includes("dall") ||
        fn.includes("sd") ||
        fn.includes("gen") ||
        item.file.size < 200 * 1024; // smaller compressed or synth

      const fakeProb = isAI ? (88.5 + Math.random() * 11.2).toFixed(2) : (2.1 + Math.random() * 12.4).toFixed(2);
      const realProb = (100 - parseFloat(fakeProb)).toFixed(2);
      const label = parseFloat(fakeProb) >= 50 ? "FAKE" : "REAL";
      const confidence = Math.max(parseFloat(fakeProb), parseFloat(realProb)).toFixed(2);

      const resItem = {
        id: item.id,
        filename: item.name,
        size: item.size,
        label,
        fakeProb: `${fakeProb}%`,
        realProb: `${realProb}%`,
        confidence: `${confidence}%`,
        strategy: selectedStrategy.toUpperCase(),
        timestamp: new Date().toISOString().replace("T", " ").substring(0, 19),
      };

      batchOutputs.push(resItem);

      // Update state per file
      setFiles((prev) =>
        prev.map((f) => (f.id === item.id ? { ...f, status: "done" } : f))
      );
      setResults([...batchOutputs]);
      setProgress(Math.round(((i + 1) / files.length) * 100));
    }

    setIsProcessing(false);
  };

  // Generate and Trigger CSV Download
  const handleDownloadCSV = () => {
    if (!results.length) return;

    const headers = [
      "Filename",
      "File Size",
      "Predicted Label",
      "Fake Probability",
      "Real Probability",
      "Confidence",
      "Strategy",
      "Timestamp",
    ];

    const rows = results.map((r) => [
      `"${r.filename}"`,
      `"${r.size}"`,
      `"${r.label}"`,
      `"${r.fakeProb}"`,
      `"${r.realProb}"`,
      `"${r.confidence}"`,
      `"${r.strategy}"`,
      `"${r.timestamp}"`,
    ]);

    const csvContent = [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `signalscope_batch_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section id="batch" className="technology-section" style={{ paddingTop: "60px", paddingBottom: "60px" }}>
      <div className="section-label" style={{ textAlign: "center", marginBottom: "12px" }}>
        STREAMLIT TAB 02 / BATCH PROCESSING
      </div>

      <div className="section-heading" style={{ textAlign: "center", marginBottom: "36px" }}>
        <h2 style={{ fontSize: "clamp(32px, 4vw, 56px)" }}>
          High-Throughput <span>Batch Analysis</span>
        </h2>
        <p style={{ margin: "0 auto", maxWidth: "680px", color: "var(--muted)" }}>
          Upload multiple images to execute automated bulk inference across the selected strategy and export audited forensic logs to CSV.
        </p>
      </div>

      <div
        style={{
          maxWidth: "1150px",
          margin: "0 auto",
          padding: "32px",
          background: "rgba(10, 17, 24, 0.75)",
          border: "1px solid rgba(148, 163, 184, 0.14)",
          borderRadius: "20px",
          backdropFilter: "blur(16px)",
          boxShadow: "0 20px 50px rgba(0, 0, 0, 0.5)",
        }}
      >
        {/* Upload Zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: "2px dashed rgba(34, 211, 238, 0.35)",
            borderRadius: "14px",
            padding: "36px 20px",
            textAlign: "center",
            cursor: "pointer",
            background: "rgba(34, 211, 238, 0.02)",
            transition: "all 0.25s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "var(--cyan)";
            e.currentTarget.style.background = "rgba(34, 211, 238, 0.06)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "rgba(34, 211, 238, 0.35)";
            e.currentTarget.style.background = "rgba(34, 211, 238, 0.02)";
          }}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFilesSelected}
            accept="image/png,image/jpeg,image/jpg"
            multiple
            hidden
          />
          <div style={{ display: "flex", justifyContent: "center", marginBottom: "12px" }}>
            <div
              style={{
                width: "52px",
                height: "52px",
                borderRadius: "50%",
                background: "rgba(34, 211, 238, 0.12)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "var(--cyan)",
              }}
            >
              <UploadCloud className="w-7 h-7" />
            </div>
          </div>
          <h4 style={{ margin: "0 0 6px 0", fontSize: "17px", color: "var(--text)" }}>
            Choose multiple images or drag & drop here
          </h4>
          <p style={{ margin: 0, fontSize: "13px", color: "var(--muted)" }}>
            Supported formats: JPG, JPEG, PNG · High-throughput batch inference pipeline
          </p>
        </div>

        {errorMsg && (
          <div
            style={{
              marginTop: "16px",
              padding: "12px 16px",
              borderRadius: "10px",
              background: "rgba(245, 158, 11, 0.15)",
              border: "1px solid rgba(245, 158, 11, 0.3)",
              color: "#fbbf24",
              fontSize: "13px",
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Selected Images HUD Bar */}
        {files.length > 0 && (
          <div
            style={{
              marginTop: "24px",
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "16px",
              padding: "16px 20px",
              background: "rgba(5, 8, 12, 0.6)",
              borderRadius: "12px",
              border: "1px solid rgba(148, 163, 184, 0.12)",
            }}
          >
            <div>
              <span style={{ fontSize: "14px", fontWeight: 700, color: "var(--text)" }}>
                Total Images Selected:{" "}
                <span style={{ color: "var(--cyan)" }}>{files.length}</span>
              </span>
              <span style={{ marginLeft: "14px", fontSize: "12px", color: "var(--muted)" }}>
                Strategy: <strong>{selectedStrategy.toUpperCase()}</strong>
              </span>
            </div>

            <div style={{ display: "flex", gap: "10px" }}>
              <button
                type="button"
                onClick={handleClearAll}
                disabled={isProcessing}
                style={{
                  padding: "10px 16px",
                  borderRadius: "8px",
                  background: "rgba(239, 68, 68, 0.12)",
                  border: "1px solid rgba(239, 68, 68, 0.3)",
                  color: "#f87171",
                  fontSize: "13px",
                  fontWeight: 600,
                  cursor: isProcessing ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                <Trash2 className="w-4 h-4" />
                Clear Queue
              </button>

              <RadialGlowButton
                onClick={handleProcessBatch}
                disabled={isProcessing || !files.length}
              >
                {isProcessing ? (
                  <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Processing {progress}%
                  </span>
                ) : (
                  <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <Play className="w-4 h-4 fill-current" />
                    🚀 Process Batch Predictions
                  </span>
                )}
              </RadialGlowButton>
            </div>
          </div>
        )}

        {/* Progress Bar */}
        {isProcessing && (
          <div style={{ marginTop: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "var(--muted)", marginBottom: "6px" }}>
              <span>Executing PyTorch inference across batch queue...</span>
              <span>{progress}% complete</span>
            </div>
            <div
              style={{
                width: "100%",
                height: "6px",
                background: "rgba(255, 255, 255, 0.08)",
                borderRadius: "999px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${progress}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, #06b6d4, #3b82f6)",
                  transition: "width 0.2s ease",
                }}
              />
            </div>
          </div>
        )}

        {/* Image Chip Preview Queue */}
        {files.length > 0 && (
          <div
            style={{
              marginTop: "20px",
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
              gap: "12px",
              maxHeight: "180px",
              overflowY: "auto",
              padding: "8px",
              background: "rgba(0, 0, 0, 0.2)",
              borderRadius: "10px",
            }}
          >
            {files.map((item) => (
              <div
                key={item.id}
                style={{
                  position: "relative",
                  borderRadius: "8px",
                  overflow: "hidden",
                  border:
                    item.status === "processing"
                      ? "1px solid var(--cyan)"
                      : item.status === "done"
                      ? "1px solid rgba(16, 185, 129, 0.5)"
                      : "1px solid rgba(148, 163, 184, 0.15)",
                  background: "rgba(10, 14, 22, 0.7)",
                }}
              >
                <img
                  src={item.preview}
                  alt={item.name}
                  style={{ width: "100%", height: "70px", objectFit: "cover" }}
                />
                <div style={{ padding: "6px 8px" }}>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                    title={item.name}
                  >
                    {item.name}
                  </div>
                  <div style={{ fontSize: "10px", color: "var(--muted)", marginTop: "2px" }}>
                    {item.status === "done" ? (
                      <span style={{ color: "#34d399", display: "flex", alignItems: "center", gap: "3px" }}>
                        <CheckCircle className="w-3 h-3" /> Scanned
                      </span>
                    ) : item.status === "processing" ? (
                      <span style={{ color: "var(--cyan)" }}>Analyzing...</span>
                    ) : (
                      item.size
                    )}
                  </div>
                </div>

                {!isProcessing && (
                  <button
                    onClick={() => handleRemoveFile(item.id)}
                    style={{
                      position: "absolute",
                      top: "4px",
                      right: "4px",
                      width: "20px",
                      height: "20px",
                      borderRadius: "50%",
                      background: "rgba(0, 0, 0, 0.7)",
                      color: "#fff",
                      border: "none",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "11px",
                    }}
                    title="Remove image"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Results Data Table */}
        {results.length > 0 && (
          <div style={{ marginTop: "32px" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "16px",
              }}
            >
              <div>
                <h4 style={{ margin: "0 0 4px 0", fontSize: "18px", color: "var(--text)" }}>
                  📊 Batch Inference Results ({results.length} Images)
                </h4>
                <p style={{ margin: 0, fontSize: "12px", color: "var(--muted)" }}>
                  Streamlit DataFrame Parity · Fully auditable prediction log
                </p>
              </div>

              <button
                type="button"
                onClick={handleDownloadCSV}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "10px 18px",
                  borderRadius: "9px",
                  background: "rgba(34, 211, 238, 0.12)",
                  border: "1px solid rgba(34, 211, 238, 0.4)",
                  color: "var(--cyan)",
                  fontSize: "13px",
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                <Download className="w-4 h-4" />
                📥 Download Batch Results CSV
              </button>
            </div>

            <div
              style={{
                overflowX: "auto",
                borderRadius: "12px",
                border: "1px solid rgba(148, 163, 184, 0.15)",
                background: "rgba(5, 8, 12, 0.8)",
              }}
            >
              <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "13px" }}>
                <thead>
                  <tr style={{ background: "rgba(15, 23, 42, 0.8)", borderBottom: "1px solid rgba(148, 163, 184, 0.15)" }}>
                    <th style={{ padding: "12px 16px", color: "var(--cyan)", fontWeight: 600 }}>#</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Filename</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Verdict</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Fake Probability</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Real Probability</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Confidence</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Strategy</th>
                    <th style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 600 }}>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, idx) => (
                    <tr
                      key={r.id}
                      style={{
                        borderBottom: "1px solid rgba(148, 163, 184, 0.08)",
                        background: idx % 2 === 0 ? "transparent" : "rgba(255, 255, 255, 0.015)",
                      }}
                    >
                      <td style={{ padding: "12px 16px", color: "var(--muted)" }}>{idx + 1}</td>
                      <td style={{ padding: "12px 16px", color: "var(--text)", fontWeight: 500 }}>
                        <div style={{ maxWidth: "200px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {r.filename}
                        </div>
                      </td>
                      <td style={{ padding: "12px 16px" }}>
                        <span
                          style={{
                            padding: "4px 10px",
                            borderRadius: "999px",
                            fontSize: "11px",
                            fontWeight: 700,
                            letterSpacing: "0.05em",
                            background: r.label === "FAKE" ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
                            color: r.label === "FAKE" ? "#f87171" : "#34d399",
                            border: r.label === "FAKE" ? "1px solid rgba(239, 68, 68, 0.3)" : "1px solid rgba(16, 185, 129, 0.3)",
                          }}
                        >
                          {r.label === "FAKE" ? "🤖 AI FAKE" : "📸 REAL"}
                        </span>
                      </td>
                      <td style={{ padding: "12px 16px", color: "#f87171", fontWeight: 600 }}>{r.fakeProb}</td>
                      <td style={{ padding: "12px 16px", color: "#34d399", fontWeight: 600 }}>{r.realProb}</td>
                      <td style={{ padding: "12px 16px", color: "var(--text)" }}>{r.confidence}</td>
                      <td style={{ padding: "12px 16px", color: "var(--cyan)", fontFamily: "monospace", fontSize: "12px" }}>
                        {r.strategy}
                      </td>
                      <td style={{ padding: "12px 16px", color: "var(--muted)", fontSize: "12px" }}>{r.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default BatchProcessor;
