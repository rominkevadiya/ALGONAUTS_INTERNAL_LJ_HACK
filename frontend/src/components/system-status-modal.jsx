import React from "react";
import { X, Cpu, CheckCircle2, ShieldAlert, Database, Layers, Activity } from "lucide-react";

export function SystemStatusModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in duration-200">
      <div 
        className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border border-cyan-500/30 bg-[#0a1118]/95 p-6 md:p-8 text-slate-100 shadow-[0_0_50px_rgba(34,211,238,0.15)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors"
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-800">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Model Specifications & System Status
            </h3>
            <p className="text-xs text-slate-400">
              Corresponds to Streamlit Sidebar Runtime Configuration
            </p>
          </div>
        </div>

        {/* System Status Banner */}
        <div className="mb-6 p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </div>
            <div>
              <div className="text-sm font-semibold text-emerald-300">
                ResNet-50 Checkpoint Active
              </div>
              <div className="text-xs text-emerald-400/80">
                PyTorch Neural Pipeline · PyTorch 2.x Architecture
              </div>
            </div>
          </div>
          <div className="px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-xs font-mono font-medium text-emerald-300 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5" />
            🔥 GPU / Native CUDA Ready
          </div>
        </div>

        {/* Grid of Key Architecture Info */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Architecture</div>
            <div className="text-sm font-bold text-white mt-1">ResNet-50</div>
          </div>
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Stem Input</div>
            <div className="text-sm font-bold text-cyan-400 mt-1">32 × 32 px</div>
          </div>
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Throughput</div>
            <div className="text-sm font-bold text-emerald-400 mt-1">~916 img/s</div>
          </div>
          <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-center">
            <div className="text-xs text-slate-400">Parameters</div>
            <div className="text-sm font-bold text-slate-200 mt-1">23.5 Million</div>
          </div>
        </div>

        {/* Dataset Information */}
        <div className="mb-6 p-4 rounded-xl bg-slate-900/70 border border-slate-800">
          <h4 className="text-sm font-semibold text-cyan-300 flex items-center gap-2 mb-2">
            <Database className="w-4 h-4" />
            Training Dataset: CIFAKE (120,000 Images)
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed mb-3">
            Trained on a rigorously balanced benchmark comprising 120,000 total samples:
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg bg-black/40 border border-slate-800/80">
              <span className="font-semibold text-emerald-400">Authentic (60,000):</span>
              <p className="text-slate-400 mt-0.5">Real CIFAR-10 optical photographs across natural categories.</p>
            </div>
            <div className="p-2.5 rounded-lg bg-black/40 border border-slate-800/80">
              <span className="font-semibold text-rose-400">AI-Generated (60,000):</span>
              <p className="text-slate-400 mt-0.5">Synthesized via Stable Diffusion v1.4 latent diffusion models.</p>
            </div>
          </div>
        </div>

        {/* Strengths & Limitations */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <h5 className="text-xs font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5 mb-2">
              <CheckCircle2 className="w-4 h-4" />
              Model Strengths
            </h5>
            <ul className="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li>Highly sensitive to latent diffusion deconvolution patterns.</li>
              <li>Exceptional robustness to mild compression and image resizing.</li>
              <li>High-speed batched evaluation capability (~916 img/s).</li>
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/80">
            <h5 className="text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center gap-1.5 mb-2">
              <ShieldAlert className="w-4 h-4" />
              Model Limitations
            </h5>
            <ul className="text-xs text-slate-300 space-y-1.5 list-disc pl-4">
              <li>May exhibit reduced sensitivity on older non-diffusion GANs.</li>
              <li>Extreme photographic noise or heavy filters can trigger false positives.</li>
              <li>Domain shift on uncalibrated ultra-high-resolution webcams.</li>
            </ul>
          </div>
        </div>

        {/* Confusion Matrix Insights */}
        <div className="p-4 rounded-xl bg-cyan-950/30 border border-cyan-500/20">
          <h5 className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-1.5 mb-2">
            <Layers className="w-4 h-4" />
            Confusion Matrix Insights
          </h5>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-300">
            <div>
              <strong className="text-white">High Recall on Fake:</strong> The model rarely misses synthetic AI-generated images (98.34% recall).
            </div>
            <div>
              <strong className="text-white">Precision on Real:</strong> Conservative decision boundary ensures photographic authentic fidelity is protected.
            </div>
          </div>
        </div>

        {/* Footer info */}
        <div className="mt-6 pt-4 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
          <span>Normalization: ImageNet Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default SystemStatusModal;
