import React from 'react';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AboutModal: React.FC<AboutModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">Model Specifications Card</div>
          <button className="close-modal-btn" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
          <div>
            <div style={{ fontSize: '20px', fontWeight: 900, color: 'var(--brand-primary)', letterSpacing: '-0.03em' }}>Collision 1.0B Flagship</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>999,376,128 Parameters · Causal Transformer Core · Grounded Architecture</div>
          </div>

          <p>
            Collision is an open, enterprise-grade causal transformer built specifically for CPU-first execution and accelerated multi-GPU environments, delivering verifiable generation with sentence-level factual attribution.
          </p>

          <div style={{ background: 'var(--bg-surface)', padding: '14px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>Architecture & Technical Specifications</div>
            <ul style={{ paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12.5px' }}>
              <li><strong>Model Architecture:</strong> Collision 1.0B (Causal Decoder-Only Transformer)</li>
              <li><strong>Active Parameters:</strong> 999,376,128 (~1.00B)</li>
              <li><strong>Layer Distribution:</strong> 24 Layers, $d_{'{model}'}=2048$, 16 Attention Heads, $d_{'{ff}'}=5376$</li>
              <li><strong>Positional & Activation:</strong> Rotary Position Embedding (RoPE) + SwiGLU</li>
              <li><strong>Context Window:</strong> 1,024 token sequence length</li>
              <li><strong>Verification Engine:</strong> Grounded RAG with NLI claim verification</li>
              <li><strong>Inference Optimization:</strong> Vectorized CPU kernels (AVX-512 / ARM NEON, 11.8ms latency)</li>
              <li><strong>Production Endpoint:</strong> REST `/v1/ask` with latency, source, and claim metrics</li>
            </ul>
          </div>

          <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '4px' }}>
            © 2026 Collision AI Research. Open Weights & Research Checkpoints.
          </div>
        </div>
      </div>
    </div>
  );
};
