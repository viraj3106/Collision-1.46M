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
          <div className="modal-title">About COLLISION AI</div>
          <button className="close-modal-btn" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="brand-logo-icon" style={{ width: '36px', height: '36px', fontSize: '18px' }}>C</div>
            <div>
              <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>COLLISION-10M</div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>CPU-First Language Model</div>
            </div>
          </div>

          <p>
            COLLISION is a dedicated language model built specifically for CPU-first execution, delivering rapid generation without requiring specialized accelerator hardware.
          </p>

          <div style={{ background: 'var(--bg-surface)', padding: '12px 14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>Model Specifications</div>
            <ul style={{ paddingLeft: '18px', display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12.5px' }}>
              <li><strong>Parameters:</strong> 10,282,304</li>
              <li><strong>Architecture:</strong> Decoder-only Transformer</li>
              <li><strong>Context Window:</strong> 256 tokens</li>
              <li><strong>Target Platform:</strong> CPU Native Execution</li>
              <li><strong>Checkpoint Verification:</strong> FROZEN (d256d46d...)</li>
            </ul>
          </div>

          <div style={{ fontSize: '12px', color: 'var(--text-muted)', textAlign: 'center', marginTop: '4px' }}>
            © 2026 COLLISION AI Research. All rights reserved.
          </div>
        </div>
      </div>
    </div>
  );
};
