import React from 'react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  temperature: number;
  setTemperature: (v: number) => void;
  maxTokens: number;
  setMaxTokens: (v: number) => void;
  onClearHistory: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
  onClearHistory
}) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">Settings & Preferences</div>
          <button className="close-modal-btn" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
              Temperature ({temperature})
            </label>
            <input 
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
            />
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Controls randomness. Lower values produce more deterministic responses.
            </span>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
              Max Tokens ({maxTokens})
            </label>
            <input 
              type="range"
              min="20"
              max="256"
              step="10"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: 'var(--accent-purple)' }}
            />
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Maximum token length generated per turn (Upper context bound: 256).
            </span>
          </div>

          <div style={{ paddingTop: '12px', borderTop: '1px solid var(--border-subtle)' }}>
            <label style={{ display: 'block', fontSize: '13.5px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
              Data & Storage
            </label>
            <button 
              className="sidebar-footer-btn"
              onClick={onClearHistory}
              style={{ color: '#EF4444', backgroundColor: '#FEF2F2', border: '1px solid #FCA5A5' }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              </svg>
              Clear all local chat history
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
