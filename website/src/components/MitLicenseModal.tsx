import React, { useState } from 'react';

interface MitLicenseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MitLicenseModal: React.FC<MitLicenseModalProps> = ({ isOpen, onClose }) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const licenseText = `MIT License

Copyright (c) 2026 COLLISION AI Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.`;

  const handleCopy = () => {
    navigator.clipboard.writeText(licenseText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card license-modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className="license-badge-icon">⚖️</div>
            <div>
              <div className="modal-title">MIT License</div>
              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Open-source permissive release for Collision AI</div>
            </div>
          </div>
          <button className="close-modal-btn" onClick={onClose} title="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <div className="license-permissions-row">
          <div className="permission-item allowed">
            <span className="perm-dot">✓</span> Commercial Use
          </div>
          <div className="permission-item allowed">
            <span className="perm-dot">✓</span> Modification
          </div>
          <div className="permission-item allowed">
            <span className="perm-dot">✓</span> Distribution
          </div>
          <div className="permission-item allowed">
            <span className="perm-dot">✓</span> Private Use
          </div>
          <div className="permission-item neutral">
            <span className="perm-dot">ℹ</span> License & Copyright Notice
          </div>
        </div>

        <div className="license-code-block">
          <div className="license-code-header">
            <span>LICENSE · Plaintext</span>
            <button className="copy-license-btn" onClick={handleCopy}>
              {copied ? (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                  <span>Copy License</span>
                </>
              )}
            </button>
          </div>
          <pre className="license-pre">
            <code>{licenseText}</code>
          </pre>
        </div>

        <div className="license-footer-note">
          <span>Applies to Collision-1.0B open weights, Python SDK, C++ AVX-512 kernels, and web platform.</span>
        </div>
      </div>
    </div>
  );
};
