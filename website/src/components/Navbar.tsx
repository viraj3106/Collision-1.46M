import React, { useState } from 'react';
import { CollisionWordmark } from './CollisionWordmark';

interface NavbarProps {
  currentView: 'landing' | 'playground';
  onNavigate: (view: 'landing' | 'playground') => void;
  onOpenSettings: () => void;
  onOpenAbout: () => void;
  onOpenLicense: () => void;
  isDarkTheme: boolean;
  onToggleTheme: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentView,
  onNavigate,
  onOpenSettings,
  onOpenAbout,
  onOpenLicense,
  isDarkTheme,
  onToggleTheme,
}) => {
  const [activeMegaMenu, setActiveMegaMenu] = useState<string | null>(null);

  const toggleMegaMenu = (menuName: string) => {
    setActiveMegaMenu(prev => (prev === menuName ? null : menuName));
  };

  return (
    <header className="site-header">
      <div className="header-container">
        {/* Left: Automated Looping Brand Wordmark in Header */}
        <div className="header-left">
          <div className="brand-nav-link" onClick={() => { onNavigate('landing'); setActiveMegaMenu(null); }}>
            <CollisionWordmark mode="auto-loop" fontSize="18px" intervalSeconds={7} />
          </div>

          <nav className="nav-links-desktop">
            <button
              className={`nav-item-btn ${activeMegaMenu === 'models' ? 'active' : ''}`}
              onClick={() => toggleMegaMenu('models')}
            >
              <span>Models</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>

            <button
              className={`nav-item-btn ${activeMegaMenu === 'products' ? 'active' : ''}`}
              onClick={() => toggleMegaMenu('products')}
            >
              <span>Products</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>

            <button
              className={`nav-item-btn ${activeMegaMenu === 'solutions' ? 'active' : ''}`}
              onClick={() => toggleMegaMenu('solutions')}
            >
              <span>Solutions</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>

            <button
              className={`nav-item-btn ${currentView === 'playground' ? 'active' : ''}`}
              onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}
            >
              <span>Studio / Playground</span>
            </button>

            <button
              className="nav-item-btn"
              onClick={() => { onOpenAbout(); setActiveMegaMenu(null); }}
            >
              <span>Research & About</span>
            </button>
          </nav>
        </div>

        {/* Right Action buttons */}
        <div className="header-right">
          {/* Dark / Light Toggle */}
          <button
            className="header-btn header-btn-secondary"
            onClick={onToggleTheme}
            title="Toggle theme mode"
            style={{ padding: '0 14px' }}
          >
            {isDarkTheme ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
              </svg>
            )}
          </button>

          <button
            className="header-btn header-btn-secondary"
            onClick={onOpenSettings}
            title="Model Inference Settings"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            <span>Config</span>
          </button>

          <button
            className="header-btn header-btn-primary"
            onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}
          >
            <span>Start Building</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </button>
        </div>
      </div>

      {/* Mega Menu Dropdowns */}
      {activeMegaMenu === 'models' && (
        <div className="mega-menu-overlay" onMouseLeave={() => setActiveMegaMenu(null)}>
          <div className="mega-menu-grid">
            <div className="mega-col">
              <div className="mega-col-title">Flagship Models</div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">⚡</div>
                <div>
                  <div className="mega-link-name">COLLISION-1.0B</div>
                  <div className="mega-link-desc">1B parameter frontier dense transformer for ultra-fast local & CPU deployment.</div>
                </div>
              </div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">🧠</div>
                <div>
                  <div className="mega-link-name">COLLISION-Reason</div>
                  <div className="mega-link-desc">Step-by-step verified reasoning & mathematical derivation engine.</div>
                </div>
              </div>
            </div>

            <div className="mega-col">
              <div className="mega-col-title">Specialized Models</div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">🌐</div>
                <div>
                  <div className="mega-link-name">COLLISION-Grounded</div>
                  <div className="mega-link-desc">Live web retrieval with attribution citation & claim-level verification.</div>
                </div>
              </div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">💻</div>
                <div>
                  <div className="mega-link-name">COLLISION-Code</div>
                  <div className="mega-link-desc">Optimized syntax execution, refactoring, and AST inspection.</div>
                </div>
              </div>
            </div>

            <div className="mega-col">
              <div className="mega-col-title">Resources & Legal</div>
              <div className="mega-link-item" onClick={() => { onOpenAbout(); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">📄</div>
                <div>
                  <div className="mega-link-name">Model Card & Weights</div>
                  <div className="mega-link-desc">Open-weight checkpoints, tokenizer vocabs, and training manifests.</div>
                </div>
              </div>
              <div className="mega-link-item" onClick={() => { onOpenLicense(); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">⚖️</div>
                <div>
                  <div className="mega-link-name">MIT License</div>
                  <div className="mega-link-desc">Permissive open-source commercial & research licensing terms.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeMegaMenu === 'products' && (
        <div className="mega-menu-overlay" onMouseLeave={() => setActiveMegaMenu(null)}>
          <div className="mega-menu-grid">
            <div className="mega-col">
              <div className="mega-col-title">Platform & Tools</div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">🎛️</div>
                <div>
                  <div className="mega-link-name">COLLISION Studio</div>
                  <div className="mega-link-desc">Interactive developer playground, prompt tester, and token inspector.</div>
                </div>
              </div>
              <div className="mega-link-item" onClick={() => { onNavigate('landing'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">🔨</div>
                <div>
                  <div className="mega-link-name">Forge</div>
                  <div className="mega-link-desc">Fine-tuning, RLHF alignment, and custom distillation pipeline.</div>
                </div>
              </div>
            </div>

            <div className="mega-col">
              <div className="mega-col-title">Agents & Cloud</div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">🤖</div>
                <div>
                  <div className="mega-link-name">COLLISION Vibe</div>
                  <div className="mega-link-desc">Autonomous multi-step coding agent for terminal and background tasks.</div>
                </div>
              </div>
              <div className="mega-link-item" onClick={() => { onNavigate('landing'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">☁️</div>
                <div>
                  <div className="mega-link-name">AI Cloud Inference</div>
                  <div className="mega-link-desc">Sub-millisecond latency endpoints with verified uptime SLA.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeMegaMenu === 'solutions' && (
        <div className="mega-menu-overlay" onMouseLeave={() => setActiveMegaMenu(null)}>
          <div className="mega-menu-grid">
            <div className="mega-col">
              <div className="mega-col-title">Domains</div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">💼</div>
                <div>
                  <div className="mega-link-name">Financial & Risk AI</div>
                  <div className="mega-link-desc">Deterministic calculations and low-latency regulatory parsing.</div>
                </div>
              </div>
              <div className="mega-link-item" onClick={() => { onNavigate('playground'); setActiveMegaMenu(null); }}>
                <div className="mega-link-icon">🔬</div>
                <div>
                  <div className="mega-link-name">Research & Science</div>
                  <div className="mega-link-desc">Open science models built with zero hallucination constraints.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
