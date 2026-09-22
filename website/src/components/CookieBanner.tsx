import React, { useState, useEffect } from 'react';

interface CookieBannerProps {
  onOpenPreferences?: () => void;
}

export const CookieBanner: React.FC<CookieBannerProps> = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [showPreferences, setShowPreferences] = useState(false);
  const [analyticsEnabled, setAnalyticsEnabled] = useState(true);
  const [performanceEnabled, setPerformanceEnabled] = useState(true);

  useEffect(() => {
    const consent = localStorage.getItem('collision_cookie_consent');
    if (!consent) {
      const timer = setTimeout(() => {
        setIsVisible(true);
      }, 1200);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleAcceptAll = () => {
    localStorage.setItem(
      'collision_cookie_consent',
      JSON.stringify({ essential: true, analytics: true, performance: true, timestamp: Date.now() })
    );
    setIsVisible(false);
  };

  const handleDeclineNonEssential = () => {
    localStorage.setItem(
      'collision_cookie_consent',
      JSON.stringify({ essential: true, analytics: false, performance: false, timestamp: Date.now() })
    );
    setIsVisible(false);
  };

  const handleSaveCustom = () => {
    localStorage.setItem(
      'collision_cookie_consent',
      JSON.stringify({ essential: true, analytics: analyticsEnabled, performance: performanceEnabled, timestamp: Date.now() })
    );
    setShowPreferences(false);
    setIsVisible(false);
  };

  if (!isVisible) return null;

  return (
    <div className="cookie-banner-wrapper">
      <div className="cookie-banner-card">
        {/* Glow accent beam */}
        <div className="cookie-glow-beam"></div>

        <div className="cookie-banner-content">
          <div className="cookie-icon-wrap">
            <span style={{ fontSize: '20px' }}>🍪</span>
          </div>

          <div className="cookie-text-area">
            <div className="cookie-title">Cookie & Telemetry Preferences</div>
            <p className="cookie-desc">
              Collision AI uses essential cookies to preserve inference sessions and maintain hardware cache states. We also collect anonymous benchmark metrics to optimize our AVX-512 CPU kernels.
            </p>
          </div>
        </div>

        {showPreferences && (
          <div className="cookie-preferences-drawer">
            <div className="pref-item">
              <div className="pref-info">
                <span className="pref-name">Essential Inference State</span>
                <span className="pref-sub">Required for model sessions, context streaming, and security tokens.</span>
              </div>
              <span className="pref-badge-locked">Always Active</span>
            </div>

            <div className="pref-item">
              <div className="pref-info">
                <span className="pref-name">Performance & Latency Telemetry</span>
                <span className="pref-sub">Anonymous execution time (ms) and token throughput diagnostics.</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={performanceEnabled}
                  onChange={(e) => setPerformanceEnabled(e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>

            <div className="pref-item">
              <div className="pref-info">
                <span className="pref-name">Developer Analytics</span>
                <span className="pref-sub">Anonymous code snippet copy frequency and API usage telemetry.</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={analyticsEnabled}
                  onChange={(e) => setAnalyticsEnabled(e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>
        )}

        <div className="cookie-actions-row">
          {showPreferences ? (
            <>
              <button className="cookie-btn-secondary" onClick={() => setShowPreferences(false)}>
                Back
              </button>
              <button className="cookie-btn-primary" onClick={handleSaveCustom}>
                Save Preferences
              </button>
            </>
          ) : (
            <>
              <button className="cookie-btn-link" onClick={() => setShowPreferences(true)}>
                Customize
              </button>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button className="cookie-btn-secondary" onClick={handleDeclineNonEssential}>
                  Decline Optional
                </button>
                <button className="cookie-btn-primary" onClick={handleAcceptAll}>
                  Accept All
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
