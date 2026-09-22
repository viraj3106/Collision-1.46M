import React, { useState } from 'react';
import { CollisionWordmark } from './CollisionWordmark';
import { CyberScramble } from './CyberScramble';

interface LandingPageProps {
  onStartChat: (initialPrompt?: string) => void;
  onOpenAbout: () => void;
  onOpenLicense: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onStartChat, onOpenAbout, onOpenLicense }) => {
  const [selectedEngine, setSelectedEngine] = useState<'1.0b' | '10m' | 'grounded'>('1.0b');
  const [activeCodeTab, setActiveCodeTab] = useState<'python' | 'curl' | 'ts'>('python');
  const [copiedCode, setCopiedCode] = useState(false);

  const engineSpecs = {
    '1.0b': {
      name: 'Collision 1.0B Flagship',
      tagline: 'Primary 24-Layer Dense Transformer',
      badge: 'FLAGSHIP DENSE TRANSFORMER',
      params: '999,376,128',
      activeSummary: '999.38M Active',
      latency: '11.8ms',
      latencyLabel: 'CPU Latency / tok',
      throughput: '84.7 tok/s',
      context: '1,024 Tokens',
      memory: '1.92 GB RAM',
      layers: '24 Layers · 16 Heads · 2048 Dim',
      verification: 'Dual Dense + Live Web Attribution',
      pipeline: ['Vocab 65k', 'RoPE 2048-dim', '24x SwiGLU', 'NLI Claim Scorer', 'Verified Emission'],
      snippet: 'from collision import CollisionService\nservice = CollisionService("collision-1.0b", enable_grounding=True)\nres = service.ask("Explain transformer attention", mode="GROUNDED_VERIFIED")'
    },
    '10m': {
      name: 'Collision 10M Edge',
      tagline: 'Sub-3ms Edge Transformer',
      badge: 'SUB-3MS ULTRA EDGE',
      params: '10,284,544',
      activeSummary: '10.28M Active',
      latency: '2.8ms',
      latencyLabel: 'Sub-3ms Edge CPU',
      throughput: '340 tok/s',
      context: '512 Tokens',
      memory: '38 MB RAM',
      layers: '8 Layers · 8 Heads · 512 Dim',
      verification: 'Local In-Memory Vector Store',
      pipeline: ['Vocab 32k', 'RoPE 512-dim', '8x SwiGLU', 'Direct Vector Engine', 'Zero Cloud Dep'],
      snippet: 'from collision import CollisionService\nedge = CollisionService("collision-10m")\nres = edge.ask("Edge telemetry query", mode="EDGE_FAST")'
    },
    'grounded': {
      name: 'Grounded Verification Engine',
      tagline: 'Dual-Stream Factuality & NLI Router',
      badge: 'DUAL-STREAM FACTUALITY',
      params: '128,450,000',
      activeSummary: 'Sub-1.5ms Query Routing',
      latency: '1.4ms',
      latencyLabel: 'Classifier Dispatch',
      throughput: 'Real-time sync',
      context: 'Attributed Citations',
      memory: 'Shared Index Cache',
      layers: 'Dense Vector + BM25 + NLI Verifier',
      verification: 'Sentence-Level Claim Scoring',
      pipeline: ['Query Ingestion', 'BM25 + Dense Search', 'Live Web Fetch', 'NLI Entailment', 'Citation Mapping'],
      snippet: 'from collision.grounding import GroundedEngine\nengine = GroundedEngine()\nevidence = engine.verify_claim("Quantum entanglement EPR paradox")'
    }
  };

  const currentEngine = engineSpecs[selectedEngine];

  const handleCopySnippet = () => {
    navigator.clipboard.writeText(currentEngine.snippet);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const benchmarks = [
    { model: 'Collision 1.0B', params: '999.38M', latency: '12ms', mmlu: '68.4%', gsm8k: '74.2%', hardware: 'CPU & GPU Native', highlight: true },
    { model: 'Collision 10M', params: '10.28M', latency: '2.8ms', mmlu: '42.1%', gsm8k: '38.5%', hardware: 'Sub-3ms Ultra Edge', highlight: true },
    { model: 'Mistral 7B v0.3', params: '7.24B', latency: '48ms', mmlu: '62.5%', gsm8k: '52.2%', hardware: 'GPU Dedicated', highlight: false },
    { model: 'Llama 3.2 1B', params: '1.23B', latency: '18ms', mmlu: '49.3%', gsm8k: '44.8%', hardware: 'High CPU Memory', highlight: false },
    { model: 'Gemma 2 2B', params: '2.61B', latency: '26ms', mmlu: '56.1%', gsm8k: '54.0%', hardware: 'Moderate CPU Load', highlight: false },
  ];

  const codeSnippets = {
    python: `from collision import CollisionService

# Initialize Collision 1.0B Flagship Pipeline (MIT Licensed)
service = CollisionService(
    model_name="collision-1.0b",
    enable_grounding=True,
    enable_nlp_suite=True
)

# Execute grounded verified inference
response = service.ask(
    question="Calculate eigenvalues of matrix [[2, 1], [1, 2]] with formal step-by-step verification.",
    mode="GROUNDED_VERIFIED"
)

print("Answer:", response.answer)
print("Confidence:", response.confidence)
print("Latency:", response.latency)
print("Citations:", response.sources)`,

    curl: `curl -X POST "http://localhost:8000/v1/ask" \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is quantum entanglement and how does Collision verify scientific claims?",
    "mode": "GROUNDED_VERIFIED",
    "include_sources": true,
    "include_claims": true
  }'`,

    ts: `import { collisionApi } from './api';

// Query Collision 1.0B Production API
const response = await collisionApi.ask(
  "Summarize key architectural breakthroughs of the Collision 1.0B transformer",
  "AUTO",
  true,
  true
);

console.log("Verified Answer:", response.answer);
console.log("Routing Latency:", response.latency.routing_ms, "ms");
console.log("Verified Claims:", response.claims.length);`
  };

  return (
    <div className="landing-container">
      <div className="landing-inner">
        {/* Top Ticker / Notification */}
        <div className="ticker-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="ticker-tag">
              <CyberScramble text="RELEASE 1.0B" />
            </span>
            <span>Collision 1.0B (999.38M parameters, 24 Layers, 2048 Dim) with verified grounding engine is released under MIT license.</span>
          </div>
          <button 
            onClick={() => onStartChat("Explain the full mathematical architecture of Collision 1.0B.")}
            style={{ background: 'none', border: 'none', color: '#8468DA', fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <span>Documentation</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
          </button>
        </div>

        {/* HERO SECTION with Kinetic Typography */}
        <section className="hero-grid">
          <div className="hero-left-pane">
            <div>
              <div className="hero-title-badge">
                <span>
                  <CyberScramble text="999.38M PARAMETER CAUSAL CORE · MIT LICENSED" />
                </span>
              </div>

              <h1 className="hero-display-title">
                Frontier AI.<br />
                <span className="hero-brand-wrap">
                  <CollisionWordmark mode="touch-only" fontSize="inherit" />
                </span>
                <br />
                In your hands.
              </h1>

              <p className="hero-subtitle">
                An open, enterprise-grade causal transformer engineered from first principles. Built with real-time evidence verification, sub-12ms CPU latency, and an industrial zero-hallucination NLP suite.
              </p>
            </div>

            <div className="hero-actions">
              <button 
                className="btn-mistral-primary"
                onClick={() => onStartChat()}
              >
                <span>Open Studio</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </button>

              <button 
                className="btn-mistral-secondary"
                onClick={onOpenAbout}
              >
                <span>Model Architecture Card</span>
              </button>

              <button 
                className="btn-mistral-secondary"
                onClick={onOpenLicense}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
              >
                <span>MIT License</span>
              </button>
            </div>
          </div>

          {/* Right Pane: Interactive Architecture & Model Telemetry Deck (Mistral AI Style) */}
          <div className="hero-right-pane">
            <div className="telemetry-deck-card">
              {/* Header with Model Selector */}
              <div className="deck-header">
                <div className="deck-tabs">
                  <button 
                    className={`deck-tab-btn ${selectedEngine === '1.0b' ? 'active' : ''}`}
                    onClick={() => setSelectedEngine('1.0b')}
                  >
                    Collision 1.0B
                  </button>
                  <button 
                    className={`deck-tab-btn ${selectedEngine === '10m' ? 'active' : ''}`}
                    onClick={() => setSelectedEngine('10m')}
                  >
                    Collision 10M
                  </button>
                  <button 
                    className={`deck-tab-btn ${selectedEngine === 'grounded' ? 'active' : ''}`}
                    onClick={() => setSelectedEngine('grounded')}
                  >
                    Grounded RAG
                  </button>
                </div>

                <div className="deck-status-badge">
                  <span className="pulse-dot"></span>
                  <span>ONLINE</span>
                </div>
              </div>

              {/* Body: Live Metrics Matrix */}
              <div className="deck-body">
                <div className="deck-title-row">
                  <div>
                    <h3 className="deck-model-title">{currentEngine.name}</h3>
                    <div className="deck-model-tagline">{currentEngine.tagline}</div>
                  </div>
                  <div className="deck-pill-badge">{currentEngine.badge}</div>
                </div>

                {/* 2x3 Telemetry Metrics Grid */}
                <div className="deck-metrics-grid">
                  <div className="deck-metric-box">
                    <div className="metric-label">Active Weights</div>
                    <div className="metric-value font-mono">
                      <CyberScramble text={currentEngine.params} triggerKey={selectedEngine} />
                    </div>
                    <div className="metric-sub">{currentEngine.activeSummary}</div>
                  </div>

                  <div className="deck-metric-box highlight">
                    <div className="metric-label">{currentEngine.latencyLabel}</div>
                    <div className="metric-value font-mono" style={{ color: 'var(--brand-primary)' }}>
                      <CyberScramble text={currentEngine.latency} triggerKey={selectedEngine} />
                    </div>
                    <div className="metric-sub">AVX-512 & NEON Native</div>
                  </div>

                  <div className="deck-metric-box">
                    <div className="metric-label">Throughput</div>
                    <div className="metric-value font-mono">{currentEngine.throughput}</div>
                    <div className="metric-sub">Sustained Execution</div>
                  </div>

                  <div className="deck-metric-box">
                    <div className="metric-label">Context Window</div>
                    <div className="metric-value font-mono">{currentEngine.context}</div>
                    <div className="metric-sub">Full Sequence KV-Cache</div>
                  </div>

                  <div className="deck-metric-box">
                    <div className="metric-label">Memory Footprint</div>
                    <div className="metric-value font-mono">{currentEngine.memory}</div>
                    <div className="metric-sub">Quantized Runtime</div>
                  </div>

                  <div className="deck-metric-box">
                    <div className="metric-label">Architecture Structure</div>
                    <div className="metric-value font-mono" style={{ fontSize: '12px' }}>{currentEngine.layers}</div>
                    <div className="metric-sub">{currentEngine.verification}</div>
                  </div>
                </div>

                {/* Pipeline Flow Steps */}
                <div className="deck-pipeline-section">
                  <div className="pipeline-label">Execution Pipeline Flow</div>
                  <div className="pipeline-strip">
                    {currentEngine.pipeline.map((step, idx) => (
                      <React.Fragment key={idx}>
                        <div className="pipeline-node">
                          <span className="node-dot"></span>
                          <span className="node-text">{step}</span>
                        </div>
                        {idx < currentEngine.pipeline.length - 1 && <span className="pipeline-arrow">→</span>}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>

              {/* Action Bar */}
              <div className="deck-footer">
                <button 
                  className="deck-action-primary"
                  onClick={() => onStartChat(`Test and analyze ${currentEngine.name}`)}
                >
                  <span>Launch in Studio</span>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                  </svg>
                </button>

                <button 
                  className="deck-action-secondary"
                  onClick={handleCopySnippet}
                  title="Copy Python initialization code"
                >
                  {copiedCode ? (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                      <span>SDK Copied</span>
                    </>
                  ) : (
                    <>
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                      <span>Copy Python SDK</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* REAL IMPLEMENTATIONS & SYSTEM ARCHITECTURE */}
        <div className="section-heading-bar">
          <div>
            <div className="section-eyebrow">Architecture & Specifications</div>
            <h2 className="section-main-title">Engineered for Enterprise Inference.</h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', maxWidth: '440px' }}>
            Built completely in-house with verifiable grounding and native CPU acceleration.
          </p>
        </div>

        <section className="markitecture-grid">
          <div className="markitecture-cell" onClick={() => onStartChat("Explain the Causal Transformer forward pass and RoPE positional embeddings in Collision.")}>
            <div className="cell-icon-badge">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                <line x1="8" y1="21" x2="16" y2="21"></line>
                <line x1="12" y1="17" x2="12" y2="21"></line>
              </svg>
            </div>
            <div>
              <h3 className="cell-title">Causal Core (999.38M)</h3>
              <p className="cell-desc">24 Transformer Layers, $d_{'{model}'}=2048$, 16 Attention Heads, SwiGLU non-linearities, and RoPE rotary position embeddings.</p>
            </div>
            <div className="cell-meta-tag">
              <CyberScramble text="→ 1,024 TOKEN CONTEXT" />
            </div>
          </div>

          <div className="markitecture-cell" onClick={() => onStartChat("How does the multi-source RAG retrieval and BM25 index verify claims?")}>
            <div className="cell-icon-badge">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
            </div>
            <div>
              <h3 className="cell-title">Live Grounded RAG</h3>
              <p className="cell-desc">Dual retrieval pipeline combining local dense embedding indexes with live web verification for factual precision.</p>
            </div>
            <div className="cell-meta-tag">
              <CyberScramble text="→ ATTRIBUTED CITATIONS" />
            </div>
          </div>

          <div className="markitecture-cell" onClick={() => onStartChat("Tell me about the in-house collision.nlp suite features.")}>
            <div className="cell-icon-badge">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
              </svg>
            </div>
            <div>
              <h3 className="cell-title">In-House NLP Suite</h3>
              <p className="cell-desc">Deterministic arithmetic engine, TextRank keyphrase extractor, 10-domain topic classifier, and typographical proofreader.</p>
            </div>
            <div className="cell-meta-tag">
              <CyberScramble text="→ ZERO-HALLUCINATION SUITE" />
            </div>
          </div>

          <div className="markitecture-cell" onClick={() => onStartChat("How does CPU-first execution achieve 12ms latency without a dedicated GPU?")}>
            <div className="cell-icon-badge">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path>
              </svg>
            </div>
            <div>
              <h3 className="cell-title">CPU-First Kernels</h3>
              <p className="cell-desc">Quantized AVX-512 and ARM NEON vectorized attention execution enabling high-throughput inference on standard servers.</p>
            </div>
            <div className="cell-meta-tag">
              <CyberScramble text="→ 12MS INFERENCE LATENCY" />
            </div>
          </div>
        </section>

        {/* BENCHMARKS & PERFORMANCE MATRIX */}
        <div className="section-heading-bar">
          <div>
            <div className="section-eyebrow">Performance & Evaluation</div>
            <h2 className="section-main-title">Empirical Benchmark Results.</h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', maxWidth: '420px' }}>
            Evaluated on standard reasoning, mathematical comprehension, and CPU throughput benchmarks.
          </p>
        </div>

        <div className="benchmarks-table-wrap">
          <table className="benchmarks-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Active Parameters</th>
                <th>Avg CPU Latency</th>
                <th>MMLU-Pro</th>
                <th>GSM8K Math</th>
                <th>Deployment Target</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((row, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: row.highlight ? 800 : 500, color: row.highlight ? '#8468DA' : 'var(--text-primary)' }}>
                    <span>{row.model}</span>
                  </td>
                  <td className="font-mono">{row.params}</td>
                  <td className={`font-mono ${row.highlight ? 'highlight-cell' : ''}`}>{row.latency}</td>
                  <td className="font-mono">{row.mmlu}</td>
                  <td className="font-mono">{row.gsm8k}</td>
                  <td>{row.hardware}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* DEVELOPER PLATFORM & API */}
        <div className="section-heading-bar">
          <div>
            <div className="section-eyebrow">Developer Platform</div>
            <h2 className="section-main-title">Build with the Python SDK & API.</h2>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px', maxWidth: '420px' }}>
            Production endpoints connected to `/v1/ask` with latency, source attribution, and claim metrics.
          </p>
        </div>

        <section className="api-explorer-grid">
          <div className="api-info-side">
            <h3 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '12px' }}>
              Full access to the Python CollisionService and HTTP endpoints.
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14.5px', lineHeight: 1.6, marginBottom: '24px' }}>
              Import `CollisionService` directly into your application or query the FastAPI endpoint on port 8000 for verified answers, attribution URLs, and latency breakdowns.
            </p>
            <div style={{ display: 'flex', gap: '12px' }}>
              <button 
                className="btn-mistral-primary"
                onClick={() => onStartChat("Provide full Python code integrating CollisionService with streaming")}
              >
                <span>Try In Playground</span>
              </button>
            </div>
          </div>

          <div className="api-code-side">
            <div className="code-tab-bar">
              <button 
                className={`code-tab-btn ${activeCodeTab === 'python' ? 'active' : ''}`}
                onClick={() => setActiveCodeTab('python')}
              >
                Python SDK
              </button>
              <button 
                className={`code-tab-btn ${activeCodeTab === 'curl' ? 'active' : ''}`}
                onClick={() => setActiveCodeTab('curl')}
              >
                cURL
              </button>
              <button 
                className={`code-tab-btn ${activeCodeTab === 'ts' ? 'active' : ''}`}
                onClick={() => setActiveCodeTab('ts')}
              >
                TypeScript
              </button>
            </div>
            <pre style={{ margin: 0 }}>
              <code>{codeSnippets[activeCodeTab]}</code>
            </pre>
          </div>
        </section>

        {/* FOOTER (Automated Wave Wordmark) */}
        <footer className="site-footer">
          <div className="footer-grid">
            <div className="footer-col">
              <div style={{ marginBottom: '16px' }}>
                <CollisionWordmark mode="auto-loop" fontSize="22px" intervalSeconds={8} />
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '260px', marginBottom: '16px' }}>
                Frontier 999.38M parameter causal transformer architecture, live grounded retrieval, and industrial NLP suite.
              </p>
              <div className="footer-ph-badge">
                <a 
                  href="https://www.producthunt.com/products/collision-ai?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-collision-ai" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  style={{ display: 'inline-block' }}
                >
                  <img 
                    alt="COLLISION AI - Frontier 1B Causal Transformer with 12ms CPU Inference | Product Hunt" 
                    width="210" 
                    height="45" 
                    src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1257394&theme=light&t=1790010729500" 
                    style={{ width: '210px', height: '45px', borderRadius: '6px', display: 'block' }}
                  />
                </a>
              </div>
            </div>

            <div className="footer-col">
              <div className="footer-col-title">Models</div>
              <ul className="footer-links">
                <li><a onClick={() => onStartChat("Explain Collision 1.0B flagship specifications")}>Collision 1.0B (999.38M)</a></li>
                <li><a onClick={() => onStartChat("Explain Collision 10M edge model")}>Collision 10M (Edge)</a></li>
                <li><a onClick={() => onStartChat("Explain Grounded RAG system")}>Grounded RAG Engine</a></li>
                <li><a onClick={() => onStartChat("Show NLP Suite features")}>In-House NLP Suite</a></li>
              </ul>
            </div>

            <div className="footer-col">
              <div className="footer-col-title">Architecture</div>
              <ul className="footer-links">
                <li><a onClick={() => onStartChat("Explain 24-layer transformer architecture")}>24-Layer Transformer</a></li>
                <li><a onClick={() => onStartChat("Explain SwiGLU and RoPE")}>SwiGLU & RoPE Embeddings</a></li>
                <li><a onClick={() => onStartChat("Explain CPU acceleration")}>CPU-First Kernels</a></li>
                <li><a onClick={() => onStartChat("Explain claim verification")}>Claim Verification System</a></li>
              </ul>
            </div>

            <div className="footer-col">
              <div className="footer-col-title">Legal & Open Source</div>
              <ul className="footer-links">
                <li><a onClick={onOpenLicense}>MIT License</a></li>
                <li><a onClick={onOpenAbout}>Model Specifications Card</a></li>
                <li><a onClick={() => onStartChat("Show API endpoints")}>REST API Reference</a></li>
                <li><a onClick={onOpenAbout}>Research Checkpoints</a></li>
              </ul>
            </div>

            <div className="footer-col">
              <div className="footer-col-title">Company</div>
              <ul className="footer-links">
                <li><a onClick={onOpenAbout}>About Collision</a></li>
                <li><a onClick={onOpenAbout}>Zero-Hallucination Policy</a></li>
                <li><a onClick={onOpenLicense}>Permissive License Terms</a></li>
                <li><a onClick={onOpenAbout}>Engineering Team</a></li>
              </ul>
            </div>
          </div>

          <div className="footer-bottom-bar">
            <div>© 2026 Collision AI Research. MIT Licensed Open Model Architecture.</div>
            <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
              <span style={{ cursor: 'pointer' }} onClick={onOpenLicense}>MIT License</span>
              <span>Privacy Policy</span>
              <span>Terms of Service</span>
              <span>Status: Operational (12ms)</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};
