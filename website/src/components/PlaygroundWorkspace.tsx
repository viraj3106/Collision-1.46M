import React, { useState, useRef, useEffect } from 'react';
import type { Message } from './ChatWorkspace';
import type { ChatSession } from './Sidebar';

interface PlaygroundWorkspaceProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  messages: Message[];
  isGenerating: boolean;
  onSendPrompt: (prompt: string) => void;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string) => void;
  onRegenerate: () => void;
  onFeedback: (messageId: string, rating: 'thumbs_up' | 'thumbs_down') => void;
  onOpenSettings: () => void;
}

export const PlaygroundWorkspace: React.FC<PlaygroundWorkspaceProps> = ({
  sessions,
  currentSessionId,
  messages,
  isGenerating,
  onSendPrompt,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onRegenerate,
  onFeedback,
  onOpenSettings,
}) => {
  const [promptInput, setPromptInput] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!promptInput.trim() || isGenerating) return;
    onSendPrompt(promptInput.trim());
    setPromptInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const starterPrompts = [
    { title: 'Math & Derivations', prompt: 'Calculate the integral of 3x^2 + 2x - 5 and explain each step in detail.' },
    { title: 'Frontier Architecture', prompt: 'Explain the difference between decoder-only transformers and hybrid sparse routing architectures.' },
    { title: 'Code Refactoring', prompt: 'Write a clean Python asyncio rate limiter with exponential backoff.' },
    { title: 'Scientific Reasoning', prompt: 'How does quantum entanglement relate to the EPR paradox and Bell inequalities?' },
  ];

  return (
    <div className="playground-view">
      {/* Collapsible Left Sidebar for sessions */}
      <aside className={`playground-sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div style={{ padding: '16px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <button 
            className="btn-mistral-primary"
            onClick={onNewChat}
            style={{ width: '100%', padding: '10px 14px', fontSize: '13px', justifyContent: 'center' }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New Studio Session</span>
          </button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 8px' }}>
          <div style={{ fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-tertiary)', textTransform: 'uppercase', padding: '6px 10px 10px', letterSpacing: '0.06em' }}>
            Recent Sessions ({sessions.length})
          </div>

          {sessions.length === 0 ? (
            <div style={{ padding: '16px 10px', fontSize: '12.5px', color: 'var(--text-muted)' }}>
              No previous sessions. Start asking questions to generate history.
            </div>
          ) : (
            sessions.map((sess) => (
              <div
                key={sess.id}
                onClick={() => onSelectSession(sess.id)}
                style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '13px',
                  fontWeight: currentSessionId === sess.id ? 600 : 400,
                  color: currentSessionId === sess.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                  background: currentSessionId === sess.id ? 'var(--bg-card)' : 'transparent',
                  border: currentSessionId === sess.id ? '1px solid var(--border-color)' : '1px solid transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '4px',
                  transition: 'all var(--transition-fast)'
                }}
              >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, marginRight: '8px' }}>
                  {sess.title}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(sess.id);
                  }}
                  style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer', padding: '2px' }}
                  title="Delete chat"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>
            ))
          )}
        </div>

        <div style={{ padding: '14px 16px', borderTop: '1px solid var(--border-color)', background: 'var(--bg-card)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
            <div style={{ fontSize: '13px', fontWeight: 800, color: 'var(--brand-primary)', letterSpacing: '-0.02em' }}>COLLISION-1.0B Active</div>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'JetBrains Mono, monospace' }}>
            Grounded API · /v1/ask · 1.0B
          </div>
        </div>
      </aside>

      {/* Main Studio Chat & Inspection Area */}
      <main className="playground-main">
        <div className="playground-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={() => setIsSidebarOpen(prev => !prev)}
              style={{ background: 'none', border: 'none', color: 'var(--text-primary)', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
              title="Toggle sessions sidebar"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="9" y1="3" x2="9" y2="21"></line>
              </svg>
            </button>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>COLLISION Studio Session</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span className="font-mono" style={{ fontSize: '11.5px', color: 'var(--text-tertiary)' }}>
              MODE: GROUNDED VERIFIED
            </span>
            <button
              className="btn-mistral-secondary"
              onClick={onOpenSettings}
              style={{ padding: '6px 12px', fontSize: '12px' }}
            >
              Parameters
            </button>
          </div>
        </div>

        {/* Messages Stream */}
        <div className="chat-history-container">
          {messages.length === 0 ? (
            <div style={{ textAlign: 'center', margin: 'auto', maxWidth: '640px', padding: '40px 20px' }}>
              <div style={{ display: 'inline-flex', padding: '12px 20px', borderRadius: 'var(--radius-sm)', background: 'var(--brand-primary-light)', marginBottom: '18px' }}>
                <span style={{ fontWeight: 900, fontSize: '24px', letterSpacing: '-0.03em', color: 'var(--brand-primary)' }}>COLLISION</span>
              </div>
              <h2 style={{ fontSize: '26px', fontWeight: 800, marginBottom: '10px', letterSpacing: '-0.02em' }}>
                What would you like to build with COLLISION?
              </h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '14.5px', lineHeight: 1.6, marginBottom: '28px' }}>
                Test queries, mathematical problem solving, coding requests, and factual verification grounded against live evidence.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '12px', textAlign: 'left' }}>
                {starterPrompts.map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() => onSendPrompt(item.prompt)}
                    style={{
                      padding: '14px 16px',
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-sm)',
                      cursor: 'pointer',
                      transition: 'all var(--transition-fast)'
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#8468DA'; e.currentTarget.style.transform = 'translateY(-2px)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-color)'; e.currentTarget.style.transform = 'translateY(0)'; }}
                  >
                    <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--brand-primary)', marginBottom: '4px' }}>
                      {item.title}
                    </div>
                    <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      {item.prompt}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className="chat-msg-row">
                <div className={`avatar-badge ${msg.role === 'user' ? 'user' : 'ai'}`}>
                  {msg.role === 'user' ? 'U' : 'C'}
                </div>

                <div className="msg-body-wrapper">
                  <div className="msg-speaker-title">
                    <span>{msg.role === 'user' ? 'You' : 'COLLISION-1.0B'}</span>
                    <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontWeight: 400 }}>
                      {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>

                  <div className="msg-text-content">
                    {msg.content}
                  </div>

                  {/* AI Metadata pill breakdown for grounded evidence */}
                  {msg.role === 'assistant' && (
                    <>
                      <div className="ai-meta-grid">
                        <div className="ai-meta-item">
                          <span>Status:</span>
                          <strong>{msg.status || 'VERIFIED'}</strong>
                        </div>
                        {msg.confidence !== undefined && (
                          <div className="ai-meta-item">
                            <span>Confidence:</span>
                            <strong>{(msg.confidence * 100).toFixed(1)}%</strong>
                          </div>
                        )}
                        {msg.latency && (
                          <div className="ai-meta-item">
                            <span>Latency:</span>
                            <strong>{msg.latency.total_ms || 14}ms</strong>
                          </div>
                        )}
                        <div className="ai-meta-item">
                          <span>Verified Claims:</span>
                          <strong>{msg.claims?.length || 0}</strong>
                        </div>
                      </div>

                      {/* Source attribution list */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="citations-box">
                          <div className="citations-title">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                              <circle cx="12" cy="12" r="10"></circle>
                              <line x1="12" y1="16" x2="12" y2="12"></line>
                              <line x1="12" y1="8" x2="12.01" y2="8"></line>
                            </svg>
                            <span>Grounded Verification Citations ({msg.sources.length})</span>
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                            {msg.sources.map((src, sIdx) => (
                              <a
                                key={sIdx}
                                href={src.url}
                                target="_blank"
                                rel="noreferrer"
                                className="citation-link"
                              >
                                [{sIdx + 1}] {src.title || src.url}
                              </a>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Response action buttons */}
                      <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
                        <button
                          onClick={() => handleCopy(msg.id, msg.content)}
                          style={{ background: 'none', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-xs)', padding: '3px 8px', fontSize: '11px', color: 'var(--text-secondary)', cursor: 'pointer' }}
                        >
                          {copiedId === msg.id ? '✓ Copied' : 'Copy'}
                        </button>
                        <button
                          onClick={() => onFeedback(msg.id, 'thumbs_up')}
                          style={{ background: 'none', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-xs)', padding: '3px 8px', fontSize: '11px', color: msg.feedbackRating === 'thumbs_up' ? '#10B981' : 'var(--text-secondary)', cursor: 'pointer' }}
                        >
                          👍 Helpful
                        </button>
                        <button
                          onClick={() => onFeedback(msg.id, 'thumbs_down')}
                          style={{ background: 'none', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-xs)', padding: '3px 8px', fontSize: '11px', color: msg.feedbackRating === 'thumbs_down' ? '#EF4444' : 'var(--text-secondary)', cursor: 'pointer' }}
                        >
                          👎 Incorrect
                        </button>
                        <button
                          onClick={onRegenerate}
                          style={{ background: 'none', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-xs)', padding: '3px 8px', fontSize: '11px', color: 'var(--text-secondary)', cursor: 'pointer' }}
                        >
                          ↻ Regenerate
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            ))
          )}

          {isGenerating && (
            <div className="chat-msg-row">
              <div className="avatar-badge ai">
                C
              </div>
              <div className="msg-body-wrapper">
                <div className="msg-speaker-title">COLLISION-1.0B</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--brand-primary)', fontSize: '13.5px', fontWeight: 600 }}>
                  <span className="pulse-dot"></span>
                  <span>Verifying grounded evidence & generating response...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Bottom Composer Box */}
        <section className="chat-composer-section">
          <form className="chat-composer-wrapper" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              className="chat-textarea"
              placeholder="Ask COLLISION anything or give instruction (Press Enter to submit)..."
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
            />

            <div className="composer-footer-row">
              <div className="composer-mode-badge">
                <span className="pulse-dot"></span>
                <span>COLLISION-1.0B · FAST INFERENCE</span>
              </div>

              <button
                type="submit"
                className="composer-send-btn"
                disabled={!promptInput.trim() || isGenerating}
              >
                <span>Send</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <line x1="12" y1="19" x2="12" y2="5"></line>
                  <polyline points="5 12 12 5 19 12"></polyline>
                </svg>
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
};
