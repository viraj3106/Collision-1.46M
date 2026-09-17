import React, { useState, useRef, useEffect } from 'react';
import type { SourceInfo, ClaimInfo, LatencyInfo } from '../api';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  status?: 'ANSWERED' | 'INSUFFICIENT_INFORMATION' | 'CONFLICT' | 'ERROR' | string;
  mode?: string;
  confidence?: number;
  sources?: SourceInfo[];
  claims?: ClaimInfo[];
  latency?: LatencyInfo;
  tokens?: {
    prompt_tokens: number;
    completion_tokens: number;
    latency_ms: number;
  };
  feedbackRating?: 'thumbs_up' | 'thumbs_down' | null;
}

interface ChatWorkspaceProps {
  messages: Message[];
  isGenerating: boolean;
  onSendPrompt: (prompt: string) => void;
  onRegenerate: () => void;
  onFeedback: (messageId: string, rating: 'thumbs_up' | 'thumbs_down') => void;
  onOpenSidebar: () => void;
  isSidebarOpen: boolean;
  onNewChat: () => void;
}

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  messages,
  isGenerating,
  onSendPrompt,
  onRegenerate,
  onFeedback,
  onOpenSidebar,
  isSidebarOpen,
  onNewChat
}) => {
  const [inputPrompt, setInputPrompt] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (!inputPrompt.trim() || isGenerating) return;
    onSendPrompt(inputPrompt.trim());
    setInputPrompt('');
  };

  const handleCopyText = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const renderInlineFormatted = (text: string) => {
    // 1. Split on inline code blocks
    const codeParts = text.split(/(`[^`]+`)/g);
    return codeParts.map((cPart, cIdx) => {
      if (cPart.startsWith('`') && cPart.endsWith('`') && cPart.length >= 2) {
        return <code key={cIdx} className="inline-code">{cPart.slice(1, -1)}</code>;
      }
      // 2. Split on bold tokens (**bold**)
      const boldParts = cPart.split(/(\*\*[^*]+\*\*)/g);
      return (
        <React.Fragment key={cIdx}>
          {boldParts.map((bPart, bIdx) => {
            if (bPart.startsWith('**') && bPart.endsWith('**') && bPart.length >= 4) {
              return <strong key={bIdx} className="markdown-bold">{bPart.slice(2, -2)}</strong>;
            }
            // 3. Split on italic tokens (*italic*)
            const italicParts = bPart.split(/(\*[^*]+\*)/g);
            return (
              <React.Fragment key={bIdx}>
                {italicParts.map((iPart, iIdx) => {
                  if (iPart.startsWith('*') && iPart.endsWith('*') && iPart.length >= 2 && !iPart.startsWith('**')) {
                    return <em key={iIdx} className="markdown-italic">{iPart.slice(1, -1)}</em>;
                  }
                  return iPart;
                })}
              </React.Fragment>
            );
          })}
        </React.Fragment>
      );
    });
  };

  const renderContent = (content: string) => {
    // Code block split parsing
    const parts = content.split(/(```[\s\S]*?```)/g);
    return parts.map((part, index) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const lines = part.slice(3, -3).trim().split('\n');
        let language = 'text';
        if (lines[0] && !lines[0].includes(' ') && lines[0].length < 15) {
          language = lines[0];
          lines.shift();
        }
        const codeText = lines.join('\n');
        const codeId = `code-${index}`;
        return (
          <div key={index} className="code-block">
            <div className="code-header">
              <span>{language}</span>
              <button 
                className="copy-code-btn"
                onClick={() => handleCopyText(codeId, codeText)}
              >
                {copiedId === codeId ? (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    Copied
                  </>
                ) : (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    Copy code
                  </>
                )}
              </button>
            </div>
            <div className="code-content">{codeText}</div>
          </div>
        );
      } else {
        // Handle paragraphs, lists, and headings
        const paragraphs = part.split('\n\n');
        return paragraphs.map((p, pIdx) => {
          const trimmed = p.trim();
          if (!trimmed) return null;

          // Check if heading (### or ##)
          if (trimmed.startsWith('### ') || trimmed.startsWith('## ')) {
            const hText = trimmed.replace(/^#{2,3}\s+/, '');
            return (
              <h4 key={`${index}-${pIdx}`} className="markdown-heading">
                {renderInlineFormatted(hText)}
              </h4>
            );
          }

          // Check if list of bullet items
          const lines = trimmed.split('\n');
          const isBulletList = lines.every(l => l.trim().startsWith('• ') || l.trim().startsWith('- ') || l.trim().startsWith('* '));

          if (isBulletList && lines.length > 0) {
            return (
              <ul key={`${index}-${pIdx}`} className="markdown-list">
                {lines.map((l, lIdx) => {
                  const cleanItem = l.trim().replace(/^[•\-\*]\s+/, '');
                  return (
                    <li key={lIdx} className="markdown-list-item">
                      <span className="bullet-dot">•</span>
                      <div className="bullet-content">{renderInlineFormatted(cleanItem)}</div>
                    </li>
                  );
                })}
              </ul>
            );
          }

          // Check if mixed lines with some bullets
          return (
            <div key={`${index}-${pIdx}`} className="markdown-paragraph">
              {lines.map((l, lIdx) => {
                const lineTrim = l.trim();
                if (lineTrim.startsWith('• ') || lineTrim.startsWith('- ')) {
                  const cleanItem = lineTrim.replace(/^[•\-]\s+/, '');
                  return (
                    <div key={lIdx} className="markdown-list-item" style={{ marginTop: '4px' }}>
                      <span className="bullet-dot">•</span>
                      <div className="bullet-content">{renderInlineFormatted(cleanItem)}</div>
                    </div>
                  );
                }
                return (
                  <div key={lIdx} style={{ margin: lIdx > 0 ? '4px 0 0 0' : '0' }}>
                    {renderInlineFormatted(l)}
                  </div>
                );
              })}
            </div>
          );
        });
      }
    });
  };

  return (
    <div className="main-workspace">
      {/* Workspace Top Bar (Inspired by Reference 2 - WRITER screen) */}
      <header className="workspace-header">
        <div className="workspace-title-group">
          {!isSidebarOpen && (
            <button 
              className="sidebar-toggle-btn"
              onClick={onOpenSidebar}
              title="Open sidebar"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            </button>
          )}
          <div className="header-model-pill">
            <span className="status-dot"></span>
            COLLISION-10M
            <span className="header-model-meta">· 10.28M parameters · CPU-first</span>
          </div>
        </div>

        <div className="workspace-actions">
          <button className="workspace-action-btn" onClick={onNewChat} title="Start new session">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>New Chat</span>
          </button>
        </div>
      </header>

      {/* Messages Stream */}
      <div className="messages-container">
        <div className="messages-content-wrapper">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-row ${msg.role}`}>
              <div className="message-header">
                {msg.role === 'user' ? (
                  'You'
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span>COLLISION</span>
                    {msg.mode && (
                      <span className="mode-tag font-mono">
                        {msg.mode}
                      </span>
                    )}
                    {msg.status === 'CONFLICT' && (
                      <span className="status-tag status-conflict">
                        Evidence Conflict
                      </span>
                    )}
                    {msg.status === 'INSUFFICIENT_INFORMATION' && (
                      <span className="status-tag status-insufficient">
                        Insufficient Grounding
                      </span>
                    )}
                    {msg.status === 'ERROR' && (
                      <span className="status-tag status-error">
                        Error
                      </span>
                    )}
                  </div>
                )}
              </div>

              {msg.role === 'user' ? (
                <div className="user-bubble">
                  {msg.content}
                </div>
              ) : (
                <div className="assistant-body">
                  {/* Status Banner for special states */}
                  {msg.status === 'CONFLICT' && (
                    <div className="status-banner banner-conflict">
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                      </svg>
                      <span><strong>Notice:</strong> The available verified sources contain conflicting information on this topic.</span>
                    </div>
                  )}

                  {msg.status === 'INSUFFICIENT_INFORMATION' && (
                    <div className="status-banner banner-insufficient">
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                      </svg>
                      <span><strong>Uncertainty Notice:</strong> Verified knowledge is insufficient to establish a conclusive answer.</span>
                    </div>
                  )}

                  {/* Main Response Text */}
                  {renderContent(msg.content)}

                  {/* Grounded Sources & Citations */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-container">
                      <div className="sources-header">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                        </svg>
                        <span>Verified Sources ({msg.sources.length})</span>
                      </div>
                      <div className="sources-grid">
                        {msg.sources.map((src, sIdx) => {
                          const hasUrl = Boolean(src.url && (src.url.startsWith('http://') || src.url.startsWith('https://')));
                          const CardElement = hasUrl ? 'a' : 'div';
                          return (
                            <CardElement
                              key={src.source_id || sIdx}
                              {...(hasUrl ? { href: src.url, target: '_blank', rel: 'noopener noreferrer' } : {})}
                              className={`source-chip ${hasUrl ? 'clickable' : ''}`}
                            >
                              <div className="source-chip-top">
                                <span className="source-badge font-mono">{src.source_type || 'evidence'}</span>
                                {src.retrieval_score != null && src.retrieval_score > 0 && (
                                  <span className="source-score font-mono">
                                    {(src.retrieval_score * 100).toFixed(0)}% match
                                  </span>
                                )}
                              </div>
                              <div className="source-chip-title" title={src.title || src.url}>
                                {src.title || src.url || `Evidence #${sIdx + 1}`}
                              </div>
                              {src.snippet && (
                                <div className="source-chip-snippet" title={src.snippet}>
                                  {src.snippet}
                                </div>
                              )}
                              {hasUrl && (
                                <div className="source-chip-url font-mono">
                                  {src.url.replace(/^https?:\/\//, '').slice(0, 38)}
                                </div>
                              )}
                            </CardElement>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Response Action Toolbar (Inspired by Reference 2) */}
                  <div className="response-toolbar">
                    <button 
                      className="toolbar-btn" 
                      onClick={() => handleCopyText(msg.id, msg.content)}
                      title="Copy response"
                    >
                      {copiedId === msg.id ? (
                        <>
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>
                          Copied
                        </>
                      ) : (
                        <>
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                          Copy
                        </>
                      )}
                    </button>

                    <button 
                      className="toolbar-btn"
                      onClick={onRegenerate}
                      title="Regenerate response"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
                      Regenerate
                    </button>

                    <button 
                      className={`toolbar-btn ${msg.feedbackRating === 'thumbs_up' ? 'active-positive' : ''}`}
                      onClick={() => onFeedback(msg.id, 'thumbs_up')}
                      title="Helpful"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path></svg>
                      {msg.feedbackRating === 'thumbs_up' && ' Helpful'}
                    </button>

                    <button 
                      className={`toolbar-btn ${msg.feedbackRating === 'thumbs_down' ? 'active-negative' : ''}`}
                      onClick={() => onFeedback(msg.id, 'thumbs_down')}
                      title="Not helpful"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path></svg>
                      {msg.feedbackRating === 'thumbs_down' && ' Feedback recorded'}
                    </button>

                    {(msg.latency?.total_ms != null || msg.tokens?.latency_ms != null) && (
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }} className="font-mono">
                        {(msg.latency?.total_ms ?? msg.tokens?.latency_ms ?? 0).toFixed(0)}ms
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="message-row assistant">
              <div className="message-header" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span>COLLISION</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style={{ filter: 'drop-shadow(0 0 4px #A58BFF)' }}>
                  <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z" fill="#A58BFF" />
                </svg>
              </div>
              <div className="gemini-thinking-glow-card">
                <div className="thinking-indicator">
                  <span className="pulse-dot"></span>
                  <span className="thinking-text-glow">COLLISION is thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Bottom Composer */}
      <div className="bottom-composer-wrapper">
        <div className="composer-box">
          <textarea
            ref={textareaRef}
            className="composer-input"
            placeholder="Ask COLLISION anything..."
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
          />
          <div className="composer-actions-bar">
            <div className="composer-tools">
              <span className="composer-tool-btn">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                COLLISION-10M
              </span>
            </div>
            <button 
              className="send-btn"
              disabled={!inputPrompt.trim() || isGenerating}
              onClick={handleSend}
              title="Send message"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
