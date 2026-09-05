import React, { useState, useRef, useEffect } from 'react';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
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

  const renderContent = (content: string) => {
    // Basic code block split parsing for slick rendering
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
        // Handle inline code or line breaks
        const paragraphs = part.split('\n\n');
        return paragraphs.map((p, pIdx) => {
          if (!p.trim()) return null;
          return (
            <p key={`${index}-${pIdx}`} style={{ margin: '4px 0' }}>
              {p.split(/(`[^`]+`)/g).map((sub, sIdx) => {
                if (sub.startsWith('`') && sub.endsWith('`')) {
                  return <code key={sIdx} className="inline-code">{sub.slice(1, -1)}</code>;
                }
                return sub;
              })}
            </p>
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
                {msg.role === 'user' ? 'You' : 'COLLISION'}
              </div>

              {msg.role === 'user' ? (
                <div className="user-bubble">
                  {msg.content}
                </div>
              ) : (
                <div className="assistant-body">
                  {renderContent(msg.content)}

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

                    {msg.tokens && (
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                        {msg.tokens.latency_ms.toFixed(0)}ms
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="message-row assistant">
              <div className="message-header">COLLISION</div>
              <div className="thinking-indicator">
                <span className="pulse-dot"></span>
                <span>COLLISION is thinking...</span>
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
