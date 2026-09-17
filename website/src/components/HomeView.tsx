import React, { useState, useRef, useEffect } from 'react';

interface HomeViewProps {
  onSendPrompt: (prompt: string) => void;
  onOpenSidebar: () => void;
  isSidebarOpen: boolean;
}

export const HomeView: React.FC<HomeViewProps> = ({ onSendPrompt, onOpenSidebar, isSidebarOpen }) => {
  const [prompt, setPrompt] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!prompt.trim()) return;
    onSendPrompt(prompt.trim());
  };

  const suggestionPrompts = [
    {
      category: '💬 Conversational & Identity',
      prompt: 'Hi COLLISION, what can you do?',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      )
    },
    {
      category: '🧮 Exact Math & Calculations',
      prompt: 'What is 45 * 128 + 15% of 850?',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="4" y1="9" x2="20" y2="9"></line>
          <line x1="4" y1="15" x2="20" y2="15"></line>
          <line x1="10" y1="3" x2="8" y2="21"></line>
          <line x1="16" y1="3" x2="14" y2="21"></line>
        </svg>
      )
    },
    {
      category: '🔬 Science & Open Knowledge',
      prompt: 'What is quantum computing and how does it work?',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      )
    },
    {
      category: '🏛️ History & Culture',
      prompt: 'Who was Albert Einstein and what was his major discovery?',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path>
        </svg>
      )
    }
  ];

  return (
    <div className="home-view">
      {!isSidebarOpen && (
        <button 
          className="sidebar-toggle-btn"
          onClick={onOpenSidebar}
          style={{ position: 'absolute', top: 16, left: 16 }}
          title="Open sidebar"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="12" x2="21" y2="12"></line>
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <line x1="3" y1="18" x2="21" y2="18"></line>
          </svg>
        </button>
      )}

      <h1 className="home-greeting" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span>Hello, I'm COLLISION.</span>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" style={{ filter: 'drop-shadow(0 0 8px #A58BFF)' }}>
          <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z" fill="url(#sparkleGrad)" />
          <defs>
            <linearGradient id="sparkleGrad" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
              <stop stopColor="#A58BFF" />
              <stop offset="1" stopColor="#8B7CF6" />
            </linearGradient>
          </defs>
        </svg>
      </h1>
      <p className="home-subtitle">Hierarchical Hybrid Intelligence System · Conversational, Math, Live Web & Grounded AI.</p>

      <div className="composer-box">
        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder="Ask COLLISION anything (conversations, calculations, facts, science)..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
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
              Hierarchical Hybrid AI
            </span>
          </div>
          <button 
            className="send-btn"
            disabled={!prompt.trim()}
            onClick={handleSubmit}
            title="Send prompt"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="19" x2="12" y2="5"></line>
              <polyline points="5 12 12 5 19 12"></polyline>
            </svg>
          </button>
        </div>
      </div>

      <div className="suggestions-grid">
        {suggestionPrompts.map((item, idx) => (
          <div 
            key={idx} 
            className="suggestion-card"
            onClick={() => onSendPrompt(item.prompt)}
          >
            <div className="suggestion-title">
              {item.icon}
              <span>{item.category}</span>
            </div>
            <div className="suggestion-prompt">{item.prompt}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
