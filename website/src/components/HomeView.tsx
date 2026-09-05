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
      category: 'Explain something',
      prompt: 'Explain how transformer models generate text sequences efficiently on CPU.',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      )
    },
    {
      category: 'Ask a technical question',
      prompt: 'What are key memory bandwidth optimizations for 10M parameter models?',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="16 18 22 12 16 6"></polyline>
          <polyline points="8 6 2 12 8 18"></polyline>
        </svg>
      )
    },
    {
      category: 'Explore an idea',
      prompt: 'Draft a architecture proposal for zero-dependency CPU AI workloads.',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"></path>
        </svg>
      )
    },
    {
      category: 'Test COLLISION',
      prompt: 'Summarize the core capabilities of COLLISION-10M architecture.',
      icon: (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
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

      <h1 className="home-greeting">Hello, I'm COLLISION.</h1>
      <p className="home-subtitle">Explore what a CPU-first language model can generate.</p>

      <div className="composer-box">
        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder="Ask COLLISION anything..."
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
              COLLISION-10M
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
