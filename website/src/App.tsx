import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import type { ChatSession } from './components/Sidebar';
import { HomeView } from './components/HomeView';
import { ChatWorkspace } from './components/ChatWorkspace';
import type { Message } from './components/ChatWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { AboutModal } from './components/AboutModal';
import { collisionApi } from './api';

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionMessages, setSessionMessages] = useState<Record<string, Message[]>>({});
  const [isGenerating, setIsGenerating] = useState(false);

  // COLLISION Model Info & Links: Small AI. Built from scratch. TRY COLLISION → PLAYGROUND | GET API KEY → DEVELOPER PORTAL
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(120);

  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);

  // Load from localStorage on init
  useEffect(() => {
    try {
      const savedSessions = localStorage.getItem('collision_sessions');
      const savedMessages = localStorage.getItem('collision_messages');
      if (savedSessions) {
        setSessions(JSON.parse(savedSessions));
      }
      if (savedMessages) {
        setSessionMessages(JSON.parse(savedMessages));
      }
    } catch (e) {
      console.warn("Could not parse local storage history:", e);
    }
  }, []);

  // Save to localStorage when updated
  useEffect(() => {
    try {
      localStorage.setItem('collision_sessions', JSON.stringify(sessions));
      localStorage.setItem('collision_messages', JSON.stringify(sessionMessages));
    } catch (e) {
      console.warn("Could not save to local storage:", e);
    }
  }, [sessions, sessionMessages]);

  const handleNewChat = () => {
    setCurrentSessionId(null);
  };

  const handleSelectSession = (id: string) => {
    setCurrentSessionId(id);
  };

  const handleDeleteSession = (id: string) => {
    setSessions(prev => prev.filter(s => s.id !== id));
    setSessionMessages(prev => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    if (currentSessionId === id) {
      setCurrentSessionId(null);
    }
  };

  const handleClearAllHistory = () => {
    setSessions([]);
    setSessionMessages({});
    setCurrentSessionId(null);
    localStorage.removeItem('collision_sessions');
    localStorage.removeItem('collision_messages');
    setIsSettingsOpen(false);
  };

  const handleSendPrompt = async (promptText: string) => {
    if (!promptText.trim() || isGenerating) return;

    let targetSessionId = currentSessionId;
    
    // Create new session if none is selected
    if (!targetSessionId) {
      targetSessionId = `session-${Date.now()}`;
      const newSession: ChatSession = {
        id: targetSessionId,
        title: promptText.length > 32 ? promptText.slice(0, 32) + '...' : promptText,
        updatedAt: Date.now()
      };
      setSessions(prev => [newSession, ...prev]);
      setCurrentSessionId(targetSessionId);
    }

    // Append user message
    const userMsg: Message = {
      id: `msg-user-${Date.now()}`,
      role: 'user',
      content: promptText,
      timestamp: Date.now()
    };

    setSessionMessages(prev => ({
      ...prev,
      [targetSessionId!]: [...(prev[targetSessionId!] || []), userMsg]
    }));

    setIsGenerating(true);

    try {
      // Call Phase 99 Production Grounded Endpoint: POST /v1/ask via Collision API Client
      const response = await collisionApi.ask(promptText);

      const assistantMsg: Message = {
        id: `msg-ast-${Date.now()}`,
        role: 'assistant',
        content: response.answer || 'No answer could be generated from verified evidence.',
        timestamp: Date.now(),
        status: response.status,
        mode: response.mode,
        confidence: response.confidence,
        sources: response.sources || [],
        claims: response.claims || [],
        latency: response.latency,
        tokens: {
          prompt_tokens: 0,
          completion_tokens: 0,
          latency_ms: response.latency?.total_ms || 0
        }
      };

      setSessionMessages(prev => ({
        ...prev,
        [targetSessionId!]: [...(prev[targetSessionId!] || []), assistantMsg]
      }));
    } catch (err: any) {
      // Clean, user-friendly error without stack traces or leaks
      const errorMsg: Message = {
        id: `msg-err-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Could not complete request: ${err.message || 'COLLISION API unreachable'}. Please verify that the backend is running on port 8000.`,
        timestamp: Date.now(),
        status: 'ERROR',
        mode: 'ERROR'
      };

      setSessionMessages(prev => ({
        ...prev,
        [targetSessionId!]: [...(prev[targetSessionId!] || []), errorMsg]
      }));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRegenerate = () => {
    if (!currentSessionId) return;
    const currentMsgs = sessionMessages[currentSessionId] || [];
    const lastUserMsg = [...currentMsgs].reverse().find(m => m.role === 'user');
    if (lastUserMsg) {
      handleSendPrompt(lastUserMsg.content);
    }
  };

  const handleFeedback = async (messageId: string, rating: 'thumbs_up' | 'thumbs_down') => {
    if (!currentSessionId) return;

    // Update state locally first
    setSessionMessages(prev => {
      const msgs = prev[currentSessionId] || [];
      const updated = msgs.map(m => m.id === messageId ? { ...m, feedbackRating: rating } : m);
      return { ...prev, [currentSessionId]: updated };
    });

    // Find targeted message and previous user prompt
    const msgs = sessionMessages[currentSessionId] || [];
    const targetMsg = msgs.find(m => m.id === messageId);
    const targetIdx = msgs.findIndex(m => m.id === messageId);
    const promptMsg = targetIdx > 0 ? msgs[targetIdx - 1] : null;

    if (targetMsg && promptMsg) {
      try {
        await fetch(`${API_BASE_URL}/v1/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: "anonymous_chat_user",
            prompt: promptMsg.content,
            model: "collision-10m",
            response: targetMsg.content,
            rating: rating,
            feedback: "",
            category: "general",
            consent: true
          })
        });
      } catch (e) {
        console.warn("Failed to submit feedback to backend:", e);
      }
    }
  };

  const currentMessages = currentSessionId ? (sessionMessages[currentSessionId] || []) : [];

  return (
    <div className="app-container">
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenAbout={() => setIsAboutOpen(true)}
      />

      {/* Main View Switching: Home (Reference 1) vs Chat Workspace (Reference 2) */}
      {!currentSessionId || currentMessages.length === 0 ? (
        <HomeView
          onSendPrompt={handleSendPrompt}
          onOpenSidebar={() => setIsSidebarOpen(true)}
          isSidebarOpen={isSidebarOpen}
        />
      ) : (
        <ChatWorkspace
          messages={currentMessages}
          isGenerating={isGenerating}
          onSendPrompt={handleSendPrompt}
          onRegenerate={handleRegenerate}
          onFeedback={handleFeedback}
          onOpenSidebar={() => setIsSidebarOpen(true)}
          isSidebarOpen={isSidebarOpen}
          onNewChat={handleNewChat}
        />
      )}

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        temperature={temperature}
        setTemperature={setTemperature}
        maxTokens={maxTokens}
        setMaxTokens={setMaxTokens}
        onClearHistory={handleClearAllHistory}
      />

      <AboutModal
        isOpen={isAboutOpen}
        onClose={() => setIsAboutOpen(false)}
      />

      {isGenerating && <div className="gemini-screen-ambient-glow" />}
    </div>
  );
}
