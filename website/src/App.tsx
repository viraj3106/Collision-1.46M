import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import type { ChatSession } from './components/Sidebar';
import { HomeView } from './components/HomeView';
import { ChatWorkspace } from './components/ChatWorkspace';
import type { Message } from './components/ChatWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { AboutModal } from './components/AboutModal';

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
      // 1. Try real backend inference call to /v1/chat/generate
      let res = await fetch(`${API_BASE_URL}/v1/chat/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'collision-10m',
          prompt: promptText,
          max_tokens: maxTokens,
          temperature: temperature,
          top_k: 50,
          top_p: 0.9
        })
      });

      // 2. Fallback to /v1/playground/generate or /v1/generate if needed
      if (!res.ok && res.status === 404) {
        res = await fetch(`${API_BASE_URL}/v1/playground/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'collision-10m',
            prompt: promptText,
            max_tokens: maxTokens,
            temperature: temperature
          })
        });
      }

      if (res.ok) {
        const data = await res.json();
        const generatedText = data.text || 'No response generated.';
        
        const assistantMsg: Message = {
          id: `msg-ast-${Date.now()}`,
          role: 'assistant',
          content: generatedText,
          timestamp: Date.now(),
          tokens: {
            prompt_tokens: data.usage?.prompt_tokens || 0,
            completion_tokens: data.usage?.completion_tokens || 0,
            latency_ms: data.performance?.latency_ms || 0
          }
        };

        setSessionMessages(prev => ({
          ...prev,
          [targetSessionId!]: [...(prev[targetSessionId!] || []), assistantMsg]
        }));
      } else {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail?.message || errData.error?.message || `Server returned HTTP ${res.status}`);
      }
    } catch (err: any) {
      // Friendly error handling state
      const errorMsg: Message = {
        id: `msg-err-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Could not complete request: ${err.message || 'Model API unreachable'}. Please verify that the COLLISION backend server is running on port 8000.`,
        timestamp: Date.now()
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
    </div>
  );
}
