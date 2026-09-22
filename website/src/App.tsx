import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './components/LandingPage';
import { PlaygroundWorkspace } from './components/PlaygroundWorkspace';
import type { Message } from './components/ChatWorkspace';
import type { ChatSession } from './components/Sidebar';
import { SettingsModal } from './components/SettingsModal';
import { AboutModal } from './components/AboutModal';
import { MitLicenseModal } from './components/MitLicenseModal';
import { CookieBanner } from './components/CookieBanner';
import { AmbientQuantumNodes } from './components/AmbientQuantumNodes';
import { collisionApi } from './api';

export default function App() {
  const [currentView, setCurrentView] = useState<'landing' | 'playground'>('landing');
  const [isDarkTheme, setIsDarkTheme] = useState(false);

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [sessionMessages, setSessionMessages] = useState<Record<string, Message[]>>({});
  const [isGenerating, setIsGenerating] = useState(false);

  // Model hyperparams
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(128);

  // Modals
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [isLicenseOpen, setIsLicenseOpen] = useState(false);

  // Sync theme
  useEffect(() => {
    if (isDarkTheme) {
      document.documentElement.classList.add('dark-theme');
    } else {
      document.documentElement.classList.remove('dark-theme');
    }
  }, [isDarkTheme]);

  // Load chat history from localStorage
  useEffect(() => {
    try {
      const savedSessions = localStorage.getItem('collision_sessions');
      const savedMessages = localStorage.getItem('collision_messages');
      const savedTheme = localStorage.getItem('collision_theme');
      if (savedSessions) setSessions(JSON.parse(savedSessions));
      if (savedMessages) setSessionMessages(JSON.parse(savedMessages));
      if (savedTheme === 'dark') setIsDarkTheme(true);
    } catch (e) {
      console.warn("Could not load local storage:", e);
    }
  }, []);

  // Save history
  useEffect(() => {
    try {
      localStorage.setItem('collision_sessions', JSON.stringify(sessions));
      localStorage.setItem('collision_messages', JSON.stringify(sessionMessages));
      localStorage.setItem('collision_theme', isDarkTheme ? 'dark' : 'light');
    } catch (e) {
      console.warn("Could not save to local storage:", e);
    }
  }, [sessions, sessionMessages, isDarkTheme]);

  const handleToggleTheme = () => {
    setIsDarkTheme(prev => !prev);
  };

  const handleNewChat = () => {
    setCurrentSessionId(null);
    setCurrentView('playground');
  };

  const handleSelectSession = (id: string) => {
    setCurrentSessionId(id);
    setCurrentView('playground');
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

    if (!targetSessionId) {
      targetSessionId = `session-${Date.now()}`;
      const newSession: ChatSession = {
        id: targetSessionId,
        title: promptText.length > 36 ? promptText.slice(0, 36) + '...' : promptText,
        updatedAt: Date.now()
      };
      setSessions(prev => [newSession, ...prev]);
      setCurrentSessionId(targetSessionId);
    }

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

    setCurrentView('playground');
    setIsGenerating(true);

    try {
      const response = await collisionApi.ask(promptText);

      const assistantMsg: Message = {
        id: `msg-ast-${Date.now()}`,
        role: 'assistant',
        content: response.answer || 'Response generated successfully.',
        timestamp: Date.now(),
        status: response.status,
        mode: response.mode,
        confidence: response.confidence,
        sources: response.sources || [],
        claims: response.claims || [],
        latency: response.latency
      };

      setSessionMessages(prev => ({
        ...prev,
        [targetSessionId!]: [...(prev[targetSessionId!] || []), assistantMsg]
      }));
    } catch (err: any) {
      const errorMsg: Message = {
        id: `msg-err-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Request could not be completed: ${err.message || 'COLLISION API unreachable'}. Please make sure backend is active on port 8000.`,
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
    const msgs = sessionMessages[currentSessionId] || [];
    const lastUserMsg = [...msgs].reverse().find(m => m.role === 'user');
    if (lastUserMsg) {
      handleSendPrompt(lastUserMsg.content);
    }
  };

  const handleFeedback = (messageId: string, rating: 'thumbs_up' | 'thumbs_down') => {
    if (!currentSessionId) return;
    setSessionMessages(prev => {
      const msgs = prev[currentSessionId] || [];
      const updated = msgs.map(m => m.id === messageId ? { ...m, feedbackRating: rating } : m);
      return { ...prev, [currentSessionId]: updated };
    });
  };

  const currentMessages = currentSessionId ? (sessionMessages[currentSessionId] || []) : [];

  return (
    <div className="app-root">
      {/* Ambient background particles with rare micro-depth parallax */}
      <AmbientQuantumNodes />

      {/* Top Navigation Bar */}
      <Navbar
        currentView={currentView}
        onNavigate={(view) => setCurrentView(view)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenAbout={() => setIsAboutOpen(true)}
        onOpenLicense={() => setIsLicenseOpen(true)}
        isDarkTheme={isDarkTheme}
        onToggleTheme={handleToggleTheme}
      />

      {/* Main Views */}
      {currentView === 'landing' ? (
        <LandingPage
          onStartChat={(prompt) => {
            if (prompt) {
              handleSendPrompt(prompt);
            } else {
              setCurrentView('playground');
            }
          }}
          onOpenAbout={() => setIsAboutOpen(true)}
          onOpenLicense={() => setIsLicenseOpen(true)}
        />
      ) : (
        <PlaygroundWorkspace
          sessions={sessions}
          currentSessionId={currentSessionId}
          messages={currentMessages}
          isGenerating={isGenerating}
          onSendPrompt={handleSendPrompt}
          onSelectSession={handleSelectSession}
          onNewChat={handleNewChat}
          onDeleteSession={handleDeleteSession}
          onRegenerate={handleRegenerate}
          onFeedback={handleFeedback}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />
      )}

      {/* Modals & Overlays */}
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

      <MitLicenseModal
        isOpen={isLicenseOpen}
        onClose={() => setIsLicenseOpen(false)}
      />

      {/* Cookie Consent & Telemetry Manager */}
      <CookieBanner />
    </div>
  );
}
