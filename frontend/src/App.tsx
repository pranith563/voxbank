import React, { useState, useEffect, useCallback } from "react";
import "./index.css";
import Register from "./Register";
import VoiceSearchGeminiBrowser from "./VoiceSearchGeminiBrowser";
import Login from "./Login";
import Header from "./components/Header";

function generateSessionId() {
  if (typeof window !== "undefined" && "crypto" in window && "randomUUID" in window.crypto) {
    return window.crypto.randomUUID();
  }
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function App() {
  const [view, setView] = useState<"assistant" | "register" | "login">("assistant");
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>(() => {
    if (typeof window === "undefined") return "local-session";
    const existing = window.localStorage.getItem("voxbank_session_id");
    if (existing) return existing;
    const id = generateSessionId();
    window.localStorage.setItem("voxbank_session_id", id);
    return id;
  });

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [language, setLanguage] = useState("en-US");
  const [voiceType, setVoiceType] = useState("default");
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Toggle Theme Effect
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
  }, [theme]);

  const toggleTheme = (newTheme: "dark" | "light") => {
    setTheme(newTheme);
  };

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    async function fetchSessionProfile() {
      try {
        const resp = await fetch(`/api/session/me?session_id=${encodeURIComponent(sessionId)}`, {
          method: "GET",
        });
        if (!resp.ok) return;
        const json = await resp.json();
        if (cancelled) return;
        if (json?.authenticated && json.user?.username) {
          setCurrentUser(json.user.username as string);
        } else {
          setCurrentUser(null);
        }
      } catch {
        // best-effort; ignore network errors
      }
    }

    // initial fetch
    fetchSessionProfile();
    // poll periodically so conversational login updates the header
    const id = window.setInterval(fetchSessionProfile, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [sessionId]);

  function handleLogout(options?: { skipServer?: boolean }) {
    const skipServer = options?.skipServer ?? false;
    if (!skipServer) {
      try {
        fetch("/api/auth/logout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        }).catch(() => undefined);
      } catch {
        // ignore
      }
    }

    setCurrentUser(null);
    setView("assistant");

    const newId = generateSessionId();
    setSessionId(newId);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("voxbank_session_id", newId);
      window.localStorage.removeItem("voxbank_username");
    }
  }

  const handleForcedLogout = useCallback(() => {
    handleLogout({ skipServer: true });
  }, [sessionId]);

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      <Header
        view={view}
        setView={setView}
        currentUser={currentUser}
        onLogout={handleLogout}
        settingsOpen={settingsOpen}
        setSettingsOpen={setSettingsOpen}
        theme={theme}
        toggleTheme={toggleTheme}
        language={language}
        setLanguage={setLanguage}
        voiceType={voiceType}
        setVoiceType={setVoiceType}
      />

      {/* Main content: pad top so header doesn't overlap */}
      <main className="pt-20 min-h-screen flex flex-col">
        <div className="flex-1 w-full">
          {view === "assistant" && (
            <VoiceSearchGeminiBrowser
              language={language}
              voiceType={voiceType}
              sessionId={sessionId}
              onForceLogout={handleForcedLogout}
            />
          )}
          {view === "register" && (
            <div className="max-w-6xl mx-auto px-6 py-12">
              <Register
                sessionId={sessionId}
                onRegistered={(user) => {
                  if (user?.username) {
                    setCurrentUser(user.username);
                  }
                  setView("assistant");
                }}
              />
            </div>
          )}
          {view === "login" && (
            <div className="max-w-6xl mx-auto px-6 py-12">
              <Login
                sessionId={sessionId}
                onLoggedIn={(user) => {
                  if (user?.username) {
                    setCurrentUser(user.username);
                  }
                  setView("assistant");
                }}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
