import React, { useState, useEffect } from "react";
import "./index.css";
import Register from "./Register";
import VoiceSearchGeminiBrowser from "./VoiceSearchGeminiBrowser";
import Login from "./Login";

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

  function handleLogout() {
    // Inform orchestrator to clear auth state for this session (best-effort)
    try {
      fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => undefined);
    } catch {
      // ignore
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

  return (
    <div className="min-h-screen bg-white">
      {/* Fixed header (another file can render content into this area too) */}
      <header className="fixed top-0 left-0 w-full h-20 bg-slate-800 text-white z-50 shadow-md">
        <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between">
          <div className="text-left">
            <h1 className="text-lg font-semibold">VoxBank</h1>
            <p className="text-sm text-slate-200">Your AI Voice Banking Companion</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setView("assistant")}
              className={`px-3 py-1 text-xs rounded-full border ${
                view === "assistant"
                  ? "bg-white text-slate-800 border-white"
                  : "border-slate-400 text-slate-100"
              }`}
            >
              Assistant
            </button>
            <button
              onClick={() => {
                if (!currentUser) {
                  setView("register");
                }
              }}
              className={`px-3 py-1 text-xs rounded-full border hidden sm:inline-flex ${
                view === "register"
                  ? "bg-white text-slate-800 border-white"
                  : "border-slate-400 text-slate-100"
              }`}
              disabled={!!currentUser}
            >
              {currentUser ? currentUser : "Register"}
            </button>

            {/* Login button (only when logged out) */}
            {!currentUser && (
              <button
                onClick={() => setView("login")}
                className={`px-3 py-1 text-xs rounded-full border hidden sm:inline-flex ${
                  view === "login"
                    ? "bg-white text-slate-800 border-white"
                    : "border-slate-400 text-slate-100"
                }`}
              >
                Login
              </button>
            )}

            {/* Settings button */}
            <button
              type="button"
              onClick={() => setSettingsOpen((v) => !v)}
              className="w-8 h-8 flex items-center justify-center rounded-full bg-slate-700 hover:bg-slate-600 text-xs"
              aria-label="Settings"
            >
              ⚙
            </button>

            {/* Logout / login indicator */}
            {currentUser && (
              <button
                type="button"
                onClick={handleLogout}
                className="px-3 py-1 text-xs rounded-full border border-slate-400 text-slate-100 hover:bg-slate-700"
              >
                Logout
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main content: pad top so header doesn't overlap */}
      <main className="pt-20">
        <div className="max-w-6xl mx-auto px-6 py-12">
          {view === "assistant" && (
            <VoiceSearchGeminiBrowser language={language} voiceType={voiceType} sessionId={sessionId} />
          )}
          {view === "register" && (
            <Register
              sessionId={sessionId}
              onRegistered={(user) => {
                if (user?.username) {
                  setCurrentUser(user.username);
                }
                setView("assistant");
              }}
            />
          )}
          {view === "login" && (
            <Login
              sessionId={sessionId}
              onLoggedIn={(user) => {
                if (user?.username) {
                  setCurrentUser(user.username);
                }
                setView("assistant");
              }}
            />
          )}
        </div>
      </main>

      {/* Settings panel */}
      {settingsOpen && (
        <div className="fixed top-20 right-4 z-40 w-64 bg-white shadow-lg rounded-xl border border-slate-200 p-4">
          <h3 className="text-sm font-semibold text-slate-800 mb-3">Settings</h3>
          <div className="mb-3">
            <label className="block text-xs font-medium text-slate-600 mb-1">Language</label>
            <select
              className="w-full border rounded-md px-2 py-1 text-xs"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              <option value="en-US">English (US)</option>
              <option value="en-GB">English (UK)</option>
              <option value="en-IN">English (India)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Voice type</label>
            <select
              className="w-full border rounded-md px-2 py-1 text-xs"
              value={voiceType}
              onChange={(e) => setVoiceType(e.target.value)}
            >
              <option value="default">Default</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
