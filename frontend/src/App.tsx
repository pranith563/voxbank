import React, { useState } from "react";
import "./index.css"; // tailwind entry (ensure tailwind is wired here)
import VoiceSearchGeminiBrowser from "./VoiceSearchGeminiBrowser";
import Register from "./Register";

function App() {
  const [view, setView] = useState<"assistant" | "register">("assistant");

  return (
    <div className="min-h-screen bg-white">
      {/* Fixed header (another file can render content into this area too) */}
      <header className="fixed top-0 left-0 w-full h-20 bg-slate-800 text-white z-50 shadow-md">
        <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between">
          <div className="text-left">
            <h1 className="text-lg font-semibold">VoxBank</h1>
            <p className="text-sm text-slate-200">Your AI Voice Banking Companion</p>
          </div>
          <div className="flex gap-2">
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
              onClick={() => setView("register")}
              className={`px-3 py-1 text-xs rounded-full border ${
                view === "register"
                  ? "bg-white text-slate-800 border-white"
                  : "border-slate-400 text-slate-100"
              }`}
            >
              Register
            </button>
          </div>
        </div>
      </header>

      {/* Main content: pad top so header doesn't overlap */}
      <main className="pt-20">
        <div className="max-w-6xl mx-auto px-6 py-12">
          {view === "assistant" ? <VoiceSearchGeminiBrowser /> : <Register />}
        </div>
      </main>
    </div>
  );
}

export default App;
