import React from "react";
import "./index.css"; // tailwind entry (ensure tailwind is wired here)
import VoiceSearchGeminiBrowser from "./VoiceSearchGeminiBrowser";

function App() {
  return (
    <div className="min-h-screen bg-white">
      {/* Fixed header (another file can render content into this area too) */}
      <header className="fixed top-0 left-0 w-full h-20 bg-slate-800 text-white z-50 shadow-md">
        <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-lg font-semibold">VoxBank</h1>
            <p className="text-sm text-slate-200">Your AI Voice Banking Companion</p>
          </div>
        </div>
      </header>

      {/* Main content: pad top so header doesn't overlap */}
      <main className="pt-20">
        <div className="max-w-6xl mx-auto px-6 py-12">
          <VoiceSearchGeminiBrowser />
        </div>
      </main>
    </div>
  );
}

export default App;
