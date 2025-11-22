import React, { useEffect, useRef, useState } from "react";
import { Mic, MessageSquare, X, Volume2, Sparkles } from "lucide-react";
import BankingLoader from "./components/BankingLoader";

/* ----- Types for SpeechRecognition (TS) ----- */
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

const WS_URL = "ws://localhost:8000/ws";
const SILENCE_TIMEOUT_MS = 2000;
const USE_SERVER_TTS = true;

type AgentMode = "idle" | "listening" | "thinking" | "speaking";

interface Props {
  language?: string;
  voiceType?: string;
  sessionId: string;
  onForceLogout?: () => void;
}

export default function VoiceSearchGeminiBrowser({
  language = "en-US",
  voiceType = "default",
  sessionId,
  onForceLogout,
}: Props): JSX.Element {
  // Initial Loader State
  const [isLoading, setIsLoading] = useState(true);

  // UI / recognition states
  const [listening, setListening] = useState(false);
  const listeningRef = useRef<boolean>(false);
  const [recognitionSupported, setRecognitionSupported] = useState(true);
  const [transcript, setTranscript] = useState<string>("");
  const [interim, setInterim] = useState<string>("");
  const [pausedForPlayback, setPausedForPlayback] = useState<boolean>(false);
  const pausedForPlaybackRef = useRef<boolean>(false);

  // WebSocket / recognition refs
  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<any>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Transcript tracking
  const finalTranscriptRef = useRef<string>("");
  const lastSentCharIndexRef = useRef<number>(0);
  const userStoppedRef = useRef<boolean>(false);
  const wasListeningBeforePlaybackRef = useRef<boolean>(false);

  // silence timer
  const silenceTimerRef = useRef<number | null>(null);

  // Agent state + chat
  const [agentMode, setAgentMode] = useState<AgentMode>("idle");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<{ role: "user" | "assistant"; text: string }[]>([]);

  // ---------- lifecycle ----------
  useEffect(() => {
    // Simulate initial loading for effect
    const timer = setTimeout(() => setIsLoading(false), 1000);

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      console.error("SpeechRecognition not supported in this browser.");
      setRecognitionSupported(false);
    } else {
      setRecognitionSupported(true);
    }

    // create audio element once (not attached to DOM)
    const audio = document.createElement("audio");
    audio.autoplay = true;
    audioRef.current = audio;
    audioRef.current.muted = false;

    // When playback starts, stop recognition (to prevent echo being re-captured)
    audioRef.current.onplay = () => {
      console.log("[Audio] onplay -> stopping recognition to avoid echo");
      wasListeningBeforePlaybackRef.current = listeningRef.current === true;
      pausedForPlaybackRef.current = true;
      setPausedForPlayback(true);
      setAgentMode("speaking");
      userStoppedRef.current = false;
      try {
        if (recognitionRef.current) recognitionRef.current.stop();
      } catch (e) {
        console.warn("[Audio] error stopping recognition on play", e);
      }
    };

    // When playback ends, clear paused state and resume recognition if needed
    audioRef.current.onended = () => {
      console.log("[Audio] onended -> resume recognition if applicable");
      pausedForPlaybackRef.current = false;
      setPausedForPlayback(false);
      setAgentMode("idle");

      if (wasListeningBeforePlaybackRef.current && !userStoppedRef.current) {
        setTimeout(() => {
          try {
            startRecognition();
            console.log("[Audio] restarted recognition after playback");
          } catch (e) {
            console.warn("[Audio] restart recognition failed", e);
          }
        }, 200);
      } else {
        console.log("[Audio] Not restarting recognition (wasListening:", wasListeningBeforePlaybackRef.current, "userStopped:", userStoppedRef.current, ")");
      }
      wasListeningBeforePlaybackRef.current = false;
    };

    // Connect WS on mount
    if (!wsRef.current) {
      wsRef.current = createWebSocket();
    }

    return () => {
      clearTimeout(timer);
      clearSilenceTimer();
      try {
        if (recognitionRef.current) {
          recognitionRef.current.onresult = null;
          recognitionRef.current.onend = null;
          recognitionRef.current.onerror = null;
          recognitionRef.current.stop();
          recognitionRef.current = null;
        }
      } catch (e) { }
      try {
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      } catch (e) { }
      if (audioRef.current) {
        try {
          audioRef.current.onplay = null;
          audioRef.current.onended = null;
          audioRef.current.src = "";
        } catch (e) { }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- silence timer helpers ----------
  function clearSilenceTimer() {
    if (silenceTimerRef.current) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }

  function startOrResetSilenceTimer() {
    clearSilenceTimer();
    silenceTimerRef.current = window.setTimeout(() => {
      console.log(`[SR] Silence timeout (${SILENCE_TIMEOUT_MS}ms) fired — sending transcript to backend (if new)`);
      // do not stop recognition here
      sendTranscriptIfNew();
      silenceTimerRef.current = null;
    }, SILENCE_TIMEOUT_MS);
  }

  // ---------- WebSocket ----------
  function createWebSocket(): WebSocket {
    console.log("[WS] Creating websocket to", WS_URL);
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[WS] Open");
    };

    ws.onmessage = async (event: any) => {
      console.log("[WS] Message received", event);

      // binary frames
      if (event.data instanceof ArrayBuffer) {
        playArrayBufferAudio(event.data, "application/octet-stream");
        return;
      }
      if (event.data instanceof Blob) {
        const arr = await event.data.arrayBuffer();
        playArrayBufferAudio(arr, event.data.type || "application/octet-stream");
        return;
      }

      // text frames
      try {
        const msgText = typeof event.data === "string" ? event.data : await event.data.text();
        let parsed: any = null;
        try {
          parsed = JSON.parse(msgText);
        } catch (e) {
          parsed = null;
        }

        if (parsed && (parsed.type === "audio" || parsed.type === "audio_base64") && parsed.audio) {
          console.log("[WS] Received base64 audio payload. Playing...");
          const mime = parsed.mime || parsed.format || "audio/wav";
          playBase64Audio(parsed.audio, mime);
          return;
        }

        if (parsed && parsed.type === "ack") {
          console.log("[WS] Server ack:", parsed);
          return;
        }

        if (parsed && parsed.type === "transcript_echo" && parsed.text) {
          console.log("[WS] server transcript echo:", parsed.text);
          return;
        }

        if (parsed && parsed.type === "logout") {
          console.log("[WS] logout event received");
          if (onForceLogout) onForceLogout();
          return;
        }

        if (parsed && parsed.type === "error") {
          console.log("[WS] Error from backend:", parsed.message);
          setAgentMode("speaking");
          speak("Sorry there is an error. Please try once again");
          return;
        }

        if (parsed && parsed.type === "reply" && parsed.text) {
          console.log("[WS] reply text:", parsed.text);
          setChatMessages((prev) => [...prev, { role: "assistant", text: parsed.text }]);
          if (!USE_SERVER_TTS) {
            // Browser TTS mode
            setAgentMode("speaking");
            speak(parsed.text);
          }
          if (parsed.logged_out) {
            if (onForceLogout) onForceLogout();
          }
          return;
        }

        console.log("[WS] Received text payload (unparsed):", msgText);
      } catch (err) {
        console.error("[WS] onmessage error:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] Error", err);
    };

    ws.onclose = (ev) => {
      console.log("[WS] Closed", ev);
    };

    return ws;
  }

  // ---------- audio helpers ----------
  function speak(text: string) {
    if (!text) return;
    if (!("speechSynthesis" in window)) {
      console.warn("speechSynthesis not supported in this browser");
      return;
    }
    try {
      // Stop recognition while TTS is speaking to avoid sending echoed text back
      if (listeningRef.current) {
        stopRecognition();
      }
      pausedForPlaybackRef.current = true;
      setPausedForPlayback(true);
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      u.onend = () => {
        pausedForPlaybackRef.current = false;
        setPausedForPlayback(false);
        setAgentMode("idle");
      };
      window.speechSynthesis.speak(u);
    } catch (e) {
      console.error("Browser TTS failed:", e);
      pausedForPlaybackRef.current = false;
      setPausedForPlayback(false);
      setAgentMode("idle");
    }
  }

  function playArrayBufferAudio(buffer: ArrayBuffer, mime: string = "audio/wav") {
    try {
      const blob = new Blob([buffer], { type: mime });
      const url = URL.createObjectURL(blob);
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.muted = false;
        // play() will trigger onplay -> stop recognition
        audioRef.current.play().catch((e) => console.error("[Audio] play error", e));
      } else {
        const a = new Audio(url);
        a.play().catch((e) => console.error("[Audio] play error", e));
      }
      // revoke URL after some time
      setTimeout(() => {
        try { URL.revokeObjectURL(url); } catch (e) { }
      }, 60000);
    } catch (err) {
      console.error("[Audio] playArrayBufferAudio failed:", err);
    }
  }

  function playBase64Audio(base64OrDataUrl: string, mimeHint: string = "audio/wav") {
    try {
      // support both "data:audio/..;base64,AAA..." and raw base64
      const dataUrlMatch = base64OrDataUrl.match(/^data:(audio\/[a-zA-Z0-9.+-]+);base64,(.*)$/);
      let base64 = base64OrDataUrl;
      let mime = mimeHint;
      if (dataUrlMatch) {
        mime = dataUrlMatch[1];
        base64 = dataUrlMatch[2];
      }
      const binary = atob(base64);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i); // <-- FIX: use decoded binary
      }
      playArrayBufferAudio(bytes.buffer, mime);
    } catch (err) {
      console.error("[Audio] failed to decode base64 audio", err);
    }
  }

  // ---------- Send transcript (only new part) ----------
  function sendTranscriptIfNew(): boolean {
    const full = (finalTranscriptRef.current || "").trim();
    if (!full) return false;
    const lastIdx = lastSentCharIndexRef.current;
    if (lastIdx >= full.length) {
      console.log("[Send] nothing new to send");
      return false;
    }
    const newText = full.slice(lastIdx).trim();
    if (!newText) {
      console.log("[Send] new substring empty after trim");
      lastSentCharIndexRef.current = full.length;
      return false;
    }

    const payload = { type: "transcript", text: newText, session_id: sessionId, output_audio: USE_SERVER_TTS };

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // If WS is not open, we might need to reconnect or just log error.
      // Since we now connect on mount, if it's closed here it might be an error state.
      console.warn("[Send] WebSocket not open. Attempting to reconnect...");
      wsRef.current = createWebSocket();
      wsRef.current.onopen = () => {
        try {
          wsRef.current?.send(JSON.stringify(payload));
        }
        catch (e) { console.error("[WS] send error after open", e); }
      };
    } else {
      try {
        wsRef.current.send(JSON.stringify(payload));
      }
      catch (e) { console.error("[WS] send error", e); }
    }

    setAgentMode("thinking");
    // Show only the new chunk in captions and chat so UI stays readable.
    setTranscript(newText);
    setInterim("");
    setChatMessages((prev) => [...prev, { role: "user", text: newText }]);
    console.log("[Send] sent new text:", newText);
    lastSentCharIndexRef.current = full.length;
    return true;
  }

  // ---------- SpeechRecognition ----------
  function startRecognition() {
    if (!recognitionSupported) {
      alert("SpeechRecognition not supported in this browser.");
      return;
    }

    const SRConstructor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SRConstructor) {
      setRecognitionSupported(false);
      return;
    }

    // Reset transcript state for a new listening session so captions and
    // payloads only reflect this turn, not the whole conversation.
    finalTranscriptRef.current = "";
    lastSentCharIndexRef.current = 0;
    setTranscript("");
    setInterim("");

    // if already running, nothing to do
    if (recognitionRef.current && listeningRef.current) {
      console.log("[SR] recognition already running");
      return;
    }

    // create fresh recognition instance
    const recognition = new SRConstructor();
    recognitionRef.current = recognition;

    recognition.lang = language || "en-US";
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    recognition.onresult = (event: any) => {
      // reset silence timer whenever we get result (interim or final)
      startOrResetSilenceTimer();

      let interimAccum = "";
      let finalAccum = finalTranscriptRef.current || "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const res = event.results[i];
        const chunk = (res[0]?.transcript || "").trim();
        if (!chunk) continue;
        if (res.isFinal) {
          if (finalAccum && !finalAccum.endsWith(" ")) {
            finalAccum += " ";
          }
          finalAccum += chunk;
        } else {
          interimAccum += chunk;
        }
      }

      if (finalAccum !== finalTranscriptRef.current) {
        finalTranscriptRef.current = finalAccum;
        setTranscript(finalAccum.trim());
      }
      setInterim(interimAccum);
    };

    recognition.onerror = (ev: any) => {
      console.error("[SR] error", ev);
    };

    recognition.onend = () => {
      // clear timer, mark not listening and null the ref so startRecognition can create a fresh object
      clearSilenceTimer();
      listeningRef.current = false;
      setListening(false);
      try {
        // some browsers keep the old object alive; make sure we release it
        recognitionRef.current = null;
      } catch (e) {
        recognitionRef.current = null;
      }
      console.log("[SR] onend fired; recognitionRef cleared");
      // If not already thinking or speaking, revert to idle
      setAgentMode((mode) => (mode === "thinking" || mode === "speaking" ? mode : "idle"));
    };

    try {
      recognition.start();
      startOrResetSilenceTimer(); // fallback if no results
      listeningRef.current = true;
      setListening(true);
      setAgentMode("listening");
      userStoppedRef.current = false;
      console.log("[SR] recognition.start() called");
    } catch (err) {
      console.error("[SR] start() error", err);
    }
  }

  function stopRecognition(retainMode: boolean = false) {
    clearSilenceTimer();
    try {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    } catch (err) {
      console.warn("[SR] stop() threw", err);
    }
    listeningRef.current = false;
    setListening(false);
    // ensure ref cleared
    recognitionRef.current = null;
    if (!retainMode) {
      setAgentMode("idle");
    }
  }

  // ---------- UI handlers ----------
  function handleMicClick() {
    // If currently listening OR playback ongoing -> user re-clicks to STOP everything
    if (listeningRef.current || pausedForPlaybackRef.current) {
      console.log("[UI] Mic re-click: stop all (user explicit)");
      userStoppedRef.current = true;

      // Send any pending transcript before stopping
      const sent = sendTranscriptIfNew();

      stopRecognition(sent);
      if (audioRef.current && !audioRef.current.paused) {
        try {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        } catch (e) { }
      }
      pausedForPlaybackRef.current = false;
      setPausedForPlayback(false);
      return;
    }

    // otherwise start listening
    console.log("[UI] Mic clicked: start listening");
    userStoppedRef.current = false;
    startRecognition();
  }

  // ---------- Render ----------
  if (isLoading) {
    return <BankingLoader />;
  }

  const modeLabel =
    agentMode === "listening"
      ? "Listening..."
      : agentMode === "thinking"
        ? "Processing..."
        : agentMode === "speaking"
          ? "Speaking..."
          : "Tap to speak";



  return (
    <div className="h-[calc(100vh-5rem)] w-full bg-background text-foreground overflow-hidden relative flex flex-col">
      {/* Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-neon-blue/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-neon-purple/10 rounded-full blur-[120px]" />
      </div>

      {/* Main Content - Voice First Interface */}
      <main className="flex-1 relative z-10 flex flex-col items-center justify-center p-4 pb-24">

        {/* Status Indicator */}
        <div className="mb-32 flex-shrink-0 text-center relative z-20">
          <h2 className="text-3xl md:text-5xl font-light tracking-tight mb-4">
            {agentMode === "idle" ? (
              <span className="text-muted-foreground">How can I help you?</span>
            ) : (
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple animate-pulse">
                {modeLabel}
              </span>
            )}
          </h2>
          <p className="text-muted-foreground text-sm tracking-widest uppercase">
            AI Banking Assistant
          </p>
        </div>

        {/* Central Shielded Core / Mic Interaction */}
        <div className="relative group flex items-center justify-center flex-shrink-0">
          {/* --- Ambient Glow (Base) --- */}
          <div className={`absolute inset-0 bg-neon-blue/20 blur-[100px] rounded-full transition-all duration-1000 ${agentMode === 'listening' ? 'scale-150 opacity-60' : 'scale-100 opacity-30'
            }`} />

          {/* --- Outer Shield Ring (Static/Slow) --- */}
          <div className={`absolute w-64 h-64 md:w-80 md:h-80 rounded-full border border-secondary/10 flex items-center justify-center transition-all duration-700 ${agentMode === 'listening' ? 'scale-110 border-neon-blue/30' : 'scale-100'
            }`}>
            {/* Decorative ticks on outer ring */}
            <div className="absolute inset-0 rounded-full border-t border-b border-border rotate-45" />
          </div>

          {/* --- Middle Rotating Shield (The "Vault Lock") --- */}
          <div className={`absolute w-52 h-52 md:w-64 md:h-64 rounded-full border border-border flex items-center justify-center transition-all duration-1000 ${agentMode === 'thinking' ? 'animate-spin-slow border-neon-purple/50' :
            agentMode === 'listening' ? 'animate-[spin_20s_linear_infinite] border-neon-blue/40' :
              'rotate-0 border-border'
            }`}>
            <div className="absolute top-0 w-1 h-1 bg-foreground/50 rounded-full shadow-[0_0_10px_white]" />
            <div className="absolute bottom-0 w-1 h-1 bg-foreground/50 rounded-full shadow-[0_0_10px_white]" />
            <div className="absolute left-0 w-1 h-1 bg-foreground/50 rounded-full shadow-[0_0_10px_white]" />
            <div className="absolute right-0 w-1 h-1 bg-foreground/50 rounded-full shadow-[0_0_10px_white]" />
          </div>

          {/* --- Inner Rotating Ring (Reverse) --- */}
          <div className={`absolute w-40 h-40 md:w-48 md:h-48 rounded-full border-2 border-dashed border-secondary/20 transition-all duration-700 ${agentMode === 'thinking' ? 'animate-spin-reverse-slower border-neon-purple' :
            agentMode === 'listening' ? 'animate-[spin_15s_linear_infinite_reverse] border-neon-blue' :
              'rotate-45 border-border'
            }`} />

          {/* --- Ripple Effect (Speaking) --- */}
          {agentMode === 'speaking' && (
            <>
              <div className="absolute w-32 h-32 rounded-full border border-neon-green/50 animate-ripple" />
              <div className="absolute w-32 h-32 rounded-full border border-neon-green/30 animate-ripple delay-75" />
              <div className="absolute w-32 h-32 rounded-full border border-neon-green/10 animate-ripple delay-150" />
            </>
          )}

          {/* --- The Core (Button) --- */}
          <button
            onClick={handleMicClick}
            aria-pressed={listening}
            className={`relative z-20 w-32 h-32 md:w-40 md:h-40 rounded-full flex items-center justify-center transition-all duration-500 shadow-2xl border-4 ${agentMode === 'listening' ? 'bg-background border-neon-blue shadow-[0_0_50px_rgba(0,243,255,0.4)] scale-105' :
              agentMode === 'thinking' ? 'bg-background border-neon-purple shadow-[0_0_50px_rgba(188,19,254,0.4)] scale-95' :
                agentMode === 'speaking' ? 'bg-background border-neon-green shadow-[0_0_50px_rgba(10,255,104,0.4)] scale-105' :
                  'bg-slate-900 border-slate-700 shadow-[0_0_30px_rgba(0,0,0,0.5)] hover:border-slate-500'
              }`}
          >
            {/* Core Inner Glow */}
            <div className={`absolute inset-2 rounded-full opacity-50 transition-colors duration-500 ${agentMode === 'listening' ? 'bg-neon-blue animate-pulse-fast' :
              agentMode === 'thinking' ? 'bg-neon-purple animate-pulse' :
                agentMode === 'speaking' ? 'bg-neon-green' :
                  'bg-slate-800'
              }`} />

            {/* Icon */}
            <div className="relative z-30">
              {agentMode === "listening" ? (
                <Mic className="w-10 h-10 md:w-12 md:h-12 text-foreground drop-shadow-[0_0_10px_rgba(255,255,255,0.8)]" />
              ) : agentMode === "speaking" ? (
                <Volume2 className="w-10 h-10 md:w-12 md:h-12 text-neon-green drop-shadow-[0_0_10px_rgba(10,255,104,0.8)]" />
              ) : agentMode === "thinking" ? (
                <Sparkles className="w-10 h-10 md:w-12 md:h-12 text-neon-purple animate-pulse drop-shadow-[0_0_10px_rgba(188,19,254,0.8)]" />
              ) : (
                <Mic className="w-10 h-10 md:w-12 md:h-12 text-muted-foreground group-hover:text-foreground transition-colors" />
              )}
            </div>
          </button>
        </div>

        {/* Live Transcript */}
        <div className="mt-24 h-24 w-full max-w-2xl flex items-center justify-center text-center px-4">
          <p className="text-lg md:text-2xl text-muted-foreground font-light leading-relaxed">
            {transcript || interim || (
              <span className="text-muted-foreground italic">Listening for command...</span>
            )}
          </p>
        </div>

      </main>

      {/* Chat Trigger - Right Side Tab */}
      <div className="fixed right-0 top-1/2 -translate-y-1/2 z-50">
        <button
          onClick={() => setChatOpen(!chatOpen)}
          className={`
            flex items-center justify-center w-10 h-24 
            bg-card/80 backdrop-blur-md border-l border-t border-b border-border 
            rounded-l-xl shadow-lg
            hover:w-12 hover:bg-accent transition-all duration-300 group
            ${chatOpen ? 'translate-x-full' : 'translate-x-0'}
          `}
        >
          <div className="flex flex-col items-center gap-2">
            <MessageSquare className="w-5 h-5 text-primary group-hover:scale-110 transition-transform" />
            <div className="w-1 h-1 rounded-full bg-primary/50" />
            <div className="w-1 h-1 rounded-full bg-primary/50" />
            <div className="w-1 h-1 rounded-full bg-primary/50" />
          </div>
        </button>
      </div>

      {/* Conversation History Sidebar */}
      <div
        className={`
          fixed inset-y-0 right-0 w-80 md:w-96 bg-background/95 backdrop-blur-xl 
          border-l border-border shadow-2xl
          z-50 transform transition-transform duration-500 ease-out
          flex flex-col
          ${chatOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        <div className="p-6 border-b border-border flex justify-between items-center bg-muted/20">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-primary" />
            <span className="text-lg font-light tracking-wide text-foreground">History</span>
          </div>
          <button
            onClick={() => setChatOpen(false)}
            className="p-2 hover:bg-muted rounded-full transition-colors text-muted-foreground hover:text-foreground"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {chatMessages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground/50 space-y-4">
              <MessageSquare className="w-12 h-12 opacity-20" />
              <p className="text-sm font-light">No conversation yet</p>
            </div>
          )}
          {chatMessages.map((m, idx) => (
            <div key={idx} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${m.role === "user"
                  ? "bg-primary text-primary-foreground rounded-tr-sm"
                  : "bg-muted text-foreground border border-border rounded-tl-sm"
                  }`}
              >
                {m.text}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
