import React, { useCallback, useEffect, useRef, useState } from "react";

/* ----- Types for SpeechRecognition (TS) ----- */
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

const WS_URL = "ws://localhost:8000/ws";
// 2 seconds silence
const SILENCE_TIMEOUT_MS = 2000;
// Toggle to prefer server-side TTS (audio from backend) vs browser speechSynthesis.
// When true, the client requests audio over WebSocket and will not use browser TTS.
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

  const triggerForceLogout = useCallback(() => {
    if (onForceLogout) {
      onForceLogout();
    }
  }, [onForceLogout]);

  // ---------- lifecycle ----------
  useEffect(() => {
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

    return () => {
      clearSilenceTimer();
      try {
        if (recognitionRef.current) {
          recognitionRef.current.onresult = null;
          recognitionRef.current.onend = null;
          recognitionRef.current.onerror = null;
          recognitionRef.current.stop();
          recognitionRef.current = null;
        }
      } catch (e) {}
      try {
        if (wsRef.current) wsRef.current.close();
      } catch (e) {}
      if (audioRef.current) {
        try {
          audioRef.current.onplay = null;
          audioRef.current.onended = null;
          audioRef.current.src = "";
        } catch (e) {}
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
          triggerForceLogout();
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
            triggerForceLogout();
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
        try { URL.revokeObjectURL(url); } catch (e) {}
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
  function sendTranscriptIfNew() {
    const full = (finalTranscriptRef.current || "").trim();
    if (!full) return;
    const lastIdx = lastSentCharIndexRef.current;
    if (lastIdx >= full.length) {
      console.log("[Send] nothing new to send");
      return;
    }
    const newText = full.slice(lastIdx).trim();
    if (!newText) {
      console.log("[Send] new substring empty after trim");
      lastSentCharIndexRef.current = full.length;
      return;
    }

    const payload = { type: "transcript", text: newText, session_id: sessionId, output_audio: USE_SERVER_TTS };

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
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

    recognition.onresult = (event: SpeechRecognitionEvent) => {
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

  function stopRecognition() {
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
    setAgentMode("idle");
  }

  // ---------- UI handlers ----------
  function handleMicClick() {
    // If currently listening OR playback ongoing -> user re-clicks to STOP everything
    if (listeningRef.current || pausedForPlaybackRef.current) {
      console.log("[UI] Mic re-click: stop all (user explicit)");
      userStoppedRef.current = true;
      stopRecognition();
      if (audioRef.current && !audioRef.current.paused) {
        try {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        } catch (e) {}
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

  function handleManualSend() {
    const full = (finalTranscriptRef.current || "").trim();
    if (!full) {
      alert("No transcript to send.");
      return;
    }
    sendTranscriptIfNew();
  }

  // ---------- Render ----------
  const showWaves = agentMode !== "idle";
  const modeLabel =
    agentMode === "listening"
      ? "Listening..."
      : agentMode === "thinking"
      ? "Thinking..."
      : agentMode === "speaking"
      ? "Speaking..."
      : "Tap to speak";

  const ringColor =
    agentMode === "listening"
      ? "bg-blue-600 shadow-[0_0_40px_rgba(37,99,235,0.7)]"
      : agentMode === "thinking"
      ? "bg-indigo-600 shadow-[0_0_40px_rgba(79,70,229,0.7)]"
      : agentMode === "speaking"
      ? "bg-emerald-600 shadow-[0_0_40px_rgba(16,185,129,0.7)]"
      : "bg-slate-800 shadow-xl";

  return (
    <>
      <div className="w-full flex flex-col items-center justify-center py-10">
        <div className="flex flex-col items-center gap-6">
          {/* Mic button */}
          <div
            onClick={handleMicClick}
            role="button"
            aria-pressed={listening}
            className={`relative w-40 h-40 rounded-full flex items-center justify-center cursor-pointer select-none transition-shadow duration-300 ${ringColor}`}
          >
            {/* Inner mic icon */}
            <div className="w-16 h-16 rounded-full bg-slate-900 flex items-center justify-center">
              <span className="text-3xl text-white">🎙</span>
            </div>

            {/* Animated ring */}
            {showWaves && (
              <svg
                className="absolute inset-0 w-full h-full"
                viewBox="0 0 160 160"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden
              >
                <g>
                  <circle cx="80" cy="80" r="52" stroke="rgba(255,255,255,0.20)" strokeWidth="2">
                    <animate attributeName="r" from="52" to="72" dur="1.6s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.9" to="0" dur="1.6s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="80" cy="80" r="42" stroke="rgba(255,255,255,0.15)" strokeWidth="2">
                    <animate attributeName="r" from="42" to="68" dur="1.6s" begin="0.4s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.8" to="0" dur="1.6s" begin="0.4s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="80" cy="80" r="34" stroke="rgba(255,255,255,0.10)" strokeWidth="2">
                    <animate attributeName="r" from="34" to="60" dur="1.6s" begin="0.8s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.7" to="0" dur="1.6s" begin="0.8s" repeatCount="indefinite" />
                  </circle>
                </g>
              </svg>
            )}
          </div>

          {/* Mode label */}
          <div className="px-6 py-2 rounded-full bg-slate-900/90 text-white text-sm shadow-md mt-2">
            {modeLabel}
          </div>

          {/* Transcript below mic */}
          <div className="mt-4 w-full max-w-xl text-center text-sm text-slate-700 min-h-[40px] px-4 py-3 rounded-2xl bg-white/80 border border-slate-200 shadow-sm">
            {transcript || (interim ? interim : "No speech yet")}
          </div>

          <div className="mt-2 text-xs text-slate-400">
            Auto-sends after 2s of silence. Tap again to stop listening or playback.
          </div>
        </div>
      </div>

      {/* Collapsible chat window */}
      <div className="fixed bottom-4 right-4 z-40">
        <button
          type="button"
          onClick={() => setChatOpen((v) => !v)}
          className="w-10 h-10 rounded-full bg-slate-900 text-white flex items-center justify-center shadow-lg hover:bg-slate-800 text-lg"
          aria-label="Toggle chat history"
        >
          💬
        </button>
        {chatOpen && (
          <div className="mt-2 w-80 max-h-96 bg-white shadow-2xl rounded-xl border border-slate-200 flex flex-col">
            <div className="px-3 py-2 text-xs font-semibold text-slate-700 border-b border-slate-100 flex items-center justify-between">
              <span>Conversation</span>
              <span className="text-[10px] text-slate-400">user ↔ assistant</span>
            </div>
            <div className="p-3 flex-1 overflow-y-auto space-y-2 text-xs">
              {chatMessages.length === 0 && (
                <div className="text-slate-400">No messages yet.</div>
              )}
              {chatMessages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`px-2 py-1 rounded-lg max-w-[75%] ${
                      m.role === "user"
                        ? "bg-slate-900 text-white"
                        : "bg-slate-100 text-slate-800"
                    }`}
                  >
                    {m.text}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </>
  );
}
