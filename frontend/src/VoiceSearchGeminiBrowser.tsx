import React, { useEffect, useRef, useState } from "react";

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

export default function VoiceSearchGeminiBrowser(): JSX.Element {
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
        let parsed = null;
        try { parsed = JSON.parse(msgText); } catch (e) { parsed = null; }

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

        if (parsed && parsed.type === "reply" && parsed.text) {
          console.log("[WS] reply text:", parsed.text);
          // Use browser TTS to speak the assistant reply.
          speak(parsed.text);
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
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      window.speechSynthesis.speak(u);
    } catch (e) {
      console.error("Browser TTS failed:", e);
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

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      wsRef.current = createWebSocket();
      wsRef.current.onopen = () => {
        try { wsRef.current?.send(JSON.stringify({ type: "transcript", text: newText })); }
        catch (e) { console.error("[WS] send error after open", e); }
      };
    } else {
      try { wsRef.current.send(JSON.stringify({ type: "transcript", text: newText })); }
      catch (e) { console.error("[WS] send error", e); }
    }

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

    // if already running, nothing to do
    if (recognitionRef.current && listeningRef.current) {
      console.log("[SR] recognition already running");
      return;
    }

    // create fresh recognition instance
    const recognition = new SRConstructor();
    recognitionRef.current = recognition;

    recognition.lang = "en-US";
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
        if (res.isFinal) {
          finalAccum += (res[0]?.transcript || "");
        } else {
          interimAccum += (res[0]?.transcript || "");
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
    };

    try {
      recognition.start();
      startOrResetSilenceTimer(); // fallback if no results
      listeningRef.current = true;
      setListening(true);
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
  // show SVG waves when either listening or paused for playback (visual keeps playing)
  const showWaves = listening || pausedForPlayback;

  return (
    <div className="w-full flex flex-col items-center">
      <div className="w-full max-w-3xl px-4">
        <div className="flex gap-3 items-center mb-6">
          <input
            className="flex-1 rounded-2xl border border-slate-200 px-4 py-3 shadow-inner focus:outline-none"
            placeholder="Speak or type to search..."
            value={transcript}
            onChange={(e) => {
              setTranscript(e.target.value);
              finalTranscriptRef.current = e.target.value;
            }}
          />
          <button
            onClick={handleManualSend}
            className="rounded-2xl border px-4 py-3 bg-white hover:bg-slate-50"
          >
            Search
          </button>
        </div>

        <div className="rounded-2xl border border-slate-100 p-6 mb-6 bg-white shadow-sm">
          <div className="text-xs text-slate-400 mb-2">Transcription</div>
          <div className="text-base text-slate-700 min-h-[44px]">
            {transcript || (interim ? interim : "No speech yet")}
          </div>
        </div>

        <div className="flex flex-col items-center">
          <div
            onClick={handleMicClick}
            role="button"
            aria-pressed={listening}
            className="relative w-[120px] h-[120px] rounded-full bg-white shadow-lg flex items-center justify-center cursor-pointer select-none"
          >
            <div className="w-8 h-8 rounded-full bg-red-600 shadow-inner" />
            {/* SVG animated radiating waves */}
            {showWaves && (
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
                <defs>
                  <radialGradient id="g1" cx="50%" cy="50%">
                    <stop offset="0%" stopColor="rgba(229,57,53,0.08)" />
                    <stop offset="100%" stopColor="rgba(229,57,53,0.02)" />
                  </radialGradient>
                </defs>
                <circle cx="60" cy="60" r="28" fill="url(#g1)" />
                <g>
                  {/* three expanding rings animated */}
                  <circle cx="60" cy="60" r="36" stroke="rgba(229,57,53,0.12)" strokeWidth="2" >
                    <animate attributeName="r" from="36" to="52" dur="1.6s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.9" to="0" dur="1.6s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="60" cy="60" r="28" stroke="rgba(229,57,53,0.10)" strokeWidth="2" >
                    <animate attributeName="r" from="28" to="48" dur="1.6s" begin="0.5s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.85" to="0" dur="1.6s" begin="0.5s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="60" cy="60" r="20" stroke="rgba(229,57,53,0.08)" strokeWidth="2" >
                    <animate attributeName="r" from="20" to="44" dur="1.6s" begin="1s" repeatCount="indefinite" />
                    <animate attributeName="opacity" from="0.8" to="0" dur="1.6s" begin="1s" repeatCount="indefinite" />
                  </circle>
                </g>
              </svg>
            )}
          </div>

          <div className="mt-3 text-sm text-slate-600">
            {pausedForPlayback ? (
              <span className="text-amber-600 font-semibold">Paused for reply</span>
            ) : (
              <span>{listening ? "Listening..." : "Tap to speak"}</span>
            )}
          </div>

          <div className="mt-2 text-xs text-slate-400">
            Auto-sends after 2s silence. Re-click mic to stop playback & listening.
          </div>
        </div>
      </div>

      {/* debug */}
      <div className="mt-8 text-xs text-slate-400">
        <div>Recognition supported: {String(recognitionSupported)}</div>
        <div>Listening (recognition running): {String(listeningRef.current)}</div>
        <div>Paused for reply (playback): {String(pausedForPlayback)}</div>
      </div>
    </div>
  );
}
