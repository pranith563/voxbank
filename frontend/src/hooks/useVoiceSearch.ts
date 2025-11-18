import { useEffect, useState } from "react";
import { useVoiceWebSocket } from "./useVoiceWebSocket";
import { useAudioSampling } from "./useAudioSampling";

/**
 * VoiceSearch Hook - Main hook that combines WebSocket and audio sampling
 *
 * - Coordinates WebSocket communication and audio processing
 * - Manages transcript state and listening state
 * - Provides unified interface for voice search functionality
 */

interface UseVoiceSearchOptions {
  wsUrl?: string;
  speechThreshold?: number;
  silenceMs?: number;
  minEndIntervalMs?: number;
  clientSendSampleRate?: number;
}

interface UseVoiceSearchReturn {
  listening: boolean;
  finalTranscript: string;
  interimTranscript: string;
  displayTranscript: string;
  supported: boolean;
  startListening: () => Promise<void>;
  stopListening: () => void;
  toggleListening: () => void;
}

export function useVoiceSearch(options: UseVoiceSearchOptions = {}): UseVoiceSearchReturn {
  const {
    wsUrl = "ws://0.0.0.0:8000/ws",
    speechThreshold = 0.01,
    silenceMs = 700,
    minEndIntervalMs = 600,
    clientSendSampleRate = 16000,
  } = options;

  const [listening, setListening] = useState(false);
  const [finalTranscript, setFinalTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [supported, setSupported] = useState(true);

  const displayTranscript = `${finalTranscript}${interimTranscript ? (finalTranscript ? " " : "") + interimTranscript : ""}`;

  // --- speak via browser TTS ---
  const speak = (text: string) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    window.speechSynthesis.speak(u);
  };

  // WebSocket hook with callbacks
  const { wsRef, openWebSocket, sendBinary, sendJSON, isConnected } = useVoiceWebSocket({
    wsUrl,
    sampleRate: clientSendSampleRate,
    callbacks: {
      onPartial: (text: string) => {
        setInterimTranscript(text);
      },
      onFinal: (text: string) => {
        setFinalTranscript((prev) => (prev ? prev + " " : "") + text);
        setInterimTranscript("");
      },
      onReply: (text: string) => {
        speak(text);
      },
      onError: (message: string) => {
        console.error("WebSocket error:", message);
      },
    },
  });

  // Audio sampling hook with callbacks
  const { startSampling, stopSampling, resetSilenceDetection } = useAudioSampling({
    speechThreshold,
    silenceMs,
    minEndIntervalMs,
    clientSendSampleRate,
    callbacks: {
      onAudioChunk: (int16Data: Int16Array) => {
        // Send audio chunk to WebSocket
        if (isConnected()) {
          sendBinary(int16Data.buffer as ArrayBuffer);
        }
      },
      onSilenceDetected: () => {
        // Send end event when silence is detected
        sendJSON({ event: "end" });
      },
    },
  });

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListeningInternal();
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.close();
        } catch {}
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- start / stop flow ---
  async function startListening() {
    if (listening) return;
    
    // Open WebSocket connection
    openWebSocket();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.info("Microphone stream OK");
      
      // Start audio sampling
      startSampling(stream);
      resetSilenceDetection();

      setFinalTranscript("");
      setInterimTranscript("");
      setListening(true);
    } catch (e) {
      console.error("getUserMedia error", e);
      setSupported(false);
    }
  }

  function stopListeningInternal() {
    // send final end event before shutting down so server finalizes any partials
    if (isConnected()) {
      sendJSON({ event: "end" });
    }

    // Stop audio sampling
    stopSampling();
    
    setListening(false);
  }

  function stopListening() {
    stopListeningInternal();
    // option: keep ws open for quick re-start, or close it:
    // closeWebSocket();
  }

  const toggleListening = () => (listening ? stopListening() : startListening());

  return {
    listening,
    finalTranscript,
    interimTranscript,
    displayTranscript,
    supported,
    startListening,
    stopListening,
    toggleListening,
  };
}
