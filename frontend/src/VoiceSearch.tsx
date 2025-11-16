import { useEffect, useRef, useState } from "react";

// shadcn/ui components (adjust paths if needed)
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { MicButton } from "@/components/MicButton";

/**
 * VoiceSearch - WebSocket streaming client with client-side silence detection
 *
 * - Sends a {"type":"meta", sampleRate, channels, encoding, lang} JSON on connect
 * - Streams PCM16LE binary frames afterwards (downsampled to 16k)
 * - Detects silence on the client and sends {"event":"end"} automatically
 * - Expects server JSON messages: {type: "partial"|"final"|"reply", text: "..."}
 */

export default function VoiceSearch({ wsUrl = "ws://0.0.0.0:8765/ws" }: { wsUrl?: string }): JSX.Element {
  const [listening, setListening] = useState(false);

  // transcripts
  const [finalTranscript, setFinalTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const displayTranscript = `${finalTranscript}${interimTranscript ? (finalTranscript ? " " : "") + interimTranscript : ""}`;

  const [supported, setSupported] = useState(true);

  // Audio / ws refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // We downsample to 16k before sending
  const clientSendSampleRate = 16000;
  const clientSampleRateRef = useRef<number>(clientSendSampleRate);

  // Avoid flooding partials
  const partialThrottleRef = useRef<number>(0);

  // Silence detection state
  const speakingRef = useRef<boolean>(false); // true while above threshold
  const lastVoiceTimeRef = useRef<number>(0); // ms timestamp of last chunk above threshold
  const silenceSentRef = useRef<boolean>(false); // whether we've sent 'end' for current silence
  const lastEndSentTimeRef = useRef<number>(0);

  // Silence detection parameters (tweak to taste)
  const SPEECH_THRESHOLD = 0.01; // RMS threshold (float PCM). Typical speech ~0.02-0.1; lower => more sensitive
  const SILENCE_MS = 700; // milliseconds of silence to consider end-of-utterance
  const MIN_END_INTERVAL_MS = 600; // minimum time between successive end events to avoid double sends

  // disable scrolling while mounted
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

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

  // --- WebSocket management ---
  function openWebSocket() {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      console.debug("WebSocket already open/connecting");
      return;
    }

    const url = wsUrl; // default: ws://0.0.0.0:8765/ws
    console.info("Opening websocket to:", url);
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.info("WS connected");
      // send meta: we send sampleRate=16000 because we downsample to 16k
      const meta = { type: "meta", sampleRate: clientSendSampleRate, channels: 1, encoding: "pcm16", lang: "en" };
      try {
        ws.send(JSON.stringify(meta));
        console.info("Sent meta", meta);
      } catch (err) {
        console.error("Failed to send meta:", err);
      }
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data);
          handleServerMessage(msg);
        } catch (e) {
          console.warn("Non-JSON ws message", e, ev.data);
        }
      } else {
        // If server sends binary audio (TTS), you can handle it here (not necessary for silence detection).
        console.debug("Received binary message from server (ignored by default)");
      }
    };

    ws.onclose = (ev) => {
      console.info("WS closed", ev.code, ev.reason);
    };

    ws.onerror = (ev) => {
      console.error("WS error", ev);
    };

    wsRef.current = ws;
    (window as any).debug_ws = ws;
  }

  function closeWebSocket() {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
      try {
        delete (window as any).debug_ws;
      } catch {}
    }
  }

  // --- handle incoming server messages ---
  function handleServerMessage(msg: any) {
    const t = msg.type;
    if (t === "partial") {
      // throttle partial updates slightly to avoid UI jitter
      partialThrottleRef.current += 1;
      if (partialThrottleRef.current % 2 === 0) {
        setInterimTranscript(msg.text || "");
      }
    } else if (t === "final") {
      const text = msg.text || "";
      setFinalTranscript((prev) => (prev ? prev + " " : "") + text);
      setInterimTranscript("");
    } else if (t === "reply") {
      const text = msg.text || "";
      // speak reply using browser TTS
      speak(text);
    } else if (t === "meta_ack") {
      // server acknowledged readiness
      console.debug("Server meta_ack", msg);
    } else if (t === "error") {
      console.error("Server error:", msg.message);
    } else {
      console.debug("Unknown server message", msg);
    }
  }

  // --- speak via browser TTS ---
  function speak(text: string) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    window.speechSynthesis.speak(u);
  }

  // --- start / stop flow --- (exposed to MicButton)
  async function startListening() {
    if (listening) return;
    openWebSocket();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.info("Microphone stream OK");
      mediaStreamRef.current = stream;
      audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      // we still set actual audio context rate in case you need it, but we send 16k
      const actualRate = audioCtxRef.current.sampleRate;
      clientSampleRateRef.current = clientSendSampleRate;

      // create source
      sourceRef.current = audioCtxRef.current.createMediaStreamSource(stream);

      // ScriptProcessorNode is easier to use cross-browser; buffer 4096 gives ~85ms at 48k
      const bufferSize = 4096;
      const processor = audioCtxRef.current.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      // reset silence detection state
      speakingRef.current = false;
      lastVoiceTimeRef.current = performance.now();
      silenceSentRef.current = false;
      lastEndSentTimeRef.current = 0;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        const float32 = e.inputBuffer.getChannelData(0);

        // --- compute RMS for silence detection ---
        let sum = 0;
        for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
        const rms = Math.sqrt(sum / float32.length);

        const now = performance.now();

        // if above threshold -> mark speaking and reset silence flags
        if (rms >= SPEECH_THRESHOLD) {
          speakingRef.current = true;
          lastVoiceTimeRef.current = now;
          silenceSentRef.current = false; // reset, we are speaking again
        } else {
          // if we were speaking and silence duration exceeded threshold -> send end
          if (speakingRef.current) {
            const silenceDur = now - lastVoiceTimeRef.current;
            if (silenceDur >= SILENCE_MS && !silenceSentRef.current) {
              const lastEnd = lastEndSentTimeRef.current;
              if (now - lastEnd > MIN_END_INTERVAL_MS) {
                // send end event to server
                try {
                  wsRef.current!.send(JSON.stringify({ event: "end" }));
                  silenceSentRef.current = true;
                  lastEndSentTimeRef.current = now;
                  // reset speaking to allow next utterance detection when voice resumes
                  speakingRef.current = false;
                  lastVoiceTimeRef.current = 0;
                  console.debug("Silence detected - sent event:end");
                } catch (err) {
                  console.error("Failed to send end event:", err);
                }
              }
            }
          }
        }

        // --- downsample to 16k and convert to Int16 ---
        const int16 = floatTo16BitPCM(downsampleBuffer(float32, audioCtxRef.current!.sampleRate, clientSendSampleRate));
        // send binary
        try {
          wsRef.current.send(int16.buffer);
        } catch (err) {
          console.error("WS send error", err);
        }
      };

      sourceRef.current.connect(processor);
      // connecting processor to destination keeps some browsers happy; set gain 0 if you don't want echo
      processor.connect(audioCtxRef.current.destination);

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
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ event: "end" }));
      } catch {}
    }

    // stop audio nodes / tracks
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
        processorRef.current.onaudioprocess = null;
      } catch {}
      processorRef.current = null;
    }
    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch {}
      sourceRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch {}
      audioCtxRef.current = null;
    }
    setListening(false);
  }

  function stopListening() {
    stopListeningInternal();
    // option: keep ws open for quick re-start, or close it:
    // closeWebSocket();
  }

  const toggleListening = () => (listening ? stopListening() : startListening());

  // --- util: convert Float32 PCM to Int16 ArrayBuffer ---
  function floatTo16BitPCM(float32Array: Float32Array) {
    const l = float32Array.length;
    const buffer = new ArrayBuffer(l * 2);
    const view = new DataView(buffer);
    let offset = 0;
    for (let i = 0; i < l; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, float32Array[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(offset, s, true); // little-endian
    }
    return new Int16Array(buffer);
  }

  // --- util: downsample float32 buffer from srcRate -> dstRate (returns Float32Array) ---
  function downsampleBuffer(buffer: Float32Array, srcRate: number, dstRate: number) {
    if (dstRate === srcRate) return buffer;
    const sampleRateRatio = srcRate / dstRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      // average the range to avoid aliasing
      let accum = 0,
        count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

  // --- UI render ---
  return (
    // Use justify-start so the search bar stays visible at top area of viewport
    <div className="h-screen w-screen bg-gray-50 flex flex-col items-center justify-start p-6 relative overflow-hidden">
      {/* MAIN UI (kept near the top so it doesn't flow below) */}
      <div className="w-full max-w-2xl mt-6">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-4 relative">
              <Input
                value={displayTranscript}
                onChange={(e) => {
                  const v = (e.target as HTMLInputElement).value;
                  setFinalTranscript(v);
                  setInterimTranscript("");
                }}
                placeholder={supported ? "Speak or type to search..." : "Speech not supported in this browser"}
                className="flex-1 text-lg pr-4"
                aria-label="Search"
              />

              {/* Search button (no mic inside the input) */}
              <Button onClick={() => alert("Search: " + displayTranscript)}>Search</Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Optionally a small transcript card under the search bar */}
      <div className="w-full max-w-2xl mt-4">
        <Card>
          <CardContent className="p-3 min-h-[64px]">
            <p className="text-sm text-gray-500">Transcription</p>
            <div className="mt-2 text-gray-800 text-base break-words min-h-[24px]">
              {displayTranscript || <span className="text-gray-400">No speech yet</span>}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* CENTERED speak button (fixed center) */}
      <div className="fixed left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 w-full flex flex-col items-center pointer-events-none">
        <div className="flex flex-col items-center pointer-events-auto">
          <div className="relative flex items-center justify-center">
            <MicButton size="big" listening={listening} onClick={toggleListening} ariaLabel={listening ? "Stop listening" : "Start listening"} />
          </div>

          {/* Label under the button */}
          <div className="mt-3 text-sm text-gray-600">Tap to speak</div>
        </div>
      </div>
    </div>
  );
}