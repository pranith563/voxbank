import { useRef, useCallback } from "react";

/**
 * Audio sampling hook for voice input processing
 * 
 * Handles audio capture, processing, downsampling, and silence detection
 */

export interface AudioSamplingCallbacks {
  onAudioChunk: (int16Data: Int16Array) => void;
  onSilenceDetected?: () => void;
}

export interface UseAudioSamplingOptions {
  speechThreshold?: number;
  silenceMs?: number;
  minEndIntervalMs?: number;
  clientSendSampleRate?: number;
  bufferSize?: number;
  callbacks: AudioSamplingCallbacks;
}

export interface AudioSamplingRefs {
  audioCtxRef: React.MutableRefObject<AudioContext | null>;
  sourceRef: React.MutableRefObject<MediaStreamAudioSourceNode | null>;
  processorRef: React.MutableRefObject<ScriptProcessorNode | null>;
  mediaStreamRef: React.MutableRefObject<MediaStream | null>;
}

export interface UseAudioSamplingReturn {
  refs: AudioSamplingRefs;
  startSampling: (stream: MediaStream) => void;
  stopSampling: () => void;
  resetSilenceDetection: () => void;
}

/**
 * Convert Float32 PCM to Int16 ArrayBuffer
 */
export function floatTo16BitPCM(float32Array: Float32Array): Int16Array {
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

/**
 * Downsample float32 buffer from srcRate -> dstRate (returns Float32Array)
 */
export function downsampleBuffer(buffer: Float32Array, srcRate: number, dstRate: number): Float32Array {
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

export function useAudioSampling(options: UseAudioSamplingOptions): UseAudioSamplingReturn {
  const {
    speechThreshold = 0.01,
    silenceMs = 700,
    minEndIntervalMs = 600,
    clientSendSampleRate = 16000,
    bufferSize = 4096,
    callbacks,
  } = options;

  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Silence detection state
  const speakingRef = useRef<boolean>(false);
  const lastVoiceTimeRef = useRef<number>(0);
  const silenceSentRef = useRef<boolean>(false);
  const lastEndSentTimeRef = useRef<number>(0);

  const resetSilenceDetection = useCallback(() => {
    speakingRef.current = false;
    lastVoiceTimeRef.current = performance.now();
    silenceSentRef.current = false;
    lastEndSentTimeRef.current = 0;
  }, []);

  const startSampling = useCallback((stream: MediaStream) => {
    console.info("Starting audio sampling");
    mediaStreamRef.current = stream;
    audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();

    // create source
    sourceRef.current = audioCtxRef.current.createMediaStreamSource(stream);

    // ScriptProcessorNode is easier to use cross-browser; buffer 4096 gives ~85ms at 48k
    const processor = audioCtxRef.current.createScriptProcessor(bufferSize, 1, 1);
    processorRef.current = processor;

    // reset silence detection state
    resetSilenceDetection();

    processor.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0);

      // --- compute RMS for silence detection ---
      let sum = 0;
      for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
      const rms = Math.sqrt(sum / float32.length);

      const now = performance.now();

      // if above threshold -> mark speaking and reset silence flags
      if (rms >= speechThreshold) {
        speakingRef.current = true;
        lastVoiceTimeRef.current = now;
        silenceSentRef.current = false;
      } else {
        // if we were speaking and silence duration exceeded threshold -> send end
        if (speakingRef.current) {
          const silenceDur = now - lastVoiceTimeRef.current;
          if (silenceDur >= silenceMs && !silenceSentRef.current) {
            const lastEnd = lastEndSentTimeRef.current;
            if (now - lastEnd > minEndIntervalMs) {
              // trigger silence detected callback
              if (callbacks.onSilenceDetected) {
                callbacks.onSilenceDetected();
              }
              silenceSentRef.current = true;
              lastEndSentTimeRef.current = now;
              speakingRef.current = false;
              lastVoiceTimeRef.current = 0;
              console.debug("Silence detected");
            }
          }
        }
      }

      // --- downsample to target rate and convert to Int16 ---
      const int16 = floatTo16BitPCM(
        downsampleBuffer(float32, audioCtxRef.current!.sampleRate, clientSendSampleRate)
      );
      
      // send processed audio chunk
      callbacks.onAudioChunk(int16);
    };

    sourceRef.current.connect(processor);
    processor.connect(audioCtxRef.current.destination);
  }, [speechThreshold, silenceMs, minEndIntervalMs, clientSendSampleRate, bufferSize, callbacks, resetSilenceDetection]);

  const stopSampling = useCallback(() => {
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
  }, []);

  return {
    refs: {
      audioCtxRef,
      sourceRef,
      processorRef,
      mediaStreamRef,
    },
    startSampling,
    stopSampling,
    resetSilenceDetection,
  };
}

