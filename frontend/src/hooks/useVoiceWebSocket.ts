import { useRef, useCallback } from "react";

/**
 * WebSocket hook for voice search communication
 * 
 * Handles WebSocket connection, message sending/receiving, and server message processing
 */

export interface ServerMessage {
  type: "partial" | "final" | "reply" | "meta_ack" | "error";
  text?: string;
  message?: string;
  [key: string]: any;
}

export interface VoiceWebSocketCallbacks {
  onPartial?: (text: string) => void;
  onFinal?: (text: string) => void;
  onReply?: (text: string) => void;
  onError?: (message: string) => void;
}

export interface UseVoiceWebSocketOptions {
  wsUrl?: string;
  sampleRate?: number;
  channels?: number;
  encoding?: string;
  lang?: string;
  callbacks?: VoiceWebSocketCallbacks;
}

export interface UseVoiceWebSocketReturn {
  wsRef: React.MutableRefObject<WebSocket | null>;
  isConnected: () => boolean;
  openWebSocket: () => void;
  closeWebSocket: () => void;
  sendBinary: (data: ArrayBuffer) => void;
  sendJSON: (data: any) => void;
}

export function useVoiceWebSocket(options: UseVoiceWebSocketOptions = {}): UseVoiceWebSocketReturn {
  const {
    wsUrl = "ws://0.0.0.0:8765/ws",
    sampleRate = 16000,
    channels = 1,
    encoding = "pcm16",
    lang = "en",
    callbacks = {},
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const partialThrottleRef = useRef<number>(0);

  const handleServerMessage = useCallback((msg: ServerMessage) => {
    const t = msg.type;
    if (t === "partial") {
      // throttle partial updates slightly to avoid UI jitter
      partialThrottleRef.current += 1;
      if (partialThrottleRef.current % 2 === 0 && callbacks.onPartial) {
        callbacks.onPartial(msg.text || "");
      }
    } else if (t === "final") {
      if (callbacks.onFinal) {
        callbacks.onFinal(msg.text || "");
      }
    } else if (t === "reply") {
      if (callbacks.onReply) {
        callbacks.onReply(msg.text || "");
      }
    } else if (t === "meta_ack") {
      console.debug("Server meta_ack", msg);
    } else if (t === "error") {
      const errorMsg = msg.message || "Unknown error";
      console.error("Server error:", errorMsg);
      if (callbacks.onError) {
        callbacks.onError(errorMsg);
      }
    } else {
      console.debug("Unknown server message", msg);
    }
  }, [callbacks]);

  const openWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      console.debug("WebSocket already open/connecting");
      return;
    }

    console.info("Opening websocket to:", wsUrl);
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.info("WS connected");
      // send meta: we send sampleRate=16000 because we downsample to 16k
      const meta = { type: "meta", sampleRate, channels, encoding, lang };
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
  }, [wsUrl, sampleRate, channels, encoding, lang, handleServerMessage]);

  const closeWebSocket = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
      try {
        delete (window as any).debug_ws;
      } catch {}
    }
  }, []);

  const isConnected = useCallback(() => {
    return wsRef.current?.readyState === WebSocket.OPEN;
  }, []);

  const sendBinary = useCallback((data: ArrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(data);
      } catch (err) {
        console.error("WS send error", err);
      }
    }
  }, []);

  const sendJSON = useCallback((data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(data));
      } catch (err) {
        console.error("WS send JSON error", err);
      }
    }
  }, []);

  return {
    wsRef,
    isConnected,
    openWebSocket,
    closeWebSocket,
    sendBinary,
    sendJSON,
  };
}

