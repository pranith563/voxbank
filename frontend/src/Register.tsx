import React, { useState, useRef } from "react";
import { Mic, Square, Play } from "lucide-react";

interface RegisterResult {
  status: string;
  user?: any;
  error?: string;
}

interface RegisterProps {
  sessionId: string;
  onRegistered?: (user: any) => void;
}

export default function Register({ sessionId, onRegistered }: RegisterProps): JSX.Element {
  const [username, setUsername] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [address, setAddress] = useState("");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [audioBase64, setAudioBase64] = useState<string | null>(null);

  async function startRecording() {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mr.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      mr.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        setAudioPreviewUrl(url);

        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result as string;
          const base64 = result.split(",")[1];
          setAudioBase64(base64);
        };
        reader.readAsDataURL(blob);

        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorderRef.current = mr;
      mr.start();
      setIsRecording(true);
      setStatusMsg("Recording voice sample...");
    } catch (e) {
      console.error("Failed to start recording", e);
      setStatusMsg("Microphone access failed. Please check permissions.");
    }
  }

  function stopRecording() {
    if (!isRecording || !mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current = null;
    setIsRecording(false);
    setStatusMsg("Recording stopped. You can preview the audio or submit the form.");
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!username || !passphrase) {
      setStatusMsg("Username and passphrase are required.");
      return;
    }
    setIsSubmitting(true);
    setStatusMsg("Registering user...");

    try {
      const usernameNorm = username.trim().toLowerCase();
      const passphraseNorm = passphrase.trim().toLowerCase();

      const payload = {
        username: usernameNorm,
        passphrase: passphraseNorm,
        email: email || null,
        full_name: fullName || null,
        phone_number: phoneNumber || null,
        address: address || null,
        date_of_birth: null,
        audio_data: audioBase64,
        session_id: sessionId,
      };

      // Call orchestrator auth register endpoint, which proxies to mock-bank
      const resp = await fetch("http://localhost:8000/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const json = await resp.json();
      if (!resp.ok) {
        setStatusMsg(`Registration failed: ${json.detail || json.message || resp.status}`);
      } else {
        const result = json as RegisterResult;
        const user = result.user || { username };
        setStatusMsg(`Registered successfully as ${user.username || username}.`);
        if (onRegistered) {
          onRegistered(user);
        }
      }
    } catch (err) {
      console.error("Register request failed", err);
      setStatusMsg("Registration failed due to a network or server error.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto glass-panel rounded-2xl p-8 animate-accordion-down">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-foreground mb-2 tracking-wider">JOIN VOXBANK</h2>
        <p className="text-sm text-muted-foreground">
          Create your secure identity. Voice authentication optional.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-neon-blue uppercase tracking-wider">Username</label>
            <input
              className="w-full bg-background/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none transition-all placeholder:text-muted-foreground"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              placeholder="jdoe"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-xs font-medium text-neon-blue uppercase tracking-wider">Passphrase</label>
            <input
              type="password"
              className="w-full bg-background/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none transition-all placeholder:text-muted-foreground"
              value={passphrase}
              onChange={(e) => setPassphrase(e.target.value)}
              required
              placeholder="••••••••"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider">Email</label>
            <input
              className="w-full bg-background/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue outline-none transition-all placeholder:text-muted-foreground"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="john@example.com"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider">Full Name</label>
            <input
              className="w-full bg-background/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue outline-none transition-all placeholder:text-muted-foreground"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="John Doe"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider">Phone Number</label>
            <input
              className="w-full bg-background/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue outline-none transition-all placeholder:text-muted-foreground"
              value={phoneNumber}
              onChange={(e) => setPhoneNumber(e.target.value)}
              placeholder="+91 0000000000"
            />
          </div>
          <div className="space-y-2">
            <label className="block text-xs font-medium text-muted-foreground uppercase tracking-wider">Address</label>
            <input
              className="w-full bg-background/50 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue outline-none transition-all placeholder:text-muted-foreground"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="123 Future St"
            />
          </div>
        </div>

        <div className="mt-8 p-6 border border-border rounded-xl bg-secondary/10">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-neon-purple mb-1">Voice Security</h3>
              <p className="text-xs text-muted-foreground">
                Record a phrase for biometric authentication.
              </p>
            </div>
            {isRecording && <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />}
          </div>

          <div className="flex items-center gap-4">
            {!isRecording ? (
              <button
                type="button"
                onClick={startRecording}
                className="flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30 transition-all text-xs font-medium"
              >
                <Mic className="w-4 h-4" />
                Record Sample
              </button>
            ) : (
              <button
                type="button"
                onClick={stopRecording}
                className="flex items-center gap-2 px-4 py-2 rounded-full bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-all text-xs font-medium"
              >
                <Square className="w-4 h-4 fill-current" />
                Stop Recording
              </button>
            )}

            {audioPreviewUrl && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-full bg-neon-blue/10 border border-neon-blue/30">
                <Play className="w-3 h-3 text-neon-blue" />
                <span className="text-xs text-neon-blue">Sample Recorded</span>
                <audio src={audioPreviewUrl} className="hidden" />
              </div>
            )}
          </div>
        </div>

        <div className="pt-6 flex items-center justify-between border-t border-border">
          {statusMsg && <p className="text-xs text-neon-blue animate-pulse">{statusMsg}</p>}
          <button
            type="submit"
            disabled={isSubmitting}
            className="ml-auto px-8 py-3 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple text-white font-bold text-sm hover:shadow-[0_0_20px_rgba(0,243,255,0.4)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? "PROCESSING..." : "CREATE ACCOUNT"}
          </button>
        </div>
      </form>
    </div>
  );
}
