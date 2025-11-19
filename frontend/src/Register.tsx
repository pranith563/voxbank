import React, { useState, useRef } from "react";

interface RegisterResult {
  status: string;
  user?: any;
  error?: string;
}

interface RegisterProps {
  onRegistered?: (user: any) => void;
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
      const resp = await fetch("/api/auth/register", {
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
    <div className="max-w-xl mx-auto bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
      <h2 className="text-lg font-semibold mb-4">Register for VoxBank</h2>
      <p className="text-sm text-slate-600 mb-6">
        Create a user with a passphrase and an optional voice sample (for future voice authentication).
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Username</label>
          <input
            className="w-full border rounded-md px-3 py-2 text-sm"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Passphrase</label>
          <input
            type="password"
            className="w-full border rounded-md px-3 py-2 text-sm"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            required
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Email</label>
            <input
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Full Name</label>
            <input
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Phone Number</label>
            <input
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={phoneNumber}
              onChange={(e) => setPhoneNumber(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Address</label>
            <input
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
            />
          </div>
        </div>

        <div className="mt-4 border-t pt-4">
          <h3 className="text-sm font-semibold text-slate-800 mb-2">Voice Sample (optional)</h3>
          <p className="text-xs text-slate-500 mb-2">
            Record a short phrase in your normal voice. This audio will be converted to an embedding and stored for future voice authentication.
          </p>
          <div className="flex items-center gap-3">
            {!isRecording ? (
              <button
                type="button"
                onClick={startRecording}
                className="px-3 py-2 text-sm rounded-md bg-red-600 text-white hover:bg-red-700"
              >
                Start Recording
              </button>
            ) : (
              <button
                type="button"
                onClick={stopRecording}
                className="px-3 py-2 text-sm rounded-md bg-slate-700 text-white hover:bg-slate-800"
              >
                Stop Recording
              </button>
            )}
            {audioPreviewUrl && (
              <audio controls src={audioPreviewUrl} className="h-8">
                Your browser does not support audio playback.
              </audio>
            )}
          </div>
        </div>

        <div className="pt-4 flex items-center justify-between">
          <button
            type="submit"
            disabled={isSubmitting}
            className="px-4 py-2 rounded-md bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-60"
          >
            {isSubmitting ? "Registering..." : "Register"}
          </button>
          {statusMsg && <p className="text-xs text-slate-600 ml-3">{statusMsg}</p>}
        </div>
      </form>
    </div>
  );
}
