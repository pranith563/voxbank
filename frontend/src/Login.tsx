import React, { useState } from "react";

interface LoginProps {
  sessionId: string;
  onLoggedIn?: (user: any) => void;
}

export default function Login({ sessionId, onLoggedIn }: LoginProps): JSX.Element {
  const [username, setUsername] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!username || !passphrase) {
      setStatusMsg("Username and passphrase are required.");
      return;
    }

    setIsSubmitting(true);
    setStatusMsg("Logging in...");

    try {
      const usernameNorm = username.trim().toLowerCase();
      const passphraseNorm = passphrase.trim().toLowerCase();

      const payload = {
        username: usernameNorm,
        passphrase: passphraseNorm,
        session_id: sessionId,
      };

      const resp = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const json = await resp.json();

      if (!resp.ok) {
        setStatusMsg(`Login failed: ${json.detail || json.message || resp.status}`);
        return;
      }

      const user = (json && json.user) || { username: usernameNorm };
      setStatusMsg(`Logged in as ${user.username || usernameNorm}.`);

      if (onLoggedIn) {
        onLoggedIn(user);
      }
    } catch (err) {
      console.error("Login request failed", err);
      setStatusMsg("Login failed due to a network or server error.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="max-w-md mx-auto bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
      <h2 className="text-lg font-semibold mb-4">Login to VoxBank</h2>
      <p className="text-sm text-slate-600 mb-6">
        Enter your username and passphrase to access your VoxBank assistant.
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

        <div className="pt-4 flex items-center justify-between">
          <button
            type="submit"
            disabled={isSubmitting}
            className="px-4 py-2 rounded-md bg-slate-900 text-white text-sm font-medium hover:bg-slate-800 disabled:opacity-60"
          >
            {isSubmitting ? "Logging in..." : "Login"}
          </button>
          {statusMsg && <p className="text-xs text-slate-600 ml-3">{statusMsg}</p>}
        </div>
      </form>
    </div>
  );
}

