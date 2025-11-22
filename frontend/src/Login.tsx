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
    <div className="max-w-md mx-auto glass-panel rounded-2xl p-8 animate-accordion-down">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-foreground mb-2 tracking-wider">LOGIN</h2>
        <p className="text-sm text-muted-foreground">
          Enter credentials to unlock your account.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label className="block text-xs font-medium text-neon-blue uppercase tracking-wider">Username</label>
          <input
            className="w-full bg-secondary/10 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none transition-all placeholder:text-muted-foreground"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            placeholder="Enter username"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-xs font-medium text-neon-blue uppercase tracking-wider">Passphrase</label>
          <input
            type="password"
            className="w-full bg-secondary/10 border border-border rounded-lg px-4 py-3 text-sm text-foreground focus:border-neon-blue focus:ring-1 focus:ring-neon-blue outline-none transition-all placeholder:text-muted-foreground"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            required
            placeholder="Enter passphrase"
          />
        </div>

        <div className="pt-6 flex items-center justify-between border-t border-border">
          {statusMsg && <p className="text-xs text-neon-blue animate-pulse">{statusMsg}</p>}
          <button
            type="submit"
            disabled={isSubmitting}
            className="ml-auto px-8 py-3 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple text-foreground font-bold text-sm hover:shadow-[0_0_20px_rgba(0,243,255,0.4)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? "AUTHENTICATING..." : "LOGIN"}
          </button>
        </div>
      </form>
    </div>
  );
}
