import { cn } from "@/lib/utils";
import { Settings, LogOut, User, Moon, Sun, Check } from "lucide-react";
import { useEffect, useRef } from "react";

interface HeaderProps {
    view: "assistant" | "register" | "login";
    setView: (view: "assistant" | "register" | "login") => void;
    currentUser: string | null;
    onLogout: () => void;
    settingsOpen: boolean;
    setSettingsOpen: (open: boolean | ((prev: boolean) => boolean)) => void;
    theme: "dark" | "light";
    toggleTheme: (theme: "dark" | "light") => void;
    language: string;
    setLanguage: (lang: string) => void;
    voiceType: string;
    setVoiceType: (voice: string) => void;
}

export default function Header({
    view,
    setView,
    currentUser,
    onLogout,
    settingsOpen,
    setSettingsOpen,
    theme,
    toggleTheme,
    language,
    setLanguage,
    voiceType,
    setVoiceType,
}: HeaderProps) {
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Click outside to close dropdown
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setSettingsOpen(false);
            }
        }
        if (settingsOpen) {
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [settingsOpen, setSettingsOpen]);

    return (
        <header className="fixed top-0 left-0 w-full h-20 z-50 glass-panel border-b border-border transition-colors duration-300">
            <div className="max-w-7xl mx-auto h-full px-6 flex items-center justify-between">
                {/* Logo Area */}
                <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView("assistant")}>
                    {/* Text Logo "Vox" */}
                    <div className="flex items-center justify-center">
                        <span className="text-3xl font-bold tracking-tighter text-neon-blue drop-shadow-[0_0_10px_rgba(0,243,255,0.3)]">
                            Vox
                        </span>
                    </div>
                    <div className="flex flex-col justify-center h-full pt-1">
                        <h1 className="text-xl font-semibold tracking-widest text-muted-foreground">
                            BANK
                        </h1>
                    </div>
                </div>

                {/* Navigation & Actions */}
                <div className="flex items-center gap-4 relative" ref={dropdownRef}>
                    {/* Navigation Pills */}
                    <div className="hidden md:flex items-center bg-secondary/20 rounded-full p-1 border border-border">
                        <button
                            onClick={() => setView("assistant")}
                            className={cn(
                                "px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-300",
                                view === "assistant"
                                    ? "bg-neon-blue/20 text-neon-blue shadow-[0_0_10px_rgba(0,243,255,0.2)]"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            Assistant
                        </button>

                        {!currentUser ? (
                            <>
                                <button
                                    onClick={() => setView("login")}
                                    className={cn(
                                        "px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-300",
                                        view === "login"
                                            ? "bg-neon-blue/20 text-neon-blue shadow-[0_0_10px_rgba(0,243,255,0.2)]"
                                            : "text-muted-foreground hover:text-foreground"
                                    )}
                                >
                                    Login
                                </button>
                                <button
                                    onClick={() => setView("register")}
                                    className={cn(
                                        "px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-300",
                                        view === "register"
                                            ? "bg-neon-blue/20 text-neon-blue shadow-[0_0_10px_rgba(0,243,255,0.2)]"
                                            : "text-muted-foreground hover:text-foreground"
                                    )}
                                >
                                    Register
                                </button>
                            </>
                        ) : (
                            <div className="px-4 py-1.5 flex items-center gap-2 text-xs text-neon-green">
                                <User className="w-3 h-3" />
                                <span>{currentUser}</span>
                            </div>
                        )}
                    </div>

                    {/* Settings Toggle */}
                    <button
                        onClick={() => setSettingsOpen((v) => !v)}
                        className={cn(
                            "w-10 h-10 rounded-full flex items-center justify-center border border-border transition-all hover:bg-secondary/20",
                            settingsOpen ? "bg-secondary/20 text-neon-blue" : "text-muted-foreground"
                        )}
                    >
                        <Settings className="w-5 h-5" />
                    </button>

                    {/* Settings Dropdown */}
                    {settingsOpen && (
                        <div className="absolute top-14 right-0 w-64 glass-panel rounded-xl overflow-hidden animate-in fade-in slide-in-from-top-2 shadow-2xl border border-border">
                            {/* Language Section */}
                            <div className="p-3 border-b border-border">
                                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-2">Language</span>
                                <select
                                    className="w-full bg-secondary/20 border border-border rounded-md px-2 py-1.5 text-xs text-foreground focus:border-neon-blue outline-none"
                                    value={language}
                                    onChange={(e) => {
                                        setLanguage(e.target.value);
                                        setSettingsOpen(false);
                                    }}
                                >
                                    <option value="en-US">English</option>
                                    <option value="en-GB">Hindi</option>
                                    <option value="en-IN">Telugu</option>
                                    <option value="en-IN">Kannada</option>
                                </select>
                            </div>

                            {/* Voice Section */}
                            <div className="p-3 border-b border-border">
                                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-2">Voice Type</span>
                                <select
                                    className="w-full bg-secondary/20 border border-border rounded-md px-2 py-1.5 text-xs text-foreground focus:border-neon-blue outline-none"
                                    value={voiceType}
                                    onChange={(e) => {
                                        setVoiceType(e.target.value);
                                        setSettingsOpen(false);
                                    }}
                                >
                                    <option value="female">Female</option>
                                    <option value="male">Male</option>
                                </select>
                            </div>

                            {/* Tone Section */}
                            <div className="p-3 border-b border-border">
                                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-2">Voice Type</span>
                                <select
                                    className="w-full bg-secondary/20 border border-border rounded-md px-2 py-1.5 text-xs text-foreground focus:border-neon-blue outline-none"
                                    value={voiceType}
                                    onChange={(e) => {
                                        setVoiceType(e.target.value);
                                        setSettingsOpen(false);
                                    }}
                                >
                                    <option value="female">Formal</option>
                                    <option value="male">Friendly</option>
                                    <option value="male">Supportive</option>
                                    <option value="male">Clear</option>
                                </select>
                            </div>

                            {/* Theme Section */}
                            <div className="p-3">
                                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider block mb-2">Theme</span>
                                <div className="flex flex-col gap-1">
                                    <button
                                        onClick={() => {
                                            toggleTheme("dark");
                                            setSettingsOpen(false);
                                        }}
                                        className={cn(
                                            "flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors",
                                            theme === "dark" ? "bg-secondary/20 text-foreground" : "text-muted-foreground hover:bg-secondary/10 hover:text-foreground"
                                        )}
                                    >
                                        <div className="flex items-center gap-2">
                                            <Moon className="w-4 h-4" />
                                            <span>Dark</span>
                                        </div>
                                        {theme === "dark" && <Check className="w-3 h-3 text-neon-blue" />}
                                    </button>
                                    <button
                                        onClick={() => {
                                            toggleTheme("light");
                                            setSettingsOpen(false);
                                        }}
                                        className={cn(
                                            "flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors",
                                            theme === "light" ? "bg-secondary/20 text-foreground" : "text-muted-foreground hover:bg-secondary/10 hover:text-foreground"
                                        )}
                                    >
                                        <div className="flex items-center gap-2">
                                            <Sun className="w-4 h-4" />
                                            <span>Light</span>
                                        </div>
                                        {theme === "light" && <Check className="w-3 h-3 text-neon-blue" />}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Logout */}
                    {currentUser && (
                        <button
                            onClick={onLogout}
                            className="w-10 h-10 rounded-full flex items-center justify-center border border-white/10 text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-all"
                            title="Logout"
                        >
                            <LogOut className="w-5 h-5" />
                        </button>
                    )}
                </div>
            </div>
        </header>
    );
}
