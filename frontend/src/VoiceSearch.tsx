import { useEffect } from "react";

// shadcn/ui components (adjust paths if needed)
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { MicButton } from "@/components/MicButton";
import { useVoiceSearch } from "@/hooks/useVoiceSearch";

/**
 * VoiceSearch - UI component for voice search interface
 * 
 * Uses the useVoiceSearch hook for all voice-related functionality
 */

export default function VoiceSearch({ wsUrl = "ws://0.0.0.0:8000/ws" }: { wsUrl?: string }): JSX.Element {
  const {
    listening,
    displayTranscript,
    supported,
    toggleListening,
  } = useVoiceSearch({ wsUrl });

  // disable scrolling while mounted
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

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
                placeholder={supported ? "Speak or type to search..." : "Speech not supported in this browser"}
                className="flex-1 text-lg pr-4"
                aria-label="Search"
                readOnly
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