
const MIC_PRIMARY = "#EF4444"; // Tailwind red-600 hex (used for rings)

/**
 * MicButton: shows a CSS ripple when `listening`.
 * size = "big" | "small"
 */
export function MicButton({
  listening,
  onClick,
  size = "big",
  ariaLabel,
}: {
  listening: boolean;
  onClick: () => void;
  size?: "big" | "small";
  ariaLabel?: string;
}) {
  const big = size === "big";
  const btnClass = big
    ? `relative flex items-center justify-center w-24 h-24 rounded-full shadow-2xl transition-transform overflow-visible ${listening ? "scale-110 bg-white" : "bg-white"}`
    : `relative inline-flex items-center justify-center w-6 h-6 rounded-full transition-colors overflow-visible ${listening ? "" : ""}`;

  const innerDotClass = big ? "w-8 h-8 rounded-full bg-red-600 z-30" : "w-2.5 h-2.5 rounded-full bg-gray-700 z-30";

  return (
    <button
      onClick={onClick}
      aria-pressed={listening}
      aria-label={ariaLabel ?? (listening ? "Stop listening" : "Start listening")}
      className={btnClass}
      style={{ border: "none", padding: 0 }}
    >
      {/* Ripple container (three rings) */}
      {listening && (
        <span aria-hidden className={`ripple-container ${big ? "big" : "small"}`} />
      )}

      {/* The inner mic/dot */}
      <span className={innerDotClass} />
      {/* visually-hidden svg mic could go here instead of inner dot */}
      <style>{`
        /* ripple styles */
        .ripple-container {
          position: absolute;
          inset: 0;
          display: block;
          pointer-events: none;
        }

        /* For big button we create larger rings; for small we keep them subtle */
        .ripple-container.big::before,
        .ripple-container.big::after,
        .ripple-container.big > .r1 {
          content: "";
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%) scale(0);
          border-radius: 9999px;
          box-sizing: border-box;
          opacity: 0.9;
          border: 3px solid ${MIC_PRIMARY};
          width: 40px;
          height: 40px;
          z-index: 10;
          animation: ripple-big 2s infinite ease-out;
        }

        /* create three rings by staggering pseudo-elements and an inner element */
        .ripple-container.big::before { animation-delay: 0s; }
        .ripple-container.big::after  { animation-delay: 0.66s; }
        .ripple-container.big > .r1  { animation-delay: 1.33s; }

        .ripple-container.big > .r1 {
          content: "";
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%) scale(0);
          border-radius: 9999px;
          box-sizing: border-box;
          opacity: 0.9;
          border: 3px solid ${MIC_PRIMARY};
          width: 40px;
          height: 40px;
          z-index: 10;
          animation: ripple-big 2s infinite ease-out;
        }

        @keyframes ripple-big {
          0% {
            transform: translate(-50%, -50%) scale(0.35);
            opacity: 0.9;
          }
          40% {
            opacity: 0.6;
          }
          100% {
            transform: translate(-50%, -50%) scale(2.4);
            opacity: 0;
          }
        }

        /* small variant */
        .ripple-container.small::before,
        .ripple-container.small::after,
        .ripple-container.small > .r1 {
          content: "";
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%) scale(0);
          border-radius: 9999px;
          box-sizing: border-box;
          opacity: 0.95;
          border: 2px solid ${MIC_PRIMARY};
          width: 14px;
          height: 14px;
          z-index: 10;
          animation: ripple-small 1.6s infinite ease-out;
        }

        .ripple-container.small::before { animation-delay: 0s; }
        .ripple-container.small::after  { animation-delay: 0.53s; }
        .ripple-container.small > .r1  { animation-delay: 1.06s; }

        @keyframes ripple-small {
          0% { transform: translate(-50%, -50%) scale(0.4); opacity: 0.95; }
          50% { opacity: 0.6; }
          100% { transform: translate(-50%, -50%) scale(2.0); opacity: 0; }
        }

        /* ensure the inner dot sits above rings */
        .${innerDotClass?.replace(/\s+/g, ".") || ""} { position: relative; }
      `}</style>
    </button>
  );
}

