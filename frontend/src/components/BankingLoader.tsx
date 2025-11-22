export default function BankingLoader() {
    return (
        <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background text-foreground">
            <div className="relative flex items-center justify-center mb-8">
                {/* Spinning Outer Ring (Loader) */}
                <div className="absolute w-32 h-32 md:w-40 md:h-40 rounded-full border-t-2 border-b-2 border-neon-blue animate-spin" />

                {/* Pulsing Middle Ring */}
                <div className="absolute w-24 h-24 md:w-32 md:h-32 rounded-full border border-neon-blue/30 animate-pulse" />

                {/* Inner Core */}
                <div className="relative z-10 flex flex-col items-center">
                    <div className="text-4xl font-bold tracking-tighter text-neon-blue drop-shadow-[0_0_25px_rgba(0,243,255,0.8)] animate-pulse">
                        Vox
                    </div>
                </div>
            </div>
        </div>
    );
}
