/**
 * Gemini Text-to-Speech Utility
 * 
 * Converts text to speech using Google Gemini API
 */

const GEMINI_API_KEY = 'AIzaSyAXaXJVyoXRaPSXIB1CxZCelBvlZEw6L7Y';

/**
 * Get Gemini voice name based on language code
 */
export function getGeminiVoice(language: string): string {
  const voiceMap: Record<string, string> = {
    'en-US': 'Aoede',     // Natural female voice
    'hi-IN': 'Kore',      // Multilingual voice
    'ta-IN': 'Kore',      // Fallback to multilingual
    'ml-IN': 'Kore',
    'te-IN': 'Aoede',
    'kn-IN': 'Kore',
    'es-ES': 'Fenrir',
    'fr-FR': 'Charon',
    'de-DE': 'Puck',
    'ja-JP': 'Kore',
    'zh-CN': 'Kore'
  };
  return voiceMap[language] || 'Aoede';
}

/**
 * Convert PCM audio data to WAV format
 */
function createWavFromPCM(pcmData: Uint8Array, channels: number, sampleRate: number, bitsPerSample: number): Uint8Array {
  const length = pcmData.length;
  const buffer = new ArrayBuffer(44 + length);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // audio format (PCM)
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bitsPerSample / 8, true); // byte rate
  view.setUint16(32, channels * bitsPerSample / 8, true); // block align
  view.setUint16(34, bitsPerSample, true);
  writeString(36, 'data');
  view.setUint32(40, length, true);
  
  // Copy PCM data
  const wavData = new Uint8Array(buffer);
  wavData.set(pcmData, 44);
  
  return wavData;
}

/**
 * Speak text using Gemini TTS API
 * 
 * @param text - Text to speak
 * @param language - Language code (default: 'en-US')
 * @returns Object with success status and audio data
 */
export async function speakGemini(text: string, language: string = 'en-US'): Promise<{
  success: boolean;
  audioData?: string;
  mimeType?: string;
  wavData?: number[];
}> {
  console.log('Starting Gemini TTS for:', text);
  
  try {
    const { GoogleGenAI } = await import('@google/genai');
    console.log('GoogleGenAI imported successfully');
    
    const ai = new GoogleGenAI({ 
      apiKey: GEMINI_API_KEY
    });
    
    const voice = getGeminiVoice(language);
    console.log('Using voice:', voice);
    
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ role: "user", parts: [{ text: text }] }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: voice
            }
          }
        }
      }
    });

    console.log('Gemini response received:', response);
    
    // Extract audio data (raw PCM from Gemini)
    const audioData = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    const mimeType = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.mimeType;
    
    console.log('Audio data length:', audioData?.length);
    console.log('MIME type:', mimeType);
    
    if (audioData) {
      console.log('Audio data found, creating WAV from PCM...');
      
      // Convert base64 to PCM data (raw audio from Gemini)
      const binaryString = atob(audioData);
      const pcmData = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        pcmData[i] = binaryString.charCodeAt(i);
      }
      
      // Create proper WAV file from PCM data (matching Python wave_file function)
      const wavData = createWavFromPCM(pcmData, 1, 24000, 2);
      
      // Create audio blob and play - create new Uint8Array to ensure proper type
      const wavArray = new Uint8Array(wavData);
      const audioBlob = new Blob([wavArray], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      audio.onerror = (e) => {
        console.error('Audio playback error:', e);
        URL.revokeObjectURL(audioUrl);
      };
      
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
      
      await audio.play();
      console.log('Successfully played Gemini PCM audio as WAV');
      
      // Return both original PCM data and WAV data for buttons
      return { 
        success: true, 
        audioData: audioData, // Original base64 PCM
        mimeType: 'audio/wav', // Converted to WAV
        wavData: Array.from(wavData) // WAV file data for buttons
      };
    } else {
      console.error('No audio data found in response');
      throw new Error('No audio data in response');
    }

  } catch (error) {
    console.error('Gemini TTS Error:', error);
    throw error;
  }
}

/**
 * Fallback to browser TTS if Gemini fails
 */
export function speakBrowser(text: string, language: string = 'en-US'): void {
  if (!("speechSynthesis" in window)) {
    console.warn('Browser TTS not supported');
    return;
  }
  
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = language;
  u.rate = 1.0;
  window.speechSynthesis.speak(u);
}

