/**
 * Gemini Speech-to-Text Utility
 * 
 * Converts audio (PCM data) to text using Google Gemini API
 */

const GEMINI_API_KEY = 'AIzaSyAXaXJVyoXRaPSXIB1CxZCelBvlZEw6L7Y';

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
 * Convert Int16Array PCM to base64 encoded WAV
 */
function pcmToBase64Wav(pcmData: Int16Array, sampleRate: number = 16000): string {
  // Convert Int16Array to Uint8Array (little-endian)
  const pcmBytes = new Uint8Array(pcmData.buffer);
  const wavData = createWavFromPCM(pcmBytes, 1, sampleRate, 16);
  
  // Convert to base64
  let binary = '';
  for (let i = 0; i < wavData.length; i++) {
    binary += String.fromCharCode(wavData[i]);
  }
  return btoa(binary);
}

/**
 * Transcribe audio using Gemini Speech-to-Text API
 * 
 * @param audioData - PCM audio data as Int16Array or base64 encoded audio
 * @param sampleRate - Sample rate of the audio (default: 16000)
 * @param language - Language code (default: 'en-US')
 * @returns Transcribed text
 */
export async function transcribeWithGemini(
  audioData: Int16Array | string,
  sampleRate: number = 16000,
  language: string = 'en-US'
): Promise<string> {
  console.log('Starting Gemini STT transcription');
  
  try {
    const { GoogleGenAI } = await import('@google/genai');
    console.log('GoogleGenAI imported successfully');
    
    const ai = new GoogleGenAI({ 
      apiKey: GEMINI_API_KEY
    });
    
    // Convert audio data to base64 WAV if needed
    let base64Audio: string;
    if (typeof audioData === 'string') {
      // Already base64 encoded
      base64Audio = audioData;
    } else {
      // Convert PCM to base64 WAV
      base64Audio = pcmToBase64Wav(audioData, sampleRate);
    }
    
    console.log('Sending audio to Gemini STT, length:', base64Audio.length);
    
    // Use Gemini's speech recognition API
    // Note: The exact API may vary - this is based on common patterns
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash-exp",
      contents: [
        {
          role: "user",
          parts: [
            {
              inlineData: {
                mimeType: "audio/wav",
                data: base64Audio
              }
            },
            {
              text: "Transcribe this audio to text."
            }
          ]
        }
      ],
      config: {
        responseModalities: ["TEXT"]
      }
    });
    
    console.log('Gemini STT response received:', response);
    
    // Extract transcribed text
    const text = response.candidates?.[0]?.content?.parts?.[0]?.text;
    
    if (text) {
      console.log('Transcription successful:', text);
      return text.trim();
    } else {
      console.error('No text found in response');
      throw new Error('No transcription text in response');
    }
    
  } catch (error) {
    console.error('Gemini STT Error:', error);
    throw error;
  }
}

/**
 * Stream transcription for real-time audio chunks
 * This accumulates audio chunks and transcribes when silence is detected
 */
export class GeminiSttStream {
  private audioChunks: Int16Array[] = [];
  private sampleRate: number;
  private language: string;
  
  constructor(sampleRate: number = 16000, language: string = 'en-US') {
    this.sampleRate = sampleRate;
    this.language = language;
  }
  
  /**
   * Add an audio chunk to the stream
   */
  addChunk(chunk: Int16Array): void {
    this.audioChunks.push(chunk);
  }
  
  /**
   * Get accumulated audio as single Int16Array
   */
  private getAccumulatedAudio(): Int16Array {
    const totalLength = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const result = new Int16Array(totalLength);
    let offset = 0;
    
    for (const chunk of this.audioChunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    
    return result;
  }
  
  /**
   * Transcribe accumulated audio and clear chunks
   */
  async transcribe(): Promise<string> {
    if (this.audioChunks.length === 0) {
      return '';
    }
    
    const audioData = this.getAccumulatedAudio();
    const text = await transcribeWithGemini(audioData, this.sampleRate, this.language);
    
    // Clear chunks after transcription
    this.audioChunks = [];
    
    return text;
  }
  
  /**
   * Clear accumulated chunks without transcribing
   */
  clear(): void {
    this.audioChunks = [];
  }
  
  /**
   * Get current chunk count
   */
  getChunkCount(): number {
    return this.audioChunks.length;
  }
}

