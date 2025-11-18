# orchestrator/src/gemini_llm_client.py
from dotenv import load_dotenv
load_dotenv()

"""
Gemini LLM client for async use with LLMAgent.

Behavior:
- Preferred: use official google-genai async SDK if installed.
- Fallback: use direct REST calls via httpx to the Generative Language endpoint.
- Exposes: async generate(prompt: str, max_tokens: int = 512) -> str

Environment:
- Set GEMINI_API_KEY to your Gemini API key (from Google AI Studio / Vertex AI).
- Optionally set GEMINI_MODEL (defaults to "gemini-2.5-pro").

Install (recommended SDK):
    pip install google-genai httpx

If you prefer REST-only:
    pip install httpx

Notes:
- The SDK and REST endpoints can evolve; this code handles the current common patterns.
- For production, prefer the official SDK and service account/ADC where appropriate.
"""

import os
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or os.environ.get("GEMINI_TOKEN")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Try to import the google-genai SDK
from google import genai  # type: ignore
from google.genai import types  # type: ignore

class GeminiLLMClient:
    """
    High-level client that tries using the google-genai SDK, otherwise uses REST via httpx.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, timeout: int = 60):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key to GeminiLLMClient."
            )
        self.model = model
        self.timeout = timeout

        try:
            # create SDK client
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Using google-genai SDK for Gemini LLM client.")
            self._mode = "sdk"
        except Exception as e:
            logger.exception("Failed to initialize google-genai client; falling back to REST. Error: %s", e)
            self.client = None
            self._mode = "rest"
            
    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
            """
            Generate text using Gemini (async).
            """

            try:
                # Async API path
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                # Response extraction
                if hasattr(response, "text"):
                    return response.text

                if hasattr(response, "candidates") and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, "content") and cand.content:
                        for part in cand.content:
                            if hasattr(part, "text"):
                                return part.text

                return str(response)

            except Exception as e:
                logger.exception("Gemini generate() failed: %s", e)
                return "Sorry, I'm having trouble generating a response."

# Example usage snippet (async)
async def _quick_demo():
    client = GeminiLLMClient()
    out = await client.generate("You are a friendly assistant. Tell me a love quote", max_tokens=200)
    print("Gemini says:", out)

if __name__ == "__main__":
    asyncio.run(_quick_demo())
