"""Gemini API wrapper with retry, rate limiting, and structured JSON output."""
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any

import google.generativeai as genai

from config import (
    GEMINI_API_KEY, PRIMARY_MODEL, FALLBACK_MODEL,
    MAX_RETRIES, RETRY_BASE_DELAY, SLEEP_BETWEEN_CALLS,
)
from data_loader import load_image_as_base64, get_mime_type


class GeminiClient:
    """Manages Gemini API calls with retry, rate limiting, and metrics."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY env var."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        self.model_name = PRIMARY_MODEL
        self.fallback_model = FALLBACK_MODEL
        self._last_call_time = 0.0

        # ─── Metrics ───
        self.total_calls = 0
        self.total_retries = 0
        self.total_failures = 0
        self.total_images_sent = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _rate_limit_wait(self):
        """Enforce minimum interval between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS - elapsed)

    def _build_image_parts(
        self, image_paths: List[str], base_dir: Path
    ) -> List[Dict]:
        """Convert image paths to inline base64 content parts."""
        parts = []
        for img_path in image_paths:
            full_path = base_dir / img_path
            b64 = load_image_as_base64(full_path)
            if b64:
                mime = get_mime_type(full_path)
                parts.append({
                    "inline_data": {"mime_type": mime, "data": b64}
                })
                self.total_images_sent += 1
            else:
                print(f"  ⚠ Image not found: {full_path}")
        return parts

    def analyze_claim(
        self,
        system_prompt: str,
        analysis_prompt: str,
        image_paths: List[str],
        base_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Send a multimodal prompt to Gemini and return parsed JSON."""
        image_parts = self._build_image_parts(image_paths, base_dir)

        # Build content: text prompt + images
        contents = [analysis_prompt] + image_parts

        for attempt in range(1, MAX_RETRIES + 1):
            model_name = self.model_name if attempt <= 2 else self.fallback_model
            try:
                self._rate_limit_wait()
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=system_prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,  # low temp for consistency
                    },
                )

                self.total_calls += 1
                self._last_call_time = time.time()

                response = model.generate_content(contents)

                # Track token usage if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    self.total_input_tokens += getattr(um, "prompt_token_count", 0)
                    self.total_output_tokens += getattr(um, "candidates_token_count", 0)

                raw_text = response.text.strip()
                # Clean markdown fences if model accidentally adds them
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[-1]
                if raw_text.endswith("```"):
                    raw_text = raw_text.rsplit("```", 1)[0]
                raw_text = raw_text.strip()

                result = json.loads(raw_text)
                return result

            except json.JSONDecodeError as e:
                self.total_retries += 1
                print(f"  ⚠ JSON parse error (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * attempt)
            except Exception as e:
                self.total_retries += 1
                err_str = str(e).lower()
                if "429" in err_str or "resource" in err_str or "quota" in err_str:
                    wait = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  ⚠ Rate limited (attempt {attempt}), waiting {wait}s...")
                    time.sleep(wait)
                elif "500" in err_str or "503" in err_str:
                    wait = RETRY_BASE_DELAY * attempt
                    print(f"  ⚠ Server error (attempt {attempt}), waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ✗ Error (attempt {attempt}): {e}")
                    traceback.print_exc()
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY)

        self.total_failures += 1
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Return usage metrics for the operational report."""
        return {
            "total_api_calls": self.total_calls,
            "total_retries": self.total_retries,
            "total_failures": self.total_failures,
            "total_images_processed": self.total_images_sent,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
