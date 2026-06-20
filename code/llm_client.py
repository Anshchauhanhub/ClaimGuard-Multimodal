"""Groq API client (OpenAI-compatible) with retry, rate limiting, and structured JSON output."""
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any

from openai import OpenAI

from config import (
    GROQ_API_KEY, GROQ_BASE_URL, PRIMARY_MODEL, FALLBACK_MODEL,
    MAX_RETRIES, RETRY_BASE_DELAY, SLEEP_BETWEEN_CALLS,
)
from data_loader import load_image_as_base64, get_mime_type


class LLMClient:
    """Manages Groq API calls with retry, rate limiting, and metrics."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY env var."
            )
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
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

    @staticmethod
    def _parse_retry_after(error_text: str) -> Optional[float]:
        """Parse 'Please try again in XmYs' from Groq error messages."""
        import re
        match = re.search(r'try again in (\d+)m([\d.]+)s', error_text)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        match = re.search(r'try again in ([\d.]+)s', error_text)
        if match:
            return float(match.group(1))
        return None

    def _build_image_content_parts(
        self, image_paths: List[str], base_dir: Path
    ) -> List[Dict]:
        """Convert image paths to OpenAI-format image_url content parts."""
        parts = []
        for img_path in image_paths:
            full_path = base_dir / img_path
            b64 = load_image_as_base64(full_path)
            if b64:
                mime = get_mime_type(full_path)
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}"
                    }
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
        """Send a multimodal prompt to Groq and return parsed JSON."""
        image_parts = self._build_image_content_parts(image_paths, base_dir)

        # Build user message content: text + images
        user_content = [
            {"type": "text", "text": analysis_prompt}
        ] + image_parts

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        for attempt in range(1, MAX_RETRIES + 1):
            model_name = self.model_name if attempt <= 2 else self.fallback_model
            try:
                self._rate_limit_wait()

                self.total_calls += 1
                self._last_call_time = time.time()

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )

                # Track token usage
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens or 0
                    self.total_output_tokens += response.usage.completion_tokens or 0

                raw_text = response.choices[0].message.content.strip()

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
                err_str = str(e)
                err_lower = err_str.lower()

                # Detect TPD (tokens per day) limit — parse exact wait time
                if "tokens per day" in err_lower or "tpd" in err_lower:
                    wait = self._parse_retry_after(err_str)
                    if wait is None:
                        wait = 300  # default 5 min wait for daily limits
                    print(f"  ⚠ Daily token limit hit! Waiting {wait}s for reset...")
                    time.sleep(wait + 5)  # add 5s buffer
                elif "429" in err_str or "rate" in err_lower or "quota" in err_lower:
                    wait = self._parse_retry_after(err_str)
                    if wait is None:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  ⚠ Rate limited (attempt {attempt}), waiting {wait}s...")
                    time.sleep(wait)
                elif "500" in err_str or "503" in err_str or "server" in err_lower:
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
