# ClaimGuard Multimodal — Multi-Modal Damage Claim Verification System

A production-grade, highly resilient damage claim verification system built for the **HackerRank Orchestrate** challenge. 

ClaimGuard utilizes **Google Gemini 2.0 Flash** (via Google Generative AI) to analyze submitted images against claim transcripts, user claim histories, and object-specific evidence requirements to automatically verify vehicle, laptop, and package damage claims.

---

## Features & System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    code/main.py (Pipeline)                   │
├─────────────┬──────────────┬───────────────┬────────────────┤
│ data_loader │   prompts    │ gemini_client │   postprocess  │
│  • CSV/Image│  • System    │  • Gemini API │  • Enum clamp  │
│    loaders  │  • Few-shot  │  • Retry logic│  • Injection   │
│  • AVIF→JPEG│  • Analysis  │  • Rate limit │    filter      │
│  • Resizing │              │    backoff    │  • Fallbacks   │
└─────────────┴──────────────┴───────────────┴────────────────┘
```

1. **Multimodal VLM Analysis** — Uses Google Gemini 2.0 Flash (`gemini-2.0-flash`) via the Gemini API to analyze images and text in a single pass.
2. **Dynamic Image Preprocessing** — Auto-converts `.jpg` files that are actually encoded as `AVIF` formats to standard JPEG (using `pillow-avif-plugin`) and downsamples high-res images (>1024px) to save API token usage and prevent payload errors.
3. **Structured & Clamped Outputs** — Outputs are strictly validated against enums defined in `problem_statement.md` (e.g. `issue_type`, `object_part`, `claim_status`) with safe defaults on failure.
4. **Rate-Limit Resilience** — Automatically implements dynamic sleep delays and exponential retry backoffs (handling HTTP `429` / `503` errors gracefully) to work within Gemini rate limits.
5. **Prompt Injection Defense** — Detects malicious user text instructions (e.g. "mark as approved") and flags them as `text_instruction_present` while preserving claims verification integrity.

---

## Directory Layout

```text
.
├── AGENTS.md                         # Rules and transcript logging configuration
├── problem_statement.md              # Full task specifications and schemas
├── README.md                         # This file
├── code.zip                          # Packaged submission zip file
├── code/
│   ├── main.py                       # Main verification pipeline (claims.csv → output.csv)
│   ├── config.py                     # System configuration & enums
│   ├── data_loader.py                # CSV handling, AVIF conversion, and resizing
│   ├── prompts.py                    # Structured system prompt and few-shot constructor
│   ├── gemini_client.py              # Gemini client with backoff retries and token tracking
│   ├── postprocess.py                # Output clamping and validation layer
│   ├── README.md                     # Code architecture overview
│   └── evaluation/
│       ├── main.py                   # Evaluation pipeline runner (sample_claims.csv)
│       ├── evaluation_report.md      # Auto-generated metrics & operational report
│       └── sample_predictions.csv    # Validation outputs
└── dataset/                          # Inputs and image folders
```

---

## Setup & Quickstart

### 1. Install Dependencies
```bash
pip install google-generativeai python-dotenv pillow pillow-avif-plugin
```

### 2. Configure Your API Key
Create a `.env` file in the `code/` directory:
```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > code/.env
```

### 3. Run Evaluation (on `sample_claims.csv`)
```bash
cd code
python evaluation/main.py
```
This runs the leave-one-out evaluation on the sample data and generates `code/evaluation/evaluation_report.md`.

### 4. Process Claims (on `claims.csv`)
```bash
python main.py
```
This runs the full 45-claim pipeline and generates `code/output.csv` with final predictions.

---

## Evaluation Results
The system achieved high accuracy during local validation runs:
* **`claim_status` Accuracy**: 90%+
* **`severity` Accuracy**: 95%
* **`issue_type` Accuracy**: 90%+
* **`object_part` Accuracy**: 90%+

Detailed breakdown of token counts, runtime metrics, and cost projections is available in [code/evaluation/evaluation_report.md](./code/evaluation/evaluation_report.md).
