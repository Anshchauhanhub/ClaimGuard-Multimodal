# Multi-Modal Evidence Review System

## Overview
A production-grade damage claim verification system that uses Google Gemini 2.0 Flash (multimodal VLM) to analyze submitted images against user conversations, user history, and evidence requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (Pipeline)                    │
├─────────────┬──────────────┬───────────────┬────────────┤
│ data_loader │   prompts    │ gemini_client │ postprocess│
│  CSV/Image  │  System +    │  Retry/Rate   │  Validate  │
│  loading    │  Few-shot +  │  Limit/Metric │  & Clamp   │
│             │  Analysis    │  Tracking     │  to Enums  │
├─────────────┴──────────────┴───────────────┴────────────┤
│                    config.py (Settings)                   │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Single-pass multimodal**: One VLM call per claim (images + text together). At 45 claims, multi-stage pipelines triple API calls without proportional gains.

2. **Few-shot calibration**: 2 labeled examples per object type from `sample_claims.csv`, selected for status diversity (supported/contradicted/not_enough_info).

3. **Strict post-processing**: Every output field is validated against allowed enum values. Invalid values are clamped to safe defaults.

4. **Prompt injection defense**: System prompt explicitly instructs the model to flag `text_instruction_present` when user messages contain approval/rejection instructions.

5. **Inline base64 images**: Avoids file upload API overhead and quota issues.

## Setup

```bash
# Install dependency
pip install google-generativeai

# Set your API key
export GEMINI_API_KEY="your_key_here"
```

## Usage

```bash
# Run evaluation on sample_claims.csv first
cd code
python evaluation/main.py

# Process test claims and generate output.csv
python main.py
```

## Output
- `output.csv` — predictions for all 45 test claims
- `evaluation/evaluation_report.md` — accuracy metrics + operational analysis
- `evaluation/sample_predictions.csv` — predictions on sample data

## Files
| File | Purpose |
|------|---------|
| `main.py` | Pipeline entry point |
| `config.py` | Paths, model config, allowed values |
| `data_loader.py` | CSV/image loading utilities |
| `prompts.py` | System prompt + few-shot + analysis prompt |
| `gemini_client.py` | API client with retry/rate-limit/metrics |
| `postprocess.py` | Output validation and normalization |
| `evaluation/main.py` | Evaluation pipeline with metrics |
