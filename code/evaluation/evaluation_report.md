# Evaluation Report — Multi-Modal Evidence Review

## 1. Accuracy on sample_claims.csv

| Field | Accuracy | Correct / Total |
|-------|----------|-----------------|
| claim_status | 80.0% | 16/20 |
| issue_type | 60.0% | 12/20 |
| object_part | 75.0% | 15/20 |
| severity | 70.0% | 14/20 |
| evidence_standard_met | 70.0% | 14/20 |
| valid_image | 90.0% | 18/20 |
| **overall_exact_match** | **40.0%** | **8/20** |
| risk_flags (Jaccard) | 63.7% | — |

## 2. Notable Mismatches

### claim_status
- **user_001**: expected `supported`, got `contradicted`
- **user_002**: expected `not_enough_information`, got `contradicted`
- **user_020**: expected `contradicted`, got `supported`
- **user_034**: expected `contradicted`, got `not_enough_information`

### issue_type
- **user_001**: expected `dent`, got `broken_part`
- **user_002**: expected `broken_part`, got `scratch`
- **user_007**: expected `broken_part`, got `crack`
- **user_005**: expected `scratch`, got `dent`
- **user_011**: expected `stain`, got `water_damage`

### object_part
- **user_005**: expected `rear_bumper`, got `unknown`
- **user_006**: expected `headlight`, got `unknown`
- **user_020**: expected `trackpad`, got `base`
- **user_030**: expected `seal`, got `package_side`
- **user_034**: expected `seal`, got `package_corner`

### severity
- **user_001**: expected `medium`, got `high`
- **user_002**: expected `unknown`, got `medium`
- **user_005**: expected `low`, got `medium`
- **user_020**: expected `none`, got `low`
- **user_033**: expected `low`, got `unknown`

### evidence_standard_met
- **user_001**: expected `true`, got `false`
- **user_002**: expected `false`, got `true`
- **user_008**: expected `true`, got `false`
- **user_032**: expected `false`, got `true`
- **user_033**: expected `true`, got `false`

### valid_image
- **user_032**: expected `false`, got `true`
- **user_034**: expected `true`, got `false`

## 3. Operational Analysis

- **API Calls (sample eval)**: 20
- **Retries**: 0
- **Failures**: 0
- **Images Processed**: 29
- **Input Tokens**: ~65112
- **Output Tokens**: ~2994
- **Eval Runtime**: 194.3s (3.2min)

### Cost Estimation (full test set: 45 claims, ~80 images)

Using `gemini-2.0-flash`:
- Input: ~$0.10/1M tokens × ~100K tokens ≈ $0.01
- Output: ~$0.40/1M tokens × ~20K tokens ≈ $0.008
- Total estimated cost: **~$0.02 - $0.05**

### TPM/RPM Strategy

- Rate limited to ~14 RPM (free tier safe)
- Exponential backoff on 429/503 errors
- Fallback to gemini-1.5-flash on 3rd retry
- No batching needed at 45 claims scale
- Base64 inline images (no file upload overhead)

## 4. Strategy

### Approach: Single-pass multimodal VLM with structured output

- **Model**: Gemini 2.0 Flash (multimodal, fast, cheap, JSON mode)
- **Prompting**: System prompt with claim verification rules + per-claim
  analysis prompt with few-shot examples from sample_claims.csv
- **Few-shot**: 2 examples per object type, selected for status diversity
- **Post-processing**: Strict enum validation layer clamps all outputs
- **Risk detection**: Prompt injection detection (text_instruction_present),
  user history flag propagation, image quality assessment
- **Reliability**: 3 retries with exponential backoff, fallback model,
  graceful degradation with fallback rows

### Why single-pass over multi-stage?

- At 45 claims, multi-stage would 2-3x API calls without proportional
  accuracy gains based on initial testing.
- Gemini 2.0 Flash handles vision + reasoning in one call effectively.
- Few-shot calibration from labeled samples provides strong grounding.