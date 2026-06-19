"""Evaluation pipeline: runs the system on sample_claims.csv and compares against expected outputs."""
import sys
import json
import time
from pathlib import Path
from typing import List, Dict

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, OUTPUT_COLUMNS
from data_loader import (
    load_csv, load_user_history, load_evidence_requirements,
    get_relevant_requirements, parse_image_paths, get_image_ids,
    write_output_csv,
)
from prompts import build_system_prompt, build_analysis_prompt, build_few_shot_examples
from llm_client import LLMClient
from postprocess import validate_and_normalize, build_fallback_row


# ─── Evaluation Metrics ───

EVAL_FIELDS = [
    "claim_status", "issue_type", "object_part", "severity",
    "evidence_standard_met", "valid_image",
]


def compute_accuracy(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Compute per-field accuracy and overall metrics."""
    results = {}
    for field in EVAL_FIELDS:
        correct = 0
        total = len(ground_truth)
        mismatches = []
        for pred, gt in zip(predictions, ground_truth):
            pred_val = str(pred.get(field, "")).strip().lower()
            gt_val = str(gt.get(field, "")).strip().lower()
            if pred_val == gt_val:
                correct += 1
            else:
                mismatches.append({
                    "user_id": gt.get("user_id", "?"),
                    "expected": gt_val,
                    "predicted": pred_val,
                })
        results[field] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "mismatches": mismatches[:5],  # cap for report readability
        }

    # Overall accuracy (all fields match)
    all_correct = 0
    for pred, gt in zip(predictions, ground_truth):
        if all(
            str(pred.get(f, "")).strip().lower() == str(gt.get(f, "")).strip().lower()
            for f in EVAL_FIELDS
        ):
            all_correct += 1
    results["overall_exact_match"] = {
        "accuracy": all_correct / len(ground_truth) if ground_truth else 0.0,
        "correct": all_correct,
        "total": len(ground_truth),
    }

    # Risk flags partial match (Jaccard similarity)
    flag_scores = []
    for pred, gt in zip(predictions, ground_truth):
        pred_flags = set(str(pred.get("risk_flags", "none")).lower().split(";"))
        gt_flags = set(str(gt.get("risk_flags", "none")).lower().split(";"))
        if pred_flags == gt_flags:
            flag_scores.append(1.0)
        else:
            intersection = pred_flags & gt_flags
            union = pred_flags | gt_flags
            flag_scores.append(len(intersection) / len(union) if union else 1.0)

    results["risk_flags_jaccard"] = {
        "mean_similarity": sum(flag_scores) / len(flag_scores) if flag_scores else 0.0,
    }

    return results


def generate_report(
    eval_results: Dict, metrics: Dict, elapsed: float, output_dir: Path
):
    """Generate evaluation_report.md."""
    report_path = output_dir / "evaluation_report.md"

    lines = [
        "# Evaluation Report — Multi-Modal Evidence Review",
        "",
        "## 1. Accuracy on sample_claims.csv",
        "",
        "| Field | Accuracy | Correct / Total |",
        "|-------|----------|-----------------|",
    ]

    for field in EVAL_FIELDS:
        r = eval_results[field]
        lines.append(
            f"| {field} | {r['accuracy']:.1%} | {r['correct']}/{r['total']} |"
        )

    overall = eval_results["overall_exact_match"]
    lines.append(
        f"| **overall_exact_match** | **{overall['accuracy']:.1%}** | "
        f"**{overall['correct']}/{overall['total']}** |"
    )

    rj = eval_results["risk_flags_jaccard"]
    lines.append(
        f"| risk_flags (Jaccard) | {rj['mean_similarity']:.1%} | — |"
    )

    # Mismatches
    lines += ["", "## 2. Notable Mismatches", ""]
    for field in EVAL_FIELDS:
        mismatches = eval_results[field].get("mismatches", [])
        if mismatches:
            lines.append(f"### {field}")
            for m in mismatches:
                lines.append(
                    f"- **{m['user_id']}**: expected `{m['expected']}`, "
                    f"got `{m['predicted']}`"
                )
            lines.append("")

    # Operational analysis
    lines += [
        "## 3. Operational Analysis",
        "",
        f"- **API Calls (sample eval)**: {metrics['total_api_calls']}",
        f"- **Retries**: {metrics['total_retries']}",
        f"- **Failures**: {metrics['total_failures']}",
        f"- **Images Processed**: {metrics['total_images_processed']}",
        f"- **Input Tokens**: ~{metrics['total_input_tokens']}",
        f"- **Output Tokens**: ~{metrics['total_output_tokens']}",
        f"- **Eval Runtime**: {elapsed:.1f}s ({elapsed/60:.1f}min)",
        "",
        "### Cost Estimation (full test set: 45 claims, ~80 images)",
        "",
        "Using `gemini-2.0-flash`:",
        "- Input: ~$0.10/1M tokens × ~100K tokens ≈ $0.01",
        "- Output: ~$0.40/1M tokens × ~20K tokens ≈ $0.008",
        "- Total estimated cost: **~$0.02 - $0.05**",
        "",
        "### TPM/RPM Strategy",
        "",
        "- Rate limited to ~14 RPM (free tier safe)",
        "- Exponential backoff on 429/503 errors",
        "- Fallback to gemini-1.5-flash on 3rd retry",
        "- No batching needed at 45 claims scale",
        "- Base64 inline images (no file upload overhead)",
        "",
        "## 4. Strategy",
        "",
        "### Approach: Single-pass multimodal VLM with structured output",
        "",
        "- **Model**: Gemini 2.0 Flash (multimodal, fast, cheap, JSON mode)",
        "- **Prompting**: System prompt with claim verification rules + per-claim",
        "  analysis prompt with few-shot examples from sample_claims.csv",
        "- **Few-shot**: 2 examples per object type, selected for status diversity",
        "- **Post-processing**: Strict enum validation layer clamps all outputs",
        "- **Risk detection**: Prompt injection detection (text_instruction_present),",
        "  user history flag propagation, image quality assessment",
        "- **Reliability**: 3 retries with exponential backoff, fallback model,",
        "  graceful degradation with fallback rows",
        "",
        "### Why single-pass over multi-stage?",
        "",
        "- At 45 claims, multi-stage would 2-3x API calls without proportional",
        "  accuracy gains based on initial testing.",
        "- Gemini 2.0 Flash handles vision + reasoning in one call effectively.",
        "- Few-shot calibration from labeled samples provides strong grounding.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {report_path}")
    return report_path


def run_evaluation():
    """Evaluate the system against sample_claims.csv."""
    print("=" * 60)
    print("  EVALUATION: Running on sample_claims.csv")
    print("=" * 60)

    dataset_dir = DATASET_DIR
    sample_csv = dataset_dir / "sample_claims.csv"
    if not sample_csv.exists():
        print(f"Error: {sample_csv} not found.")
        sys.exit(1)

    # Load sample claims (these have expected outputs)
    sample_claims = load_csv(sample_csv)
    user_history = load_user_history(dataset_dir / "user_history.csv")
    evidence_reqs = load_evidence_requirements(dataset_dir / "evidence_requirements.csv")

    print(f"  Sample claims: {len(sample_claims)}")

    client = LLMClient()
    system_prompt = build_system_prompt()
    start_time = time.time()

    predictions = []
    for idx, claim in enumerate(sample_claims):
        user_id = claim["user_id"]
        claim_object = claim["claim_object"]
        image_paths = parse_image_paths(claim["image_paths"])
        image_ids = get_image_ids(claim["image_paths"])

        print(f"  [{idx+1}/{len(sample_claims)}] {user_id} | {claim_object}")

        history = user_history.get(user_id, {})
        reqs = get_relevant_requirements(evidence_reqs, claim_object)

        # For eval, use OTHER samples as few-shot (leave-one-out style)
        other_samples = [s for s in sample_claims if s["user_id"] != user_id]
        few_shot = build_few_shot_examples(other_samples, claim_object, max_examples=2)

        analysis_prompt = build_analysis_prompt(
            claim, history, reqs, few_shot, image_ids
        )

        prediction = client.analyze_claim(
            system_prompt=system_prompt,
            analysis_prompt=analysis_prompt,
            image_paths=image_paths,
            base_dir=dataset_dir,
        )

        if prediction:
            row = validate_and_normalize(prediction, claim)
        else:
            row = build_fallback_row(claim, "Model returned no response")

        predictions.append(row)
        print(f"    → {row['claim_status']} (expected: {claim.get('claim_status', '?')})")

    elapsed = time.time() - start_time
    metrics = client.get_metrics()

    # ─── Compute metrics ───
    eval_results = compute_accuracy(predictions, sample_claims)

    # ─── Print summary ───
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    for field in EVAL_FIELDS:
        r = eval_results[field]
        print(f"  {field}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")
    overall = eval_results["overall_exact_match"]
    print(f"  OVERALL EXACT MATCH: {overall['accuracy']:.1%}")
    print(f"  Risk Flags Jaccard: {eval_results['risk_flags_jaccard']['mean_similarity']:.1%}")

    # ─── Generate report ───
    eval_dir = Path(__file__).resolve().parent
    generate_report(eval_results, metrics, elapsed, eval_dir)

    # ─── Save predictions ───
    eval_output = eval_dir / "sample_predictions.csv"
    write_output_csv(predictions, eval_output, OUTPUT_COLUMNS)
    print(f"  Predictions saved: {eval_output}")

    return eval_results


if __name__ == "__main__":
    run_evaluation()
