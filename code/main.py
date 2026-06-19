"""Main entry point: Multi-Modal Evidence Review pipeline."""
import sys
import time
from pathlib import Path

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASET_DIR, OUTPUT_CSV, OUTPUT_COLUMNS
from data_loader import (
    load_csv, load_user_history, load_evidence_requirements,
    get_relevant_requirements, parse_image_paths, get_image_ids,
    write_output_csv,
)
from prompts import build_system_prompt, build_analysis_prompt, build_few_shot_examples
from llm_client import LLMClient
from postprocess import validate_and_normalize, build_fallback_row


def process_claims(
    claims_csv: Path,
    output_path: Path,
    dataset_dir: Path,
    sample_claims_csv: Path = None,
):
    """Process all claims and write output.csv."""
    print("=" * 60)
    print("  Multi-Modal Evidence Review Pipeline")
    print("=" * 60)

    # ─── Load data ───
    claims = load_csv(claims_csv)
    user_history = load_user_history(dataset_dir / "user_history.csv")
    evidence_reqs = load_evidence_requirements(dataset_dir / "evidence_requirements.csv")

    # Load sample claims for few-shot examples
    sample_csv = sample_claims_csv or (dataset_dir / "sample_claims.csv")
    sample_claims = load_csv(sample_csv) if sample_csv.exists() else []

    print(f"  Claims to process: {len(claims)}")
    print(f"  Users with history: {len(user_history)}")
    print(f"  Evidence requirements: {len(evidence_reqs)}")
    print(f"  Sample claims for few-shot: {len(sample_claims)}")
    print()

    # ─── Initialize Gemini ───
    client = LLMClient()
    system_prompt = build_system_prompt()

    # ─── Process each claim ───
    results = []
    start_time = time.time()

    for idx, claim in enumerate(claims):
        user_id = claim["user_id"]
        claim_object = claim["claim_object"]
        image_paths = parse_image_paths(claim["image_paths"])
        image_ids = get_image_ids(claim["image_paths"])

        print(f"[{idx+1}/{len(claims)}] {user_id} | {claim_object} | "
              f"{len(image_paths)} images | IDs: {', '.join(image_ids)}")

        # Get user history and requirements
        history = user_history.get(user_id, {})
        reqs = get_relevant_requirements(evidence_reqs, claim_object)

        # Build few-shot examples
        few_shot = build_few_shot_examples(sample_claims, claim_object, max_examples=2)

        # Build analysis prompt
        analysis_prompt = build_analysis_prompt(
            claim, history, reqs, few_shot, image_ids
        )

        # Call Gemini
        prediction = client.analyze_claim(
            system_prompt=system_prompt,
            analysis_prompt=analysis_prompt,
            image_paths=image_paths,
            base_dir=dataset_dir,
        )

        if prediction:
            row = validate_and_normalize(prediction, claim)
            status_icon = {
                "supported": "✓",
                "contradicted": "✗",
                "not_enough_information": "?"
            }.get(row["claim_status"], "?")
            print(f"  → {status_icon} {row['claim_status']} | "
                  f"{row['issue_type']} | {row['object_part']} | "
                  f"severity={row['severity']}")
        else:
            row = build_fallback_row(claim, "Model returned no response")
            print(f"  → ✗ FALLBACK (model failure)")

        results.append(row)

    # ─── Write output ───
    write_output_csv(results, output_path, OUTPUT_COLUMNS)
    elapsed = time.time() - start_time

    # ─── Print summary ───
    metrics = client.get_metrics()
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Output: {output_path}")
    print(f"  Rows written: {len(results)}")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  API calls: {metrics['total_api_calls']}")
    print(f"  Retries: {metrics['total_retries']}")
    print(f"  Failures: {metrics['total_failures']}")
    print(f"  Images processed: {metrics['total_images_processed']}")
    print(f"  Input tokens: ~{metrics['total_input_tokens']}")
    print(f"  Output tokens: ~{metrics['total_output_tokens']}")
    print("=" * 60)

    return results, metrics


def main():
    """Entry point for processing claims.csv."""
    dataset_dir = DATASET_DIR
    claims_csv = dataset_dir / "claims.csv"
    output_path = OUTPUT_CSV

    if not claims_csv.exists():
        print(f"Error: {claims_csv} not found.")
        sys.exit(1)

    process_claims(claims_csv, output_path, dataset_dir)


if __name__ == "__main__":
    main()
