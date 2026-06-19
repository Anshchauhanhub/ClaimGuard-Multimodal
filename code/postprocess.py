"""Post-processing and validation of model outputs against allowed values."""
from typing import Dict, List
from config import (
    CLAIM_STATUSES, ISSUE_TYPES, OBJECT_PARTS,
    RISK_FLAGS, SEVERITIES,
)


def validate_and_normalize(prediction: Dict, claim_row: Dict) -> Dict:
    """Validate model output, clamp to allowed values, build final row."""
    claim_object = claim_row.get("claim_object", "")
    allowed_parts = OBJECT_PARTS.get(claim_object, ["unknown"])

    # ─── Normalize claim_status ───
    raw_status = str(prediction.get("claim_status", "")).lower().strip()
    claim_status = raw_status if raw_status in CLAIM_STATUSES else "not_enough_information"

    # ─── Normalize issue_type ───
    raw_issue = str(prediction.get("issue_type", "")).lower().strip()
    issue_type = raw_issue if raw_issue in ISSUE_TYPES else "unknown"

    # ─── Normalize object_part ───
    raw_part = str(prediction.get("object_part", "")).lower().strip()
    object_part = raw_part if raw_part in allowed_parts else "unknown"

    # ─── Normalize severity ───
    raw_sev = str(prediction.get("severity", "")).lower().strip()
    severity = raw_sev if raw_sev in SEVERITIES else "unknown"

    # ─── Normalize risk_flags ───
    raw_flags = str(prediction.get("risk_flags", "none")).strip()
    if raw_flags.lower() == "none" or not raw_flags:
        risk_flags = "none"
    else:
        flags = [f.strip().lower() for f in raw_flags.split(";")]
        valid_flags = [f for f in flags if f in RISK_FLAGS]
        risk_flags = ";".join(valid_flags) if valid_flags else "none"

    # ─── Normalize booleans ───
    evidence_met = _to_bool_str(prediction.get("evidence_standard_met", False))
    valid_image = _to_bool_str(prediction.get("valid_image", True))

    # ─── Normalize supporting_image_ids ───
    raw_ids = str(prediction.get("supporting_image_ids", "none")).strip()
    if raw_ids.lower() == "none" or not raw_ids:
        supporting_ids = "none"
    else:
        supporting_ids = ";".join(
            [s.strip() for s in raw_ids.split(";") if s.strip()]
        )

    # ─── Build final row ───
    return {
        "user_id": claim_row["user_id"],
        "image_paths": claim_row["image_paths"],
        "user_claim": claim_row["user_claim"],
        "claim_object": claim_row["claim_object"],
        "evidence_standard_met": evidence_met,
        "evidence_standard_met_reason": str(
            prediction.get("evidence_standard_met_reason", "")
        ).strip(),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": str(
            prediction.get("claim_status_justification", "")
        ).strip(),
        "supporting_image_ids": supporting_ids,
        "valid_image": valid_image,
        "severity": severity,
    }


def build_fallback_row(claim_row: Dict, error_msg: str = "") -> Dict:
    """Return a safe fallback row when the model fails entirely."""
    return {
        "user_id": claim_row["user_id"],
        "image_paths": claim_row["image_paths"],
        "user_claim": claim_row["user_claim"],
        "claim_object": claim_row["claim_object"],
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": f"Processing error: {error_msg}" if error_msg else "Processing error",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Could not process claim due to an error.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def _to_bool_str(val) -> str:
    """Convert various bool representations to 'true'/'false' string."""
    if isinstance(val, bool):
        return "true" if val else "false"
    s = str(val).lower().strip()
    return "true" if s in ("true", "1", "yes") else "false"
