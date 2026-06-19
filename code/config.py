"""Configuration and constants for the claim verification system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env ───
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

# ─── Paths ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
CODE_DIR = PROJECT_ROOT / "code"
OUTPUT_CSV = CODE_DIR / "output.csv"

# ─── Groq API Config ───
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # multimodal, fast
FALLBACK_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─── Rate Limiting ───
MAX_RPM = 14                              # conservative to avoid 429s
SLEEP_BETWEEN_CALLS = 60.0 / MAX_RPM     # ~4.3s between calls
MAX_RETRIES = 3
RETRY_BASE_DELAY = 8                     # exponential backoff base

# ─── Allowed Values ───
CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
]

CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown"
]

LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown"
]

PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown"
]

OBJECT_PARTS = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
]

SEVERITIES = ["none", "low", "medium", "high", "unknown"]

# ─── Output Schema ───
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity"
]
