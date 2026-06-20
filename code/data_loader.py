"""Data loaders for CSV files and images."""
import csv
import base64
from pathlib import Path
from typing import List, Dict, Optional


def load_csv(filepath: str | Path) -> List[Dict]:
    """Load a CSV file and return list of dicts."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_user_history(filepath: str | Path) -> Dict[str, Dict]:
    """Load user_history.csv into a dict keyed by user_id."""
    rows = load_csv(filepath)
    return {row["user_id"]: row for row in rows}


def load_evidence_requirements(filepath: str | Path) -> List[Dict]:
    """Load evidence_requirements.csv."""
    return load_csv(filepath)


def get_relevant_requirements(
    requirements: List[Dict], claim_object: str
) -> List[Dict]:
    """Filter evidence requirements relevant to a claim_object."""
    return [
        r for r in requirements
        if r["claim_object"] in (claim_object, "all")
    ]


def parse_image_paths(image_paths_str: str) -> List[str]:
    """Split semicolon-separated image paths."""
    return [p.strip() for p in image_paths_str.split(";") if p.strip()]


def get_image_ids(image_paths_str: str) -> List[str]:
    """Extract image IDs (filename without extension) from paths."""
    paths = parse_image_paths(image_paths_str)
    return [Path(p).stem for p in paths]


def load_image_as_base64(image_path: Path, max_dim: int = 512) -> Optional[str]:
    """Load an image, convert to JPEG if needed, resize if large, return base64."""
    if not image_path.exists():
        return None
    try:
        import pillow_avif  # enables AVIF support
    except ImportError:
        pass

    try:
        from PIL import Image
        import io

        img = Image.open(image_path)

        # Convert RGBA/P modes to RGB for JPEG
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if any dimension exceeds max_dim
        w, h = img.size
        if w > max_dim or h > max_dim:
            ratio = min(max_dim / w, max_dim / h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Encode as JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"  ⚠ Image processing error for {image_path}: {e}")
        # Fallback: try raw bytes
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None


def get_mime_type(image_path: Path) -> str:
    """Always return image/jpeg since we convert everything to JPEG."""
    return "image/jpeg"


def write_output_csv(
    rows: List[Dict], output_path: str | Path, columns: List[str]
):
    """Write output CSV with exact column order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for row in rows:
            # Ensure only the expected columns are written
            filtered = {k: row.get(k, "") for k in columns}
            writer.writerow(filtered)
