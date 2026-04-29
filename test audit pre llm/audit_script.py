#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CAD COMPLIANCE AUDIT SYSTEM  —  COMPLETE PRODUCTION SCRIPT                ║
║  Hybrid Neuro-Symbolic Pipeline Validator                                   ║
║  Version : 3.0.0                                                            ║
║  Python  : 3.9+                                                             ║
║                                                                             ║
║  USAGE:                                                                     ║
║    python audit_script.py drawing.pdf manifest.json --visual                ║
║    python audit_script.py drawing.pdf manifest.json --headless              ║
║    python audit_script.py drawing.pdf manifest.json --pixel                 ║
║    python audit_script.py drawing.pdf manifest.json --ocr                   ║
║    python audit_script.py drawing.pdf manifest.json --math                  ║
║    python audit_script.py drawing.pdf manifest.json --fingerprint           ║
║    python audit_script.py drawing.pdf manifest.json --regression            ║
║    python audit_script.py --selftest                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — PYTHON VERSION GUARD  (must be first executable line)
# ══════════════════════════════════════════════════════════════════════════════
import sys

if sys.version_info < (3, 9):
    print(
        f"[FATAL] Python 3.9+ required. "
        f"Current: {sys.version_info.major}.{sys.version_info.minor}"
    )
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — STANDARD LIBRARY IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import io
import json
import os
import re
import math
import time
import hashlib
import difflib
import logging
import argparse
import traceback
import itertools
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — THIRD-PARTY IMPORTS WITH GRACEFUL FAILURE
# ══════════════════════════════════════════════════════════════════════════════

# OpenCV + NumPy (required)
try:
    import cv2
    import numpy as np
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    print(
        "[FATAL] OpenCV / NumPy not found.\n"
        "        Install: pip install opencv-python numpy"
    )
    sys.exit(1)

# pdf2image (required for all non-selftest modes)
try:
    from pdf2image import convert_from_path
    _PDF2IMAGE_OK = True
except ImportError:
    _PDF2IMAGE_OK = False
    convert_from_path = None  # type: ignore

# pytesseract (optional — only needed for --tesseract flag)
try:
    import pytesseract as _pytesseract
    _TESSERACT_OK = True
except ImportError:
    _pytesseract = None
    _TESSERACT_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

__version__  = "3.0.0"
__pipeline__ = "Hybrid Neuro-Symbolic CAD Compliance System"

# ── Rendering ─────────────────────────────────────────────────────────────────
DEFAULT_DPI        = 150     # MUST match Node 01 DPI
DEFAULT_PAGE_IDX   = 0       # 0-based; page 1 = index 0

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_OVERLAY     = "audit_overlay_result.png"
OUTPUT_REPORT_TXT  = "audit_report.txt"
OUTPUT_REG_JSON    = "regression_report.json"
OUTPUT_CROPS_DIR   = "pixel_crops"

# ── CI/CD thresholds (§8 headless rules) ─────────────────────────────────────
MIN_LINK_RATIO         = 0.10   # ≥10 % of geometries must be linked
MAX_EMPTY_TAG_RATIO    = 0.80   # Fail if >80 % table_values empty/null
MIN_AXIOM_COUNT        = 1
MIN_VALIDATION_COUNT   = 1

# ── Visual drawing colours (BGR tuples) ───────────────────────────────────────
COLOR_GEOMETRY   = (0,   0,   255)   # Red    — axiom boxes
COLOR_TEXT_BOX   = (255, 0,   0)     # Blue   — text boxes
COLOR_PASS_LINE  = (0,   255, 0)     # Green  — PASS links
COLOR_FAIL_LINE  = (0,   255, 255)   # Yellow — FAIL/CONFLICT links
COLOR_LABEL      = (255, 255, 255)   # White  — label text
COLOR_LABEL_BG   = (30,  30,  30)    # Dark   — label background
COLOR_WARNING    = (0,   165, 255)   # Orange — OOB warning

FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.45
FONT_THICKNESS  = 1
LINE_THICKNESS  = 2
BOX_THICKNESS   = 2

# ── §9 Pixel fidelity ─────────────────────────────────────────────────────────
DEFAULT_INK_THRESHOLD  = 0.05    # Min fraction of non-white pixels
WHITE_PIXEL_THRESHOLD  = 245     # Grayscale value considered "white"

# ── §10 OCR verifier ─────────────────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD  = 0.75

# ── §11 Math verifier ────────────────────────────────────────────────────────
DEFAULT_PASS_MAX_DIST  = 500.0   # px — PASS link distance ceiling
DEFAULT_FAIL_MIN_DIST  = 50.0    # px — FAIL link distance floor
DISTANCE_TOLERANCE     = 15.0    # px — delta tolerance vs JSON value

# ── §12 Hallucination fingerprinter ──────────────────────────────────────────
DUPLICATE_IOU_THR   = 0.90
GHOST_BOX_MIN_AREA  = 100.0      # px²
MAX_BOX_PAGE_RATIO  = 0.95
CLUSTER_SPREAD_RATIO = 0.05

# ── CAD annotation regex patterns ────────────────────────────────────────────
CAD_PATTERNS: dict[str, re.Pattern] = {
    "dimension"      : re.compile(r"^\d{2,5}[xX×]\d{2,5}$"),
    "column_id"      : re.compile(r"^[A-Z]\d{1,3}$"),
    "level_tag"      : re.compile(r"^(EL|FL|GL|RL)[.\-]?\d+(\.\d+)?$", re.I),
    "rebar"          : re.compile(r"^\d+T\d+$", re.I),
    "footing_tag"    : re.compile(r"^F\d+[A-Z]?$"),
    "slab_thickness" : re.compile(r"^\d{2,3}(mm|MM|thk|THK)?$"),
    "grid_line"      : re.compile(r"^[A-Z\d]['\-/]?[A-Z\d]?$"),
    "bearing"        : re.compile(r"^\d{1,3}°?\d{0,2}'?[NSEW]?$"),
    "mixed_numeric"  : re.compile(r"^\d+[\.\-/]\d+$"),
    "load_value"     : re.compile(r"^\d+(\.\d+)?\s?(kN|kPa|MPa|N|kNm)$", re.I),
    "percentage"     : re.compile(r"^\d{1,3}(\.\d+)?%$"),
    "angle"          : re.compile(r"^\d{1,3}(\.\d+)?(°|deg)$", re.I),
}

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("CAD-Audit")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  [%(levelname)-7s]  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


log = _build_logger()

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 6 — DATA STRUCTURES (dataclasses)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PageAsset:
    """One rendered PDF page with its pixel data and metadata."""
    page_index : int
    pil_image  : Any            # PIL.Image.Image
    bgr_image  : np.ndarray     # shape (H, W, 3), dtype uint8
    width_px   : int
    height_px  : int
    dpi        : int
    page_hash  : str            # First 16 chars of SHA-256 of raw RGB bytes

    def __repr__(self) -> str:
        return (
            f"PageAsset(page={self.page_index + 1}, "
            f"{self.width_px}×{self.height_px}px @ {self.dpi}dpi, "
            f"hash={self.page_hash})"
        )


@dataclass
class AssetBundle:
    """
    Central asset store.  Loaded ONCE in extended_main() and passed
    as the first argument to every section function.
    """
    pdf_path      : str
    json_path     : str
    dpi           : int
    pages         : list[PageAsset]  = field(default_factory=list)
    manifest      : dict             = field(default_factory=dict)
    axioms        : list[dict]       = field(default_factory=list)
    validations   : list[dict]       = field(default_factory=list)
    load_time_s   : float            = 0.0
    json_size_kb  : float            = 0.0
    schema_version: str              = "unknown"

    def page(self, idx: int = 0) -> PageAsset:
        """Return PageAsset by 0-based index; clamps to last page."""
        if not self.pages:
            raise RuntimeError("AssetBundle has no loaded pages.")
        return self.pages[min(idx, len(self.pages) - 1)]


@dataclass
class CropResult:
    """Result of one pixel-level crop test (§9)."""
    record_id       : str
    record_type     : str
    bbox            : Any
    crop_shape      : tuple
    mean_intensity  : float
    ink_pixel_ratio : float
    has_ink         : bool
    status          : str    # "PASS" | "FAIL" | "SKIP"
    reason          : str


@dataclass
class MathCheckResult:
    """Result of one DHMoT distance re-computation (§11)."""
    val_id            : str
    json_status       : str
    json_distance     : Optional[float]
    computed_distance : float
    distance_delta    : float
    bbox_overlap_iou  : float
    spatial_verdict   : str   # "CONSISTENT" | "INCONSISTENT" | "SKIP"
    note              : str


@dataclass
class HallucinationFlag:
    """One detected hallucination pattern (§12)."""
    flag_type : str
    severity  : str   # "HIGH" | "MEDIUM" | "LOW"
    record_id : str
    detail    : str


@dataclass
class SuiteResult:
    """Result of one section in the regression suite (§13)."""
    section_id : str
    name       : str
    overall    : str    # "PASS" | "FAIL" | "WARN" | "SKIP"
    duration_s : float
    detail     : dict

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 7 — DATA LOADING & ASSET MANAGEMENT (§4)
# ══════════════════════════════════════════════════════════════════════════════

def load_json_safe(json_path: str) -> dict:
    """
    Parse axiom_manifest.json.
    Exits with code 1 on file-not-found or JSON parse error.
    """
    if not os.path.exists(json_path):
        log.error("JSON not found: %s", json_path)
        sys.exit(1)

    size_kb = os.path.getsize(json_path) / 1024
    log.info("Loading JSON manifest (%.1f KB): %s", size_kb, json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        log.error(
            "JSON parse error at line %d col %d: %s",
            exc.lineno, exc.colno, exc.msg,
        )
        sys.exit(1)
    except OSError as exc:
        log.error("Cannot read JSON: %s", exc)
        sys.exit(1)

    log.info("JSON loaded — %d top-level keys.", len(data))
    return data


def load_pdf_pages(
    pdf_path    : str,
    dpi         : int,
    page_indices: Optional[list[int]] = None,
) -> list[PageAsset]:
    """
    Render PDF pages to PageAsset objects.
    page_indices = None → load ALL pages.
    page_indices = [0]  → load only page 1 (default).
    Exits with code 1 on any render failure.
    """
    if not _PDF2IMAGE_OK:
        log.error(
            "pdf2image not installed — cannot render PDF.\n"
            "        Install: pip install pdf2image\n"
            "        Also needs poppler: "
            "apt-get install poppler-utils  OR  brew install poppler"
        )
        sys.exit(1)

    if not os.path.exists(pdf_path):
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    log.info("Rendering PDF at %d DPI: %s", dpi, pdf_path)
    t0 = time.time()

    try:
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as exc:
        log.error("PDF rendering failed: %s", exc)
        log.error(
            "Ensure poppler is installed: "
            "apt-get install poppler-utils  OR  brew install poppler"
        )
        sys.exit(1)

    elapsed = time.time() - t0
    log.info("PDF rendered: %d page(s) in %.2fs", len(pil_pages), elapsed)

    if page_indices is None:
        page_indices = list(range(len(pil_pages)))

    assets: list[PageAsset] = []
    for idx in page_indices:
        if idx >= len(pil_pages):
            log.warning(
                "Page index %d out of range (total %d). Skipping.",
                idx, len(pil_pages),
            )
            continue

        pil  = pil_pages[idx]
        w, h = pil.size
        rgb  = np.array(pil)
        bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        sha  = hashlib.sha256(rgb.tobytes()).hexdigest()[:16]

        assets.append(
            PageAsset(
                page_index=idx,
                pil_image=pil,
                bgr_image=bgr,
                width_px=w,
                height_px=h,
                dpi=dpi,
                page_hash=sha,
            )
        )
        log.info("  Page %d: %d×%d px  hash=%s", idx + 1, w, h, sha)

    return assets


def build_asset_bundle(
    pdf_path    : str,
    json_path   : str,
    dpi         : int               = DEFAULT_DPI,
    page_indices: Optional[list[int]] = None,
) -> AssetBundle:
    """
    Master factory — the ONLY place that touches disk for audit data.
    Returns a fully populated AssetBundle ready for all section functions.
    """
    t0     = time.time()
    bundle = AssetBundle(pdf_path=pdf_path, json_path=json_path, dpi=dpi)

    bundle.manifest      = load_json_safe(json_path)
    bundle.json_size_kb  = os.path.getsize(json_path) / 1024
    bundle.schema_version = str(
        bundle.manifest.get("schema_version")
        or bundle.manifest.get("version")
        or "unknown"
    )

    bundle.axioms      = extract_axioms(bundle.manifest)
    bundle.validations = extract_validations(bundle.manifest)

    if page_indices is None:
        page_indices = [DEFAULT_PAGE_IDX]
    bundle.pages      = load_pdf_pages(pdf_path, dpi, page_indices)
    bundle.load_time_s = round(time.time() - t0, 3)

    log.info(
        "AssetBundle ready — axioms=%d  validations=%d  pages=%d  "
        "load=%.3fs  schema=%s",
        len(bundle.axioms),
        len(bundle.validations),
        len(bundle.pages),
        bundle.load_time_s,
        bundle.schema_version,
    )
    return bundle

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 8 — DATA EXTRACTION HELPERS (§5)
# ══════════════════════════════════════════════════════════════════════════════

def extract_axioms(data: dict) -> list[dict]:
    """
    Normalise axiom records from the manifest.

    Tries root keys in priority order:
        "axioms" → "nodes" → "geometries"

    Each returned dict is guaranteed to have:
        id, label, bounding_box, page, raw
    """
    raw_list = (
        data.get("axioms")
        or data.get("nodes")
        or data.get("geometries")
        or []
    )

    results = []
    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            log.debug("Axiom index %d is not a dict — skipping.", idx)
            continue

        spatial = (
            item.get("spatial_context")
            or item.get("spatial")
            or item
        )

        bbox = (
            spatial.get("bounding_box")
            or spatial.get("bbox")
            or item.get("bounding_box")
            or item.get("bbox")
        )

        results.append(
            {
                "id"           : str(
                    item.get("id")
                    or item.get("axiom_id")
                    or f"axiom_{idx}"
                ),
                "label"        : str(
                    item.get("label") or item.get("type") or "GEOM"
                ),
                "bounding_box" : bbox,
                "page"         : int(
                    item.get("page") or item.get("page_index") or 0
                ),
                "raw"          : item,
            }
        )

    log.info("Extracted %d axioms.", len(results))
    return results


def extract_validations(data: dict) -> list[dict]:
    """
    Normalise validation / hyperedge records.

    Tries root keys in priority order:
        "validations" → "hyperedges" → "edges"

    Each returned dict is guaranteed to have:
        id, table_value, status, text_bounding_box,
        geometry_bounding_box, distance, page, raw
    """
    # Status normalisation map
    STATUS_MAP = {
        "PASSED" : "PASS",
        "OK"     : "PASS",
        "FAILED" : "FAIL",
        "ERROR"  : "FAIL",
    }

    raw_list = (
        data.get("validations")
        or data.get("hyperedges")
        or data.get("edges")
        or []
    )

    results = []
    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            log.debug("Validation index %d is not a dict — skipping.", idx)
            continue

        details = item.get("details") or item.get("metadata") or {}

        val_id = str(
            item.get("id")
            or item.get("validation_id")
            or item.get("hyperedge_id")
            or f"val_{idx}"
        )

        raw_status = str(item.get("status") or "UNKNOWN").upper().strip()
        status     = STATUS_MAP.get(raw_status, raw_status)
        # Ensure only known values pass through
        if status not in ("PASS", "FAIL", "CONFLICT", "UNKNOWN"):
            status = "UNKNOWN"

        dist_raw = (
            details.get("euclidean_distance")
            or details.get("distance")
            or item.get("distance")
        )
        try:
            dist = float(dist_raw) if dist_raw is not None else None
        except (TypeError, ValueError):
            dist = None

        results.append(
            {
                "id"                   : val_id,
                "table_value"          : (
                    item.get("table_value")
                    or item.get("label")
                    or item.get("text")
                ),
                "status"               : status,
                "text_bounding_box"    : (
                    details.get("text_bounding_box")
                    or details.get("text_bbox")
                    or item.get("text_bounding_box")
                ),
                "geometry_bounding_box": (
                    details.get("geometry_bounding_box")
                    or details.get("geom_bbox")
                    or item.get("geometry_bounding_box")
                ),
                "distance"             : dist,
                "page"                 : int(
                    item.get("page") or item.get("page_index") or 0
                ),
                "raw"                  : item,
            }
        )

    log.info("Extracted %d validations.", len(results))
    return results

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 9 — GEOMETRY & MATH UTILITIES (§6)
# ══════════════════════════════════════════════════════════════════════════════

def is_bbox_valid(
    bbox    : Any,
    canvas_w: int,
    canvas_h: int,
    margin  : int = 50,
) -> bool:
    """
    Return True iff bbox is a well-formed [x1, y1, x2, y2] with:
      • exactly 4 numeric elements
      • x2 > x1  and  y2 > y1  (non-degenerate)
      • coordinates within canvas ± margin pixels
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if (
        x1 < -margin
        or y1 < -margin
        or x2 > canvas_w + margin
        or y2 > canvas_h + margin
    ):
        return False
    return True


def bbox_center(bbox: list) -> tuple[float, float]:
    """Return (cx, cy) centre of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: Any) -> float:
    """Return pixel area of [x1, y1, x2, y2]. Returns 0.0 for invalid input."""
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    except (TypeError, ValueError, TypeError):
        return 0.0


def compute_euclidean(bbox_a: Any, bbox_b: Any) -> float:
    """
    Euclidean distance between centres of two [x1,y1,x2,y2] boxes.
    Returns -1.0 if either bbox is structurally invalid.
    """
    try:
        cx_a, cy_a = bbox_center(bbox_a)
        cx_b, cy_b = bbox_center(bbox_b)
        return math.hypot(cx_b - cx_a, cy_b - cy_a)
    except Exception:
        return -1.0


def compute_iou(bbox_a: Any, bbox_b: Any) -> float:
    """
    Intersection over Union for two [x1,y1,x2,y2] boxes.
    Returns 0.0 on invalid input or no overlap; 1.0 for identical boxes.
    """
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
    except (TypeError, ValueError):
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / max(union, 1e-9)


def fuzzy_match_score(a: Any, b: Any) -> float:
    """
    Ratcliff/Obershelp similarity in [0.0, 1.0].
    Handles None inputs gracefully.
    """
    sa = str(a).strip() if a is not None else ""
    sb = str(b).strip() if b is not None else ""
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return difflib.SequenceMatcher(None, sa, sb).ratio()


def classify_cad_text(text: Any) -> str:
    """
    Match text against CAD annotation patterns.
    Returns pattern name or "empty" / "unclassified".
    """
    if text is None:
        return "empty"
    t = str(text).strip()
    if not t:
        return "empty"
    for name, pattern in CAD_PATTERNS.items():
        if pattern.match(t):
            return name
    return "unclassified"


def crop_bbox_from_image(
    bgr_image: np.ndarray,
    bbox     : Any,
    padding  : int = 2,
) -> Optional[np.ndarray]:
    """
    Safely crop [x1,y1,x2,y2] from a BGR image with clamped padding.
    Returns None if the crop is degenerate.
    """
    h, w = bgr_image.shape[:2]
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
    except (TypeError, ValueError):
        return None

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = bgr_image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def measure_ink(
    crop           : np.ndarray,
    white_threshold: int = WHITE_PIXEL_THRESHOLD,
) -> tuple[float, float]:
    """
    Measure ink presence in a BGR crop.
    Returns (mean_intensity, ink_pixel_ratio).
      mean_intensity  : mean grayscale value (0=black, 255=white)
      ink_pixel_ratio : fraction of pixels below white_threshold
    """
    gray            = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_i          = float(np.mean(gray))
    ink_pixel_ratio = float(np.sum(gray < white_threshold)) / max(gray.size, 1)
    return mean_i, ink_pixel_ratio


def print_ascii_histogram(
    values : list[float],
    title  : str = "Distribution",
    bins   : int = 8,
    bar_max: int = 40,
    indent : str = "    ",
) -> None:
    """
    Print a fixed-width ASCII histogram.
    GAP-16 fix: step=1.0 when all values are equal (avoids divide-by-zero).
    """
    if not values:
        print(f"{indent}(no data)")
        return

    min_v = min(values)
    max_v = max(values)
    step  = (max_v - min_v) / bins if max_v > min_v else 1.0

    counts: dict[int, int] = defaultdict(int)
    for v in values:
        bucket = int((v - min_v) / step)
        counts[min(bucket, bins - 1)] += 1

    print(f"\n{indent}{title}:")
    for b in range(bins):
        lo  = min_v + b * step
        hi  = lo + step
        cnt = counts[b]
        bar = "█" * min(cnt, bar_max)
        print(f"{indent}  [{lo:8.1f} – {hi:8.1f}]  {cnt:>5}  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 10 — VISUAL AUDITOR  §7 / MODE 1
# ══════════════════════════════════════════════════════════════════════════════

def _draw_label_with_bg(
    canvas   : np.ndarray,
    text     : str,
    x        : int,
    y        : int,
    color    : tuple = COLOR_LABEL,
    bg       : tuple = COLOR_LABEL_BG,
    scale    : float = FONT_SCALE,
    thickness: int   = FONT_THICKNESS,
    padding  : int   = 3,
) -> None:
    """Draw text with a solid background rectangle for legibility."""
    h, w = canvas.shape[:2]
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    lx = max(padding, min(x, w - tw - padding))
    ly = max(th + padding, min(y, h - padding))
    cv2.rectangle(
        canvas,
        (lx - padding, ly - th - padding),
        (lx + tw + padding, ly + bl + padding),
        bg,
        cv2.FILLED,
    )
    cv2.putText(canvas, text, (lx, ly), FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_semi_transparent_box(
    canvas   : np.ndarray,
    bbox     : list,
    color    : tuple,
    alpha    : float = 0.20,
    thickness: int   = BOX_THICKNESS,
) -> None:
    """Draw a semi-transparent filled box using in-place ROI blending."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = canvas.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return
    roi = canvas[y1c:y2c, x1c:x2c].copy()
    cv2.rectangle(roi, (0, 0), (x2c - x1c, y2c - y1c), color, cv2.FILLED)
    canvas[y1c:y2c, x1c:x2c] = cv2.addWeighted(
        roi, alpha, canvas[y1c:y2c, x1c:x2c], 1 - alpha, 0
    )
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)


def _draw_legend_panel(
    canvas    : np.ndarray,
    stats     : dict,
    n_axioms  : int,
    n_vals    : int,
    page_asset: PageAsset,
) -> None:
    """Render a statistics legend in the top-right corner."""
    panel_w, panel_h = 340, 245
    pw = page_asset.width_px
    px = pw - panel_w - 12
    py = 12

    if px < 0 or py < 0:
        return

    sub   = canvas[py : py + panel_h, px : px + panel_w]
    black = np.zeros_like(sub)
    cv2.addWeighted(black, 0.70, sub, 0.30, 0, sub)
    canvas[py : py + panel_h, px : px + panel_w] = sub
    cv2.rectangle(canvas, (px, py), (px + panel_w, py + panel_h), (180, 180, 180), 1)

    n_pass = stats.get("links_PASS", 0)
    n_fail = stats.get("links_FAIL", 0) + stats.get("links_CONFLICT", 0)

    lines: list[tuple[str, tuple]] = [
        (f"AUDIT LEGEND  v{__version__}",                   (220, 220, 220)),
        ("─" * 38,                                          (70,  70,  70)),
        (f"  Total Axioms (Geometry) : {n_axioms}",         (200, 200, 255)),
        (f"  Total Validations       : {n_vals}",           (255, 200, 200)),
        (f"  Geometry Boxes Drawn    : {stats.get('axioms_drawn', 0)}",
                                                            COLOR_GEOMETRY),
        (f"  Text Boxes Drawn        : {stats.get('text_boxes_drawn', 0)}",
                                                            COLOR_TEXT_BOX),
        (f"  PASS Links  (green)     : {n_pass}",          COLOR_PASS_LINE),
        (f"  FAIL Links  (yellow)    : {n_fail}",          COLOR_FAIL_LINE),
        (f"  Incomplete Links        : {stats.get('links_incomplete', 0)}",
                                                            (150, 150, 150)),
        (f"  OOB Boxes   (flagged)   : {stats.get('oob_boxes', 0)}",
                                                            COLOR_WARNING),
        ("─" * 38,                                          (70,  70,  70)),
        (f"  DPI: {page_asset.dpi}   Page: {page_asset.page_index + 1}",
                                                            (160, 160, 160)),
        (f"  Hash: {page_asset.page_hash}",                 (130, 130, 130)),
    ]

    lx, ly = px + 8, py + 22
    for text, color in lines:
        cv2.putText(canvas, text, (lx, ly), FONT, 0.38, color, 1, cv2.LINE_AA)
        ly += 17


def run_visual_audit(
    bundle     : AssetBundle,
    output_path: str = OUTPUT_OVERLAY,
) -> None:
    """
    §7 / MODE 1 — Visual Auditor
    Draws coloured overlays on the PDF page image and saves as PNG.

    Red   boxes  = axiom geometry bounding boxes
    Blue  boxes  = text bounding boxes
    Green lines  = PASS spatial links
    Yellow lines = FAIL / CONFLICT spatial links
    Legend panel = top-right statistics summary
    """
    log.info("═" * 60)
    log.info("  §7  MODE 1 — VISUAL AUDITOR")
    log.info("═" * 60)

    page   = bundle.page(DEFAULT_PAGE_IDX)
    canvas = page.bgr_image.copy()
    h, w   = canvas.shape[:2]
    stats: dict[str, int] = defaultdict(int)

    # ── Pass 1: Geometry boxes (Red) ─────────────────────────────
    log.info("Drawing %d axiom geometry boxes …", len(bundle.axioms))
    for axiom in bundle.axioms:
        bbox = axiom.get("bounding_box")
        if not is_bbox_valid(bbox, w, h):
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                stats["oob_boxes"] += 1
            stats["axioms_skipped"] += 1
            continue

        _draw_semi_transparent_box(canvas, bbox, COLOR_GEOMETRY, alpha=0.15)
        _draw_label_with_bg(
            canvas,
            axiom["label"],
            int(bbox[0]),
            max(0, int(bbox[1]) - 6),
            color=(220, 220, 255),
            bg=(140, 0, 0),
        )
        stats["axioms_drawn"] += 1

    # ── Pass 2: Text boxes + connection lines ─────────────────────
    log.info("Drawing %d validation links …", len(bundle.validations))
    for val in bundle.validations:
        t_bbox = val.get("text_bounding_box")
        g_bbox = val.get("geometry_bounding_box")
        label  = str(val.get("table_value") or "?")[:20]
        status = val.get("status", "UNKNOWN")

        t_valid = is_bbox_valid(t_bbox, w, h)
        g_valid = is_bbox_valid(g_bbox, w, h)

        if t_valid:
            _draw_semi_transparent_box(canvas, t_bbox, COLOR_TEXT_BOX, alpha=0.18)
            _draw_label_with_bg(
                canvas,
                label,
                int(t_bbox[0]),
                max(0, int(t_bbox[1]) - 6),
            )
            stats["text_boxes_drawn"] += 1
        else:
            stats["text_boxes_skipped"] += 1

        if t_valid and g_valid:
            lc = COLOR_PASS_LINE if status == "PASS" else COLOR_FAIL_LINE
            tx, ty = int(bbox_center(t_bbox)[0]), int(bbox_center(t_bbox)[1])
            gx, gy = int(bbox_center(g_bbox)[0]), int(bbox_center(g_bbox)[1])
            cv2.line(canvas, (tx, ty), (gx, gy), lc, LINE_THICKNESS, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 4, lc, -1)
            cv2.circle(canvas, (gx, gy), 4, lc, -1)
            stats[f"links_{status}"] += 1
            stats["total_links"] += 1
        else:
            stats["links_incomplete"] += 1

    # ── Pass 3: Legend ────────────────────────────────────────────
    _draw_legend_panel(
        canvas, stats,
        len(bundle.axioms), len(bundle.validations),
        page,
    )

    # ── Save ──────────────────────────────────────────────────────
    ok = cv2.imwrite(output_path, canvas)
    if ok:
        size_kb = os.path.getsize(output_path) / 1024
        log.info("✔  Overlay saved: %s  (%.1f KB)", output_path, size_kb)
    else:
        log.error("✘  Failed to write overlay image: %s", output_path)
        sys.exit(1)

    log.info(
        "Visual summary — geom_drawn:%d  text_drawn:%d  links:%d  skipped:%d",
        stats["axioms_drawn"],
        stats["text_boxes_drawn"],
        stats["total_links"],
        stats["axioms_skipped"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 11 — HEADLESS CI/CD AUDITOR  §8 / MODE 2
# ══════════════════════════════════════════════════════════════════════════════

class AuditResult:
    """Collects CI/CD rule outcomes and formats the terminal/file report."""

    def __init__(self) -> None:
        self.passed  : list[str] = []
        self.failed  : list[str] = []
        self.warnings: list[str] = []
        self.stats   : dict      = {}

    def stat(self, key: str, value: Any) -> None:
        self.stats[key] = value

    def rule_pass(self, rule_id: str, msg: str) -> None:
        self.passed.append(f"  ✔  [{rule_id}] {msg}")

    def rule_fail(self, rule_id: str, msg: str) -> None:
        self.failed.append(f"  ✘  [{rule_id}] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(f"  ⚠   {msg}")

    @property
    def overall_pass(self) -> bool:
        return len(self.failed) == 0

    def print_report(self) -> None:
        div  = "═" * 65
        thin = "─" * 65
        print(f"\n{div}")
        print(f"  CAD COMPLIANCE AUDIT REPORT  —  {__pipeline__}")
        print(div)

        print("\n  📊  STATISTICS")
        print(thin)
        for k, v in self.stats.items():
            label = k.replace("_", " ").title()
            print(f"  {label:<40}: {v}")

        print(f"\n  📋  RULE RESULTS")
        print(thin)
        if self.passed:
            print("  PASSED:")
            for r in self.passed:
                print(r)
        if self.failed:
            print("\n  FAILED:")
            for r in self.failed:
                print(r)
        if self.warnings:
            print("\n  WARNINGS:")
            for w in self.warnings:
                print(w)

        print(f"\n{div}")
        verdict = (
            "✔  ALL RULES PASSED"
            if self.overall_pass
            else f"✘  {len(self.failed)} RULE(S) FAILED"
        )
        print(f"  VERDICT: {verdict}  →  sys.exit({0 if self.overall_pass else 1})")
        print(div)

    def save_txt(self, path: str) -> None:
        lines = (
            ["CAD COMPLIANCE AUDIT REPORT", "=" * 65, ""]
            + ["STATS:"]
            + [f"  {k}: {v}" for k, v in self.stats.items()]
            + ["", "PASSED:"] + (self.passed or ["  (none)"])
            + ["", "FAILED:"] + (self.failed or ["  (none)"])
            + ["", "WARNINGS:"] + (self.warnings or ["  (none)"])
            + ["", f"VERDICT: {'PASS' if self.overall_pass else 'FAIL'}"]
        )
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            log.info("Report saved: %s", path)
        except OSError as exc:
            log.warning("Could not save report: %s", exc)


def _check_coord_sanity(item_id: str, bbox: Any) -> Optional[str]:
    """
    Return a reason string if bbox coordinates are invalid,
    or None if they are fine.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None   # Already caught by R4
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return "non-numeric coordinates"
    if x1 < 0 or y1 < 0:
        return f"negative origin x1={x1:.1f} y1={y1:.1f}"
    if x2 <= x1:
        return f"x2({x2:.1f}) ≤ x1({x1:.1f})"
    if y2 <= y1:
        return f"y2({y2:.1f}) ≤ y1({y1:.1f})"
    return None


def run_headless_audit(bundle: AssetBundle) -> None:
    """
    §8 / MODE 2 — CI/CD Headless Auditor
    Applies rules R1–R7. Exits 0 (all pass) or 1 (any fail).

    R1 – OCR Blindness       : table_values must not be mostly empty
    R2 – Orphan Detection    : link ratio ≥ MIN_LINK_RATIO
    R3 – Schema Integrity    : required root keys exist
    R4 – Bbox Dimensionality : bboxes are 4-element lists
    R5 – Coordinate Sanity   : coords non-negative, x2>x1, y2>y1
    R6 – Status Distribution : not 100 % FAIL or UNKNOWN
    R7 – Canvas Bounds       : all bboxes within page dimensions
    """
    log.info("═" * 60)
    log.info("  §8  MODE 2 — HEADLESS CI/CD AUDITOR")
    log.info("═" * 60)

    result      = AuditResult()
    page        = bundle.page(DEFAULT_PAGE_IDX)
    page_w      = page.width_px
    page_h      = page.height_px
    axioms      = bundle.axioms
    validations = bundle.validations
    n_axioms    = len(axioms)
    n_vals      = len(validations)

    result.stat("pdf",               os.path.basename(bundle.pdf_path))
    result.stat("json",              os.path.basename(bundle.json_path))
    result.stat("json_size_kb",      f"{bundle.json_size_kb:.1f}")
    result.stat("schema_version",    bundle.schema_version)
    result.stat("page_size_px",      f"{page_w}×{page_h}")
    result.stat("render_dpi",        bundle.dpi)
    result.stat("page_hash",         page.page_hash)
    result.stat("total_axioms",      n_axioms)
    result.stat("total_validations", n_vals)

    # ── R3 – Schema Integrity ─────────────────────────────────────
    has_ax  = any(k in bundle.manifest for k in ("axioms", "nodes", "geometries"))
    has_val = any(k in bundle.manifest for k in ("validations", "hyperedges", "edges"))
    r3_ok   = True

    if not has_ax:
        result.rule_fail("R3", "Missing geometry key (axioms / nodes / geometries).")
        r3_ok = False
    if not has_val:
        result.rule_fail("R3", "Missing validation key (validations / hyperedges / edges).")
        r3_ok = False
    if n_axioms < MIN_AXIOM_COUNT:
        result.rule_fail("R3", f"Axiom array empty (found {n_axioms}).")
        r3_ok = False
    if n_vals < MIN_VALIDATION_COUNT:
        result.rule_fail("R3", f"Validation array empty (found {n_vals}).")
        r3_ok = False
    if r3_ok:
        result.rule_pass("R3", f"Schema keys present — axioms:{n_axioms}, vals:{n_vals}.")

    # ── R1 – OCR Blindness ────────────────────────────────────────
    null_c  = sum(1 for v in validations if v.get("table_value") is None)
    empty_c = sum(
        1 for v in validations
        if v.get("table_value") is not None
        and str(v["table_value"]).strip() == ""
    )
    total_empty = null_c + empty_c
    empty_ratio = total_empty / max(n_vals, 1)

    sample_vals = [
        str(v["table_value"])
        for v in validations
        if v.get("table_value") and str(v["table_value"]).strip()
    ][:5]

    result.stat("ocr_null_values",  null_c)
    result.stat("ocr_empty_values", empty_c)
    result.stat("ocr_empty_ratio",  f"{empty_ratio * 100:.1f}%")
    result.stat("ocr_sample",       str(sample_vals) if sample_vals else "N/A")

    if n_vals == 0:
        result.warn("No validations found — R1 skipped.")
    elif empty_ratio > MAX_EMPTY_TAG_RATIO:
        result.rule_fail(
            "R1",
            f"OCR Blindness: {total_empty}/{n_vals} table_values empty/null "
            f"({empty_ratio * 100:.1f}% > {MAX_EMPTY_TAG_RATIO * 100:.0f}%).",
        )
    else:
        result.rule_pass(
            "R1",
            f"OCR present: {n_vals - total_empty}/{n_vals} values filled "
            f"(empty ratio {empty_ratio * 100:.1f}%).",
        )

    # ── R2 – Orphan Detection ─────────────────────────────────────
    linked = sum(
        1 for v in validations
        if is_bbox_valid(v.get("text_bounding_box"), page_w, page_h)
        and is_bbox_valid(v.get("geometry_bounding_box"), page_w, page_h)
    )
    link_ratio = linked / max(n_axioms, 1)

    result.stat("fully_linked_validations", linked)
    result.stat("link_ratio",               f"{link_ratio * 100:.1f}%")
    result.stat("min_link_ratio_threshold", f"{MIN_LINK_RATIO * 100:.0f}%")

    if n_axioms == 0:
        result.warn("No axioms found — R2 skipped.")
    elif link_ratio < MIN_LINK_RATIO:
        result.rule_fail(
            "R2",
            f"Orphan Detection: {linked}/{n_axioms} geometries linked "
            f"({link_ratio * 100:.1f}% < {MIN_LINK_RATIO * 100:.0f}%).",
        )
    else:
        result.rule_pass(
            "R2",
            f"Link ratio acceptable: {linked}/{n_axioms} "
            f"({link_ratio * 100:.1f}%).",
        )

    # ── R4 – Bbox Dimensionality ──────────────────────────────────
    bad_dim: list[str] = []
    for a in axioms:
        b = a.get("bounding_box")
        if b is not None and not (isinstance(b, (list, tuple)) and len(b) == 4):
            bad_dim.append(f"axiom:{a['id']}")
    for v in validations:
        for key in ("text_bounding_box", "geometry_bounding_box"):
            b = v.get(key)
            if b is not None and not (isinstance(b, (list, tuple)) and len(b) == 4):
                bad_dim.append(f"val:{v['id']}:{key}")

    result.stat("malformed_bboxes", len(bad_dim))
    if bad_dim:
        result.rule_fail(
            "R4",
            f"{len(bad_dim)} malformed bbox(es) (need [x1,y1,x2,y2] 4-element list).",
        )
        for item in bad_dim[:3]:
            result.warn(f"Malformed: {item}")
    else:
        result.rule_pass("R4", "All bboxes are 4-element lists.")

    # ── R5 – Coordinate Sanity ────────────────────────────────────
    bad_coords: list[tuple] = []
    for a in axioms:
        reason = _check_coord_sanity(f"axiom:{a['id']}", a.get("bounding_box"))
        if reason:
            bad_coords.append((f"axiom:{a['id']}", reason))
    for v in validations:
        for key in ("text_bounding_box", "geometry_bounding_box"):
            reason = _check_coord_sanity(f"val:{v['id']}:{key}", v.get(key))
            if reason:
                bad_coords.append((f"val:{v['id']}:{key}", reason))

    result.stat("coord_sanity_failures", len(bad_coords))
    if bad_coords:
        result.rule_fail(
            "R5",
            f"{len(bad_coords)} bbox(es) have invalid coordinates.",
        )
        for item_id, reason in bad_coords[:3]:
            result.warn(f"Bad coord [{item_id}]: {reason}")
    else:
        result.rule_pass("R5", "All coordinates are valid and properly ordered.")

    # ── R6 – Status Distribution ──────────────────────────────────
    sc: dict[str, int] = defaultdict(int)
    for v in validations:
        sc[v["status"]] += 1

    result.stat("status_PASS",     sc["PASS"])
    result.stat("status_FAIL",     sc["FAIL"])
    result.stat("status_CONFLICT", sc["CONFLICT"])
    result.stat("status_UNKNOWN",  sc["UNKNOWN"])
    result.stat("pass_rate",       f"{sc['PASS'] / max(n_vals, 1) * 100:.1f}%")

    if n_vals > 0 and sc["PASS"] == 0 and (sc["FAIL"] + sc["CONFLICT"]) > 0:
        result.rule_fail(
            "R6",
            f"0 % PASS rate — FAIL={sc['FAIL']} CONFLICT={sc['CONFLICT']}. "
            "DHMoT may be misconfigured.",
        )
    elif n_vals > 0 and sc["UNKNOWN"] == n_vals:
        result.rule_fail(
            "R6",
            f"All {n_vals} validations have UNKNOWN status. "
            "Status field may not be populating.",
        )
    else:
        result.rule_pass(
            "R6",
            f"Status: PASS={sc['PASS']} FAIL={sc['FAIL']} "
            f"CONFLICT={sc['CONFLICT']} UNKNOWN={sc['UNKNOWN']}.",
        )

    # ── R7 – Canvas Bounds ────────────────────────────────────────
    oob: list[str] = []
    for a in axioms:
        b = a.get("bounding_box")
        if (
            isinstance(b, (list, tuple))
            and len(b) == 4
            and not is_bbox_valid(b, page_w, page_h)
        ):
            oob.append(f"axiom:{a['id']} → {b}")
    for v in validations:
        for key in ("text_bounding_box", "geometry_bounding_box"):
            b = v.get(key)
            if (
                isinstance(b, (list, tuple))
                and len(b) == 4
                and not is_bbox_valid(b, page_w, page_h)
            ):
                oob.append(f"val:{v['id']}:{key} → {b}")

    result.stat("out_of_bounds_bboxes", len(oob))
    if oob:
        result.rule_fail(
            "R7",
            f"{len(oob)} bbox(es) outside PDF canvas ({page_w}×{page_h}px). "
            "DPI mismatch or AI hallucination suspected.",
        )
        for item in oob[:3]:
            result.warn(f"OOB: {item}")
    else:
        result.rule_pass("R7", f"All bboxes within canvas ({page_w}×{page_h}px).")

    # ── Output & exit ─────────────────────────────────────────────
    result.print_report()
    result.save_txt(OUTPUT_REPORT_TXT)
    sys.exit(0 if result.overall_pass else 1)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 12 — PIXEL-LEVEL FIDELITY TESTER  §9 / MODE 3
# ══════════════════════════════════════════════════════════════════════════════

def run_pixel_fidelity_test(
    bundle       : AssetBundle,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
    save_crops   : bool  = False,
) -> dict:
    """
    §9 / MODE 3 — Pixel-Level Fidelity Tester

    For every bounding box in axioms and validations:
      1. Crop the region from the rendered PDF image.
      2. Measure ink (non-white pixel) ratio.
      3. FAIL if ink_ratio < ink_threshold  (possible hallucination).
    """
    log.info("═" * 60)
    log.info("  §9  MODE 3 — PIXEL-LEVEL FIDELITY TESTER")
    log.info("═" * 60)

    page  = bundle.page(DEFAULT_PAGE_IDX)
    bgr   = page.bgr_image
    h, w  = bgr.shape[:2]
    crops_results: list[CropResult] = []

    if save_crops and not os.path.exists(OUTPUT_CROPS_DIR):
        os.makedirs(OUTPUT_CROPS_DIR)
        log.info("Created crops directory: %s", OUTPUT_CROPS_DIR)

    def _test_one(record_id: str, record_type: str, bbox: Any) -> CropResult:
        if not is_bbox_valid(bbox, w, h):
            return CropResult(
                record_id=record_id,
                record_type=record_type,
                bbox=bbox,
                crop_shape=(0, 0, 0),
                mean_intensity=0.0,
                ink_pixel_ratio=0.0,
                has_ink=False,
                status="SKIP",
                reason="Invalid or out-of-bounds bbox",
            )

        crop = crop_bbox_from_image(bgr, bbox)
        if crop is None or crop.size == 0:
            return CropResult(
                record_id=record_id,
                record_type=record_type,
                bbox=bbox,
                crop_shape=(0, 0, 0),
                mean_intensity=0.0,
                ink_pixel_ratio=0.0,
                has_ink=False,
                status="SKIP",
                reason="Degenerate crop (zero area after clamp)",
            )

        mean_i, ink_ratio = measure_ink(crop)
        has_ink = ink_ratio >= ink_threshold

        if save_crops:
            safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", record_id)[:40]
            fn = os.path.join(
                OUTPUT_CROPS_DIR,
                f"{record_type}_{safe_id}_ink{ink_ratio:.2f}.png",
            )
            cv2.imwrite(fn, crop)

        return CropResult(
            record_id=record_id,
            record_type=record_type,
            bbox=bbox,
            crop_shape=crop.shape,
            mean_intensity=round(mean_i, 2),
            ink_pixel_ratio=round(ink_ratio, 4),
            has_ink=has_ink,
            status="PASS" if has_ink else "FAIL",
            reason=(
                f"ink={ink_ratio:.3f} ≥ {ink_threshold}"
                if has_ink
                else f"ink={ink_ratio:.3f} < {ink_threshold} → hallucination candidate"
            ),
        )

    log.info("Testing %d axiom bounding boxes …", len(bundle.axioms))
    for axiom in bundle.axioms:
        crops_results.append(
            _test_one(axiom["id"], "axiom", axiom.get("bounding_box"))
        )

    log.info("Testing %d validation bbox pairs …", len(bundle.validations))
    for val in bundle.validations:
        for key in ("text_bounding_box", "geometry_bounding_box"):
            crops_results.append(_test_one(val["id"], key, val.get(key)))

    pass_n = sum(1 for r in crops_results if r.status == "PASS")
    fail_n = sum(1 for r in crops_results if r.status == "FAIL")
    skip_n = sum(1 for r in crops_results if r.status == "SKIP")
    total  = len(crops_results)
    halluc = fail_n / max(total - skip_n, 1)

    print(f"\n  ┌─ §9 PIXEL FIDELITY RESULTS ──────────────────────────")
    print(f"  │  Total boxes tested  : {total}")
    print(f"  │  PASS (has ink)      : {pass_n}  ({pass_n / max(total, 1) * 100:.1f}%)")
    print(f"  │  FAIL (empty crop)   : {fail_n}  ({fail_n / max(total, 1) * 100:.1f}%)")
    print(f"  │  SKIP (invalid bbox) : {skip_n}")
    print(f"  │  Hallucination rate  : {halluc * 100:.1f}%")
    print(f"  └──────────────────────────────────────────────────────")

    if fail_n:
        print(f"\n  ⚠   HALLUCINATION SUSPECTS:")
        for r in crops_results:
            if r.status == "FAIL":
                print(
                    f"      [{r.record_type}:{r.record_id}] "
                    f"bbox={r.bbox}  {r.reason}"
                )

    ink_vals = [r.ink_pixel_ratio for r in crops_results if r.status != "SKIP"]
    if ink_vals:
        print_ascii_histogram(ink_vals, "Ink Ratio Distribution")

    overall = "PASS" if fail_n == 0 else "FAIL"
    log.info("§9 Verdict: %s", overall)

    return {
        "results"           : crops_results,
        "pass_count"        : pass_n,
        "fail_count"        : fail_n,
        "skip_count"        : skip_n,
        "hallucination_rate": halluc,
        "overall"           : overall,
    }

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 13 — OCR ROUND-TRIP VERIFIER  §10 / MODE 4
# ══════════════════════════════════════════════════════════════════════════════

def run_ocr_roundtrip_verifier(
    bundle         : AssetBundle,
    use_tesseract  : bool  = False,
    fuzzy_threshold: float = FUZZY_MATCH_THRESHOLD,
) -> dict:
    """
    §10 / MODE 4 — OCR Round-Trip Verifier

    For each validation:
      1. Crop the text_bounding_box region.
      2. Optionally re-run Tesseract and fuzzy-compare to JSON table_value.
      3. Classify all table_values against 12 CAD regex patterns.

    GAP-13 fix: returns "WARN" for mismatches (informational, exit 0)
                returns "FAIL" only for system errors.
    """
    log.info("═" * 60)
    log.info("  §10  MODE 4 — OCR ROUND-TRIP VERIFIER")
    log.info("═" * 60)

    page = bundle.page(DEFAULT_PAGE_IDX)
    bgr  = page.bgr_image
    h, w = bgr.shape[:2]

    if use_tesseract and not _TESSERACT_OK:
        log.warning("pytesseract not installed — falling back to pattern-only mode.")
        use_tesseract = False

    log.info("Tesseract re-OCR: %s", "ENABLED" if use_tesseract else "DISABLED")

    tess_config = (
        "--psm 8 --oem 3 "
        "-c tessedit_char_whitelist="
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "xX×./-°'kNMPamdeg% "
    )

    records   : list[dict]    = []
    class_dist: dict[str,int] = defaultdict(int)
    any_error = False

    for val in bundle.validations:
        val_id    = val["id"]
        json_text = str(val.get("table_value") or "").strip()
        t_bbox    = val.get("text_bounding_box")
        re_ocr    = ""
        fscore    = 1.0
        rstatus   = "NO_REOCR"

        # Crop text region
        crop = None
        if is_bbox_valid(t_bbox, w, h):
            crop = crop_bbox_from_image(bgr, t_bbox, padding=4)

        # Re-OCR
        if use_tesseract and crop is not None and crop.size > 0:
            try:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                if bw.shape[0] < 32:
                    scale = max(2, 32 // bw.shape[0])
                    bw = cv2.resize(
                        bw, None,
                        fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                re_ocr  = _pytesseract.image_to_string(
                    bw, config=tess_config
                ).strip()
                fscore  = fuzzy_match_score(json_text, re_ocr)
                rstatus = "MATCH" if fscore >= fuzzy_threshold else "MISMATCH"
            except Exception as exc:
                re_ocr    = f"[ERROR: {exc}]"
                rstatus   = "ERROR"
                fscore    = 0.0
                any_error = True

        pat_class = classify_cad_text(json_text)
        class_dist[pat_class] += 1

        records.append(
            {
                "id"           : val_id,
                "json_text"    : json_text,
                "re_ocr_text"  : re_ocr,
                "fuzzy_score"  : round(fscore, 3),
                "reocr_status" : rstatus,
                "pattern_class": pat_class,
                "has_crop"     : crop is not None,
            }
        )

    total      = len(records)
    match_n    = sum(1 for r in records if r["reocr_status"] == "MATCH")
    mismatch_n = sum(1 for r in records if r["reocr_status"] == "MISMATCH")
    error_n    = sum(1 for r in records if r["reocr_status"] == "ERROR")
    empty_n    = sum(1 for r in records if r["pattern_class"] == "empty")
    unclass_n  = sum(1 for r in records if r["pattern_class"] == "unclassified")

    print(f"\n  ┌─ §10 OCR ROUND-TRIP RESULTS ─────────────────────────")
    print(f"  │  Total validations   : {total}")
    print(f"  │  Tesseract active    : {use_tesseract}")
    print(f"  │  MATCH (≥{fuzzy_threshold:.2f})       : {match_n}")
    print(f"  │  MISMATCH            : {mismatch_n}")
    print(f"  │  ERROR               : {error_n}")
    print(f"  │  Empty table_values  : {empty_n}")
    print(f"  │  Unclassified        : {unclass_n}")
    print(f"  └──────────────────────────────────────────────────────")

    print(f"\n  CAD Pattern Classification:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        bar = "█" * min(cnt, 45)
        print(f"    {cls:<22} {cnt:>5}  {bar}")

    if mismatch_n > 0 and use_tesseract:
        print(f"\n  OCR MISMATCHES (score < {fuzzy_threshold}):")
        for r in records:
            if r["reocr_status"] == "MISMATCH":
                print(
                    f"    [{r['id']}]  JSON='{r['json_text']}'  "
                    f"ReOCR='{r['re_ocr_text']}'  score={r['fuzzy_score']}"
                )

    # GAP-13 fix: WARN ≠ FAIL
    if any_error:
        overall = "WARN"
    elif mismatch_n > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    log.info("§10 Verdict: %s", overall)

    return {
        "records"          : records,
        "total"            : total,
        "match_count"      : match_n,
        "mismatch_count"   : mismatch_n,
        "error_count"      : error_n,
        "empty_count"      : empty_n,
        "class_distribution": dict(class_dist),
        "overall"          : overall,
    }

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 14 — GEOMETRIC MATH VERIFIER  §11 / MODE 5
# ══════════════════════════════════════════════════════════════════════════════

def run_geometric_math_verifier(
    bundle            : AssetBundle,
    distance_tolerance: float = DISTANCE_TOLERANCE,
    pass_max_distance : float = DEFAULT_PASS_MAX_DIST,
    fail_min_distance : float = DEFAULT_FAIL_MIN_DIST,
) -> dict:
    """
    §11 / MODE 5 — Geometric Math Verifier

    Re-computes Euclidean distances from stored bboxes and verifies
    that the PASS/FAIL status assignments are mathematically consistent.
    """
    log.info("═" * 60)
    log.info("  §11  MODE 5 — GEOMETRIC MATH VERIFIER")
    log.info("═" * 60)

    math_results: list[MathCheckResult] = []
    deltas: list[float] = []

    for val in bundle.validations:
        val_id     = val["id"]
        t_bbox     = val.get("text_bounding_box")
        g_bbox     = val.get("geometry_bounding_box")
        json_status= val["status"]
        json_dist  = val.get("distance")

        if not isinstance(t_bbox, (list, tuple)) or not isinstance(g_bbox, (list, tuple)):
            math_results.append(
                MathCheckResult(
                    val_id=val_id,
                    json_status=json_status,
                    json_distance=json_dist,
                    computed_distance=-1.0,
                    distance_delta=-1.0,
                    bbox_overlap_iou=0.0,
                    spatial_verdict="SKIP",
                    note="One or both bboxes missing",
                )
            )
            continue

        comp_dist = compute_euclidean(t_bbox, g_bbox)
        iou       = compute_iou(t_bbox, g_bbox)

        if comp_dist < 0:
            math_results.append(
                MathCheckResult(
                    val_id=val_id,
                    json_status=json_status,
                    json_distance=json_dist,
                    computed_distance=-1.0,
                    distance_delta=-1.0,
                    bbox_overlap_iou=iou,
                    spatial_verdict="SKIP",
                    note="Cannot compute distance (invalid coords)",
                )
            )
            continue

        delta = 0.0
        if json_dist is not None:
            try:
                delta = abs(float(json_dist) - comp_dist)
                deltas.append(delta)
            except (TypeError, ValueError):
                delta = -1.0

        notes      : list[str] = []
        consistent  = True

        if json_status == "PASS" and comp_dist > pass_max_distance:
            consistent = False
            notes.append(
                f"PASS but dist={comp_dist:.1f}px > max={pass_max_distance}px"
            )
        if json_status == "FAIL" and comp_dist < fail_min_distance:
            consistent = False
            notes.append(
                f"FAIL but dist={comp_dist:.1f}px < min={fail_min_distance}px"
            )
        if delta > distance_tolerance and json_dist is not None:
            consistent = False
            notes.append(
                f"distance delta={delta:.1f}px > tolerance={distance_tolerance}px"
            )
        if iou > 0.5:
            notes.append(f"HIGH IoU={iou:.3f} — boxes heavily overlap")

        math_results.append(
            MathCheckResult(
                val_id=val_id,
                json_status=json_status,
                json_distance=float(json_dist) if json_dist is not None else None,
                computed_distance=round(comp_dist, 2),
                distance_delta=round(delta, 2),
                bbox_overlap_iou=round(iou, 4),
                spatial_verdict="CONSISTENT" if consistent else "INCONSISTENT",
                note="; ".join(notes) if notes else "OK",
            )
        )

    consistent_n = sum(1 for r in math_results if r.spatial_verdict == "CONSISTENT")
    incon_n      = sum(1 for r in math_results if r.spatial_verdict == "INCONSISTENT")
    skip_n       = sum(1 for r in math_results if r.spatial_verdict == "SKIP")

    valid_dists = [r.computed_distance for r in math_results if r.computed_distance >= 0]
    avg_dist    = sum(valid_dists) / max(len(valid_dists), 1)
    max_delta   = max(deltas) if deltas else 0.0
    avg_delta   = sum(deltas) / max(len(deltas), 1)

    print(f"\n  ┌─ §11 MATH VERIFIER RESULTS ───────────────────────────")
    print(f"  │  Total validations : {len(math_results)}")
    print(f"  │  CONSISTENT        : {consistent_n}")
    print(f"  │  INCONSISTENT      : {incon_n}")
    print(f"  │  SKIP              : {skip_n}")
    print(f"  │  Avg distance      : {avg_dist:.1f} px")
    print(f"  │  Avg delta vs JSON : {avg_delta:.2f} px")
    print(f"  │  Max delta         : {max_delta:.2f} px")
    print(f"  └───────────────────────────────────────────────────────")

    if incon_n:
        print(f"\n  INCONSISTENT RECORDS:")
        for r in math_results:
            if r.spatial_verdict == "INCONSISTENT":
                print(
                    f"    [{r.val_id}]  status={r.json_status}  "
                    f"dist={r.computed_distance:.1f}px  "
                    f"delta={r.distance_delta:.1f}px  "
                    f"iou={r.bbox_overlap_iou:.3f}  → {r.note}"
                )

    if valid_dists:
        print_ascii_histogram(valid_dists, "Computed Distance Distribution (px)")

    overall = "PASS" if incon_n == 0 else "FAIL"
    log.info("§11 Verdict: %s", overall)

    return {
        "results"        : math_results,
        "consistent_n"   : consistent_n,
        "inconsistent_n" : incon_n,
        "skip_n"         : skip_n,
        "avg_distance"   : avg_dist,
        "avg_delta"      : avg_delta,
        "max_delta"      : max_delta,
        "overall"        : overall,
    }

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 15 — HALLUCINATION FINGERPRINTER  §12 / MODE 6
# ══════════════════════════════════════════════════════════════════════════════

def run_hallucination_fingerprinter(bundle: AssetBundle) -> dict:
    """
    §12 / MODE 6 — Hallucination Fingerprinter

    Six detection algorithms:
      H1 – Duplicate Bbox    : IoU > 90 % between any two geometry boxes
      H2 – Ghost Box         : area < 100 px²
      H3 – Impossible Size   : area ≥ 95 % of page
      H4 – Degenerate Text   : repeated chars or line-noise OCR artifacts
      H5 – Spatial Cluster   : all boxes in < 5 % of page spread
      H6 – Phantom Links     : validation geom_bbox matches no axiom

    GAP-14 fix: H1 capped at 200 pairs to avoid O(N²) blowout.
    """
    log.info("═" * 60)
    log.info("  §12  MODE 6 — HALLUCINATION FINGERPRINTER")
    log.info("═" * 60)

    page   = bundle.page(DEFAULT_PAGE_IDX)
    page_w = page.width_px
    page_h = page.height_px
    flags  : list[HallucinationFlag] = []

    valid_axiom_bboxes = [
        (a["id"], a["bounding_box"])
        for a in bundle.axioms
        if is_bbox_valid(a.get("bounding_box"), page_w, page_h)
    ]

    # ── H1: Duplicate Bboxes ──────────────────────────────────────
    log.info("  H1: Duplicate bboxes (capped at 200) …")
    h1_pool = valid_axiom_bboxes[:200]
    if len(valid_axiom_bboxes) > 200:
        log.warning(
            "  H1: Dataset has %d axioms — only first 200 compared for performance.",
            len(valid_axiom_bboxes),
        )

    seen: set[tuple] = set()
    for (id_a, bb_a), (id_b, bb_b) in itertools.combinations(h1_pool, 2):
        pair = tuple(sorted([id_a, id_b]))
        if pair in seen:
            continue
        seen.add(pair)
        iou = compute_iou(bb_a, bb_b)
        if iou >= DUPLICATE_IOU_THR:
            flags.append(
                HallucinationFlag(
                    flag_type="DUPLICATE_BBOX",
                    severity="HIGH",
                    record_id=f"{id_a} ↔ {id_b}",
                    detail=f"IoU={iou:.3f} ≥ {DUPLICATE_IOU_THR}  bbox_a={bb_a}",
                )
            )

    # ── H2: Ghost Boxes ───────────────────────────────────────────
    log.info("  H2: Ghost boxes (area < %.0f px²) …", GHOST_BOX_MIN_AREA)
    for a_id, bb in valid_axiom_bboxes:
        area = bbox_area(bb)
        if area < GHOST_BOX_MIN_AREA:
            flags.append(
                HallucinationFlag(
                    flag_type="GHOST_BOX",
                    severity="MEDIUM",
                    record_id=a_id,
                    detail=f"area={area:.1f} px² < min={GHOST_BOX_MIN_AREA} px²  bbox={bb}",
                )
            )

    # ── H3: Impossible Size ───────────────────────────────────────
    log.info("  H3: Impossible size (≥ %.0f %% of page) …", MAX_BOX_PAGE_RATIO * 100)
    page_area = page_w * page_h
    for a_id, bb in valid_axiom_bboxes:
        area = bbox_area(bb)
        if area > page_area * MAX_BOX_PAGE_RATIO:
            flags.append(
                HallucinationFlag(
                    flag_type="IMPOSSIBLE_SIZE",
                    severity="HIGH",
                    record_id=a_id,
                    detail=(
                        f"area={area:.0f} px² ≥ "
                        f"{MAX_BOX_PAGE_RATIO * 100:.0f}% of page {page_area} px²"
                    ),
                )
            )

    # ── H4: Degenerate OCR Text ───────────────────────────────────
    log.info("  H4: Degenerate text values …")
    ocr_artifact_re = re.compile(r"[|\\]{3,}|_{4,}|\.{4,}")
    for val in bundle.validations:
        tv = str(val.get("table_value") or "").strip()
        if len(tv) > 2 and len(set(tv.replace(" ", ""))) == 1:
            flags.append(
                HallucinationFlag(
                    flag_type="DEGENERATE_TEXT",
                    severity="MEDIUM",
                    record_id=val["id"],
                    detail=f"table_value='{tv}' is all repeated characters",
                )
            )
        if ocr_artifact_re.search(tv):
            flags.append(
                HallucinationFlag(
                    flag_type="OCR_ARTIFACT",
                    severity="LOW",
                    record_id=val["id"],
                    detail=f"table_value='{tv}' contains line-noise characters",
                )
            )

    # ── H5: Spatial Clustering ────────────────────────────────────
    log.info("  H5: Spatial clustering …")
    if len(valid_axiom_bboxes) > 5:
        cxs     = [bbox_center(bb)[0] for _, bb in valid_axiom_bboxes]
        cys     = [bbox_center(bb)[1] for _, bb in valid_axiom_bboxes]
        spread_x = max(cxs) - min(cxs)
        spread_y = max(cys) - min(cys)
        if (
            spread_x < page_w * CLUSTER_SPREAD_RATIO
            and spread_y < page_h * CLUSTER_SPREAD_RATIO
        ):
            flags.append(
                HallucinationFlag(
                    flag_type="SPATIAL_CLUSTERING",
                    severity="HIGH",
                    record_id="ALL_AXIOMS",
                    detail=(
                        f"{len(valid_axiom_bboxes)} axioms span only "
                        f"{spread_x:.0f}×{spread_y:.0f} px "
                        f"(page={page_w}×{page_h} px). "
                        "Geometry may not cover the full drawing."
                    ),
                )
            )

    # ── H6: Phantom Links ─────────────────────────────────────────
    log.info("  H6: Phantom validation links …")
    axiom_keys: set[tuple] = {
        tuple(int(v) for v in bb)
        for _, bb in valid_axiom_bboxes
    }

    for val in bundle.validations:
        g_bbox = val.get("geometry_bounding_box")
        if not is_bbox_valid(g_bbox, page_w, page_h):
            continue
        g_key   = tuple(int(v) for v in g_bbox)
        matched = any(
            all(abs(g_key[i] - ak[i]) <= 5 for i in range(4))
            for ak in axiom_keys
        )
        if not matched:
            flags.append(
                HallucinationFlag(
                    flag_type="PHANTOM_LINK",
                    severity="HIGH",
                    record_id=val["id"],
                    detail=(
                        f"geometry_bounding_box={g_bbox} matches no axiom "
                        "(±5 px tolerance)"
                    ),
                )
            )

    # ── Summary ───────────────────────────────────────────────────
    high_n = sum(1 for f in flags if f.severity == "HIGH")
    med_n  = sum(1 for f in flags if f.severity == "MEDIUM")
    low_n  = sum(1 for f in flags if f.severity == "LOW")

    print(f"\n  ┌─ §12 HALLUCINATION SCAN RESULTS ──────────────────────")
    print(f"  │  Total flags   : {len(flags)}")
    print(f"  │  HIGH    🔴    : {high_n}")
    print(f"  │  MEDIUM  🟡    : {med_n}")
    print(f"  │  LOW     🔵    : {low_n}")
    print(f"  └───────────────────────────────────────────────────────")

    sev_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
    for fl in flags:
        icon = sev_icon.get(fl.severity, "⚪")
        print(f"  {icon}  [{fl.flag_type}] [{fl.record_id}] {fl.detail}")

    overall = "PASS" if high_n == 0 else "FAIL"
    log.info("§12 Verdict: %s  (%d flags total)", overall, len(flags))

    return {
        "flags"       : flags,
        "high_count"  : high_n,
        "medium_count": med_n,
        "low_count"   : low_n,
        "total_flags" : len(flags),
        "overall"     : overall,
    }

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 16 — FULL REGRESSION SUITE  §13 / MODE 7
# ══════════════════════════════════════════════════════════════════════════════

def run_full_regression(
    bundle       : AssetBundle,
    use_tesseract: bool  = False,
    save_crops   : bool  = False,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> None:
    """
    §13 / MODE 7 — Full Regression Suite

    Runs §9 → §10 → §11 → §12 in sequence.
    Scores each section and produces regression_report.json.

    Scoring  : PASS=1.0  WARN=0.5  FAIL=0.0  SKIP=excluded
    Verdict  : PASS if score==100 %, else FAIL
    Exit code: 0 (PASS) or 1 (FAIL)

    GAP-15 fix: per-section timing in regression_report.json
    """
    print("\n" + "█" * 65)
    print(f"  FULL REGRESSION SUITE — {__pipeline__}  v{__version__}")
    print("█" * 65)

    page = bundle.page(DEFAULT_PAGE_IDX)
    print(f"\n  📁  Assets:")
    print(f"    PDF         : {os.path.basename(bundle.pdf_path)}")
    print(f"    JSON        : {os.path.basename(bundle.json_path)}"
          f"  ({bundle.json_size_kb:.1f} KB)")
    print(f"    Page size   : {page.width_px}×{page.height_px} px @ {bundle.dpi} DPI")
    print(f"    Page hash   : {page.page_hash}")
    print(f"    Schema ver  : {bundle.schema_version}")
    print(f"    Axioms      : {len(bundle.axioms)}")
    print(f"    Validations : {len(bundle.validations)}")
    print(f"    Asset load  : {bundle.load_time_s:.3f}s")

    suite: list[SuiteResult] = []

    def _run_section(
        sid : str,
        name: str,
        fn  : Any,
        **kwargs: Any,
    ) -> SuiteResult:
        print(f"\n  ▶  {sid}: {name} …")
        t0 = time.time()
        try:
            detail  = fn(bundle=bundle, **kwargs)
            overall = str(detail.get("overall", "SKIP"))
        except Exception as exc:
            log.error("Section %s crashed: %s", sid, exc)
            traceback.print_exc()
            detail  = {"error": str(exc)}
            overall = "FAIL"
        dur = round(time.time() - t0, 3)
        icon = {"PASS": "✔", "FAIL": "✘", "WARN": "⚠", "SKIP": "–"}.get(overall, "?")
        print(f"  {icon}  {sid} → {overall}  ({dur}s)")
        return SuiteResult(
            section_id=sid, name=name,
            overall=overall, duration_s=dur, detail=detail,
        )

    suite.append(
        _run_section(
            "S9", "Pixel-Level Fidelity",
            run_pixel_fidelity_test,
            ink_threshold=ink_threshold,
            save_crops=save_crops,
        )
    )
    suite.append(
        _run_section(
            "S10", "OCR Round-Trip Verifier",
            run_ocr_roundtrip_verifier,
            use_tesseract=use_tesseract,
        )
    )
    suite.append(
        _run_section(
            "S11", "Geometric Math Verifier",
            run_geometric_math_verifier,
        )
    )
    suite.append(
        _run_section(
            "S12", "Hallucination Fingerprinter",
            run_hallucination_fingerprinter,
        )
    )

    # ── Scoring ───────────────────────────────────────────────────
    score_map = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0, "SKIP": None}
    votes: list[float] = []
    for r in suite:
        v = score_map.get(r.overall)
        if v is not None:
            votes.append(v)

    score_pct = (sum(votes) / max(len(votes), 1)) * 100
    n_pass    = sum(1 for r in suite if r.overall == "PASS")
    n_warn    = sum(1 for r in suite if r.overall == "WARN")
    n_fail    = sum(1 for r in suite if r.overall == "FAIL")
    verdict   = "PASS" if n_fail == 0 else "FAIL"
    total_time = sum(r.duration_s for r in suite)

    # ── Final report table ────────────────────────────────────────
    icon_map = {
        "PASS": "✔ PASS",
        "FAIL": "✘ FAIL",
        "WARN": "⚠ WARN",
        "SKIP": "– SKIP",
    }

    print("\n\n" + "█" * 65)
    print("  FINAL REGRESSION REPORT")
    print("█" * 65)
    print(f"\n  {'Sec':<5}  {'Name':<32}  {'Result':<8}  {'Time':>8}")
    print("  " + "─" * 57)

    for r in suite:
        icon = icon_map.get(r.overall, "? UNKN")
        print(
            f"  {r.section_id:<5}  {r.name:<32}  {icon:<8}  "
            f"{r.duration_s:>7.3f}s"
        )

    print("  " + "─" * 57)
    print(
        f"\n  Score       : {score_pct:.1f}%  "
        f"({n_pass} PASS / {n_warn} WARN / {n_fail} FAIL)"
    )
    print(f"  Total time  : {total_time:.3f}s  (load: {bundle.load_time_s:.3f}s)")
    print(
        f"\n  VERDICT     : "
        + (
            "✔  ALL SECTIONS PASSED"
            if verdict == "PASS"
            else "✘  FAILURES DETECTED — OUTPUT INTEGRITY COMPROMISED"
        )
    )
    print("█" * 65 + "\n")

    # ── Save regression_report.json (GAP-15: includes timing) ────
    report = {
        "version"            : __version__,
        "pipeline"           : __pipeline__,
        "pdf_path"           : bundle.pdf_path,
        "json_path"          : bundle.json_path,
        "json_size_kb"       : round(bundle.json_size_kb, 2),
        "schema_version"     : bundle.schema_version,
        "dpi"                : bundle.dpi,
        "page_size_px"       : [page.width_px, page.height_px],
        "page_hash"          : page.page_hash,
        "total_axioms"       : len(bundle.axioms),
        "total_validations"  : len(bundle.validations),
        "asset_load_time_s"  : bundle.load_time_s,
        "total_suite_time_s" : round(total_time, 3),
        "score_pct"          : round(score_pct, 2),
        "verdict"            : verdict,
        "sections"           : [
            {
                "id"        : r.section_id,
                "name"      : r.name,
                "overall"   : r.overall,
                "duration_s": r.duration_s,
                "score_vote": score_map.get(r.overall),
            }
            for r in suite
        ],
    }

    try:
        with open(OUTPUT_REG_JSON, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        log.info("Regression report saved: %s", OUTPUT_REG_JSON)
    except OSError as exc:
        log.warning("Could not save regression report: %s", exc)

    sys.exit(0 if verdict == "PASS" else 1)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 17 — UNIT TESTS  §14 / MODE 8
# ══════════════════════════════════════════════════════════════════════════════

class TestGeometryUtils(unittest.TestCase):
    """Tests for §6 geometry/math utilities."""

    # ── is_bbox_valid ────────────────────────────────────────────
    def test_bbox_valid_none(self):
        self.assertFalse(is_bbox_valid(None, 1000, 800))

    def test_bbox_valid_zero_area(self):
        self.assertFalse(is_bbox_valid([0, 0, 0, 0], 1000, 800))

    def test_bbox_valid_zero_width(self):
        self.assertFalse(is_bbox_valid([100, 100, 100, 200], 1000, 800))

    def test_bbox_valid_zero_height(self):
        self.assertFalse(is_bbox_valid([100, 100, 200, 100], 1000, 800))

    def test_bbox_valid_normal(self):
        self.assertTrue(is_bbox_valid([10, 10, 200, 300], 1000, 800))

    def test_bbox_valid_oob_left(self):
        self.assertFalse(is_bbox_valid([-200, 0, 100, 100], 1000, 800))

    def test_bbox_valid_oob_right(self):
        self.assertFalse(is_bbox_valid([0, 0, 1200, 100], 1000, 800))

    def test_bbox_valid_string(self):
        self.assertFalse(is_bbox_valid("not_a_bbox", 1000, 800))

    def test_bbox_valid_three_elements(self):
        self.assertFalse(is_bbox_valid([1, 2, 3], 1000, 800))

    def test_bbox_valid_within_margin(self):
        # Exactly at margin boundary should pass
        self.assertTrue(is_bbox_valid([-49, -49, 100, 100], 1000, 800))

    # ── compute_iou ──────────────────────────────────────────────
    def test_iou_partial_overlap(self):
        iou = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        expected = (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)
        self.assertAlmostEqual(iou, expected, places=4)

    def test_iou_no_overlap(self):
        self.assertEqual(compute_iou([0, 0, 50, 50], [100, 100, 200, 200]), 0.0)

    def test_iou_identical(self):
        self.assertAlmostEqual(
            compute_iou([0, 0, 100, 100], [0, 0, 100, 100]), 1.0, places=5
        )

    def test_iou_none_input(self):
        self.assertEqual(compute_iou(None, [0, 0, 100, 100]), 0.0)

    def test_iou_string_input(self):
        self.assertEqual(compute_iou([0, 0, 100, 100], "bad"), 0.0)

    def test_iou_contained(self):
        # Inner box completely inside outer box
        iou = compute_iou([0, 0, 100, 100], [25, 25, 75, 75])
        inner_area = 50 * 50
        outer_area = 100 * 100
        expected   = inner_area / outer_area
        self.assertAlmostEqual(iou, expected, places=4)

    # ── fuzzy_match_score ────────────────────────────────────────
    def test_fuzzy_identical(self):
        self.assertEqual(fuzzy_match_score("400x400", "400x400"), 1.0)

    def test_fuzzy_near_match(self):
        score = fuzzy_match_score("400x400", "400×400")
        self.assertGreater(score, 0.7)

    def test_fuzzy_empty_both(self):
        self.assertEqual(fuzzy_match_score("", ""), 1.0)

    def test_fuzzy_one_empty(self):
        self.assertEqual(fuzzy_match_score("", "C1"), 0.0)

    def test_fuzzy_none_inputs(self):
        self.assertEqual(fuzzy_match_score(None, None), 1.0)

    def test_fuzzy_completely_different(self):
        score = fuzzy_match_score("ABCDEF", "123456")
        self.assertLess(score, 0.4)

    # ── classify_cad_text ────────────────────────────────────────
    def test_classify_column_id(self):
        self.assertEqual(classify_cad_text("C1"), "column_id")

    def test_classify_column_id_multi(self):
        self.assertEqual(classify_cad_text("B12"), "column_id")

    def test_classify_dimension(self):
        self.assertEqual(classify_cad_text("400x400"), "dimension")

    def test_classify_dimension_upper(self):
        self.assertEqual(classify_cad_text("300X600"), "dimension")

    def test_classify_empty_string(self):
        self.assertEqual(classify_cad_text(""), "empty")

    def test_classify_none(self):
        self.assertEqual(classify_cad_text(None), "empty")

    def test_classify_whitespace(self):
        self.assertEqual(classify_cad_text("   "), "empty")

    def test_classify_footing(self):
        self.assertEqual(classify_cad_text("F1"), "footing_tag")

    def test_classify_footing_alpha(self):
        self.assertEqual(classify_cad_text("F2A"), "footing_tag")

    def test_classify_rebar(self):
        self.assertEqual(classify_cad_text("4T20"), "rebar")

    def test_classify_unclassified(self):
        self.assertEqual(classify_cad_text("XYZWHATEVER999"), "unclassified")

    # ── print_ascii_histogram ─────────────────────────────────────
    def test_histogram_equal_values_no_crash(self):
        """GAP-16: all-equal values must not raise ZeroDivisionError."""
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            print_ascii_histogram([50.0, 50.0, 50.0], bins=8)
        finally:
            sys.stdout = old
        self.assertIn("50", buf.getvalue())

    def test_histogram_empty_no_crash(self):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            print_ascii_histogram([])
        finally:
            sys.stdout = old
        self.assertIn("no data", buf.getvalue())

    def test_histogram_normal(self):
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            print_ascii_histogram([10.0, 50.0, 100.0, 200.0], bins=4)
        finally:
            sys.stdout = old
        output = buf.getvalue()
        self.assertIn("█", output)

    # ── bbox_center ───────────────────────────────────────────────
    def test_bbox_center(self):
        cx, cy = bbox_center([0, 0, 100, 100])
        self.assertEqual(cx, 50.0)
        self.assertEqual(cy, 50.0)

    def test_bbox_center_asymmetric(self):
        cx, cy = bbox_center([10, 20, 110, 60])
        self.assertEqual(cx, 60.0)
        self.assertEqual(cy, 40.0)

    # ── bbox_area ─────────────────────────────────────────────────
    def test_bbox_area_normal(self):
        self.assertEqual(bbox_area([0, 0, 100, 200]), 20000.0)

    def test_bbox_area_zero(self):
        self.assertEqual(bbox_area([50, 50, 50, 50]), 0.0)

    def test_bbox_area_none(self):
        self.assertEqual(bbox_area(None), 0.0)

    # ── compute_euclidean ─────────────────────────────────────────
    def test_euclidean_known_distance(self):
        # Centers: (5,5) and (8,9) → distance = √(9+16) = 5
        dist = compute_euclidean([0, 0, 10, 10], [3, 4, 13, 14])
        self.assertAlmostEqual(dist, 5.0, places=3)

    def test_euclidean_same_center(self):
        dist = compute_euclidean([0, 0, 10, 10], [0, 0, 10, 10])
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_euclidean_invalid(self):
        self.assertEqual(compute_euclidean(None, [0, 0, 10, 10]), -1.0)

    # ── extract_axioms ────────────────────────────────────────────
    def test_extract_axioms_empty_dict(self):
        self.assertEqual(extract_axioms({}), [])

    def test_extract_axioms_empty_list(self):
        self.assertEqual(extract_axioms({"axioms": []}), [])

    def test_extract_axioms_filters_non_dicts(self):
        result = extract_axioms({"axioms": [None, "bad", 42, {"id": "A1"}]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "A1")

    def test_extract_axioms_fallback_key_nodes(self):
        result = extract_axioms({"nodes": [{"id": "N1"}]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "N1")

    def test_extract_axioms_default_label(self):
        result = extract_axioms({"axioms": [{"id": "X1"}]})
        self.assertEqual(result[0]["label"], "GEOM")

    # ── extract_validations ───────────────────────────────────────
    def test_extract_validations_empty(self):
        self.assertEqual(extract_validations({}), [])

    def test_status_normalisation_passed(self):
        data   = {"validations": [{"id": "v1", "status": "PASSED"}]}
        result = extract_validations(data)
        self.assertEqual(result[0]["status"], "PASS")

    def test_status_normalisation_failed(self):
        data   = {"validations": [{"id": "v1", "status": "failed"}]}
        result = extract_validations(data)
        self.assertEqual(result[0]["status"], "FAIL")

    def test_status_normalisation_ok(self):
        data   = {"validations": [{"id": "v1", "status": "OK"}]}
        result = extract_validations(data)
        self.assertEqual(result[0]["status"], "PASS")

    def test_status_unknown_passthrough(self):
        data   = {"validations": [{"id": "v1", "status": "BANANA"}]}
        result = extract_validations(data)
        self.assertEqual(result[0]["status"], "UNKNOWN")


class TestMeasureInk(unittest.TestCase):
    """Tests for §6 ink measurement utility."""

    def test_all_white_image(self):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        mean_i, ink_ratio = measure_ink(white)
        self.assertAlmostEqual(ink_ratio, 0.0, places=4)
        self.assertAlmostEqual(mean_i, 255.0, places=1)

    def test_all_black_image(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        mean_i, ink_ratio = measure_ink(black)
        self.assertAlmostEqual(ink_ratio, 1.0, places=4)
        self.assertAlmostEqual(mean_i, 0.0, places=1)

    def test_half_black(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        img[:50, :] = 0
        _, ink_ratio = measure_ink(img)
        self.assertAlmostEqual(ink_ratio, 0.5, places=2)

    def test_ink_threshold_white(self):
        """TC10: all-white image → ink_ratio below default threshold."""
        white = np.full((200, 200, 3), 255, dtype=np.uint8)
        _, ink_ratio = measure_ink(white)
        self.assertLess(ink_ratio, DEFAULT_INK_THRESHOLD)

    def test_ink_threshold_black(self):
        """All-black image → ink_ratio above threshold."""
        black = np.zeros((200, 200, 3), dtype=np.uint8)
        _, ink_ratio = measure_ink(black)
        self.assertGreater(ink_ratio, DEFAULT_INK_THRESHOLD)

    def test_single_pixel_crop(self):
        """Ensure no crash on 1×1 pixel crop."""
        pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        mean_i, ink_ratio = measure_ink(pixel)
        self.assertIsInstance(mean_i, float)
        self.assertIsInstance(ink_ratio, float)


def run_self_tests() -> None:
    """
    §14 / MODE 8 — Unit Test Runner.
    Runs all built-in tests. Exits 0 (all pass) or 1 (any fail).
    No PDF or JSON files required.
    """
    print("\n" + "═" * 65)
    print(f"  §14  UNIT TEST SUITE  —  CAD Audit System  v{__version__}")
    print("═" * 65)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGeometryUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasureInk))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print(f"\n  Tests run   : {result.testsRun}")
    print(f"  Failures    : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print(
        f"  Verdict     : "
        + ("✔  ALL PASS" if result.wasSuccessful() else "✘  FAILURES DETECTED")
    )
    print("═" * 65)
    sys.exit(0 if result.wasSuccessful() else 1)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 18 — CLI ENTRYPOINT  §15
# ══════════════════════════════════════════════════════════════════════════════

def build_cli() -> argparse.ArgumentParser:
    """Build the complete argument parser for all 8 modes."""
    parser = argparse.ArgumentParser(
        prog="audit_script.py",
        description=(
            f"Hybrid Neuro-Symbolic CAD Compliance Auditor  v{__version__}\n"
            "Verifies axiom_manifest.json faithfully represents the source PDF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
MODES
─────
  --visual        §7  Generate OpenCV overlay PNG
  --headless      §8  CI/CD rule checks R1–R7 (exit 0/1)
  --pixel         §9  Pixel-level ink fidelity per bbox
  --ocr           §10 OCR round-trip + CAD pattern classification
  --math          §11 Re-compute DHMoT Euclidean distances
  --fingerprint   §12 Hallucination pattern detection (H1–H6)
  --regression    §13 Full suite §9+§10+§11+§12, scored report
  --selftest      §14 Built-in unit tests (no files needed)

EXAMPLES
────────
  python audit_script.py drawing.pdf manifest.json --visual
  python audit_script.py drawing.pdf manifest.json --headless
  python audit_script.py drawing.pdf manifest.json --regression
  python audit_script.py drawing.pdf manifest.json --regression --tesseract --save-crops
  python audit_script.py drawing.pdf manifest.json --pixel --ink-threshold 0.03
  python audit_script.py drawing.pdf manifest.json --math --pass-max-dist 400
  python audit_script.py --selftest
        """,
    )

    # Positional (optional — not needed for --selftest)
    parser.add_argument(
        "pdf_path", nargs="?", default=None,
        help="Source structural CAD PDF path.",
    )
    parser.add_argument(
        "json_path", nargs="?", default=None,
        help="axiom_manifest.json path.",
    )

    # Mode group (mutually exclusive, exactly one required)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--visual", action="store_true",
        help="§7  Generate OpenCV overlay image.",
    )
    mode.add_argument(
        "--headless", action="store_true",
        help="§8  CI/CD rule checks R1–R7.",
    )
    mode.add_argument(
        "--pixel", action="store_true",
        help="§9  Pixel-level ink fidelity test.",
    )
    mode.add_argument(
        "--ocr", action="store_true",
        help="§10 OCR round-trip verifier.",
    )
    mode.add_argument(
        "--math", action="store_true",
        help="§11 Geometric math verifier.",
    )
    mode.add_argument(
        "--fingerprint", action="store_true",
        help="§12 Hallucination fingerprinter.",
    )
    mode.add_argument(
        "--regression", action="store_true",
        help="§13 Full regression suite.",
    )
    mode.add_argument(
        "--selftest", action="store_true",
        help="§14 Run built-in unit tests (no PDF/JSON needed).",
    )

    # Options
    parser.add_argument(
        "--output", type=str, default=OUTPUT_OVERLAY,
        help=f"Overlay image output path (default: {OUTPUT_OVERLAY}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"PDF render DPI — must match Node 01 (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--page", type=int, default=1,
        help="PDF page number to audit — 1-based (default: 1).",
    )
    parser.add_argument(
        "--ink-threshold", type=float, default=DEFAULT_INK_THRESHOLD,
        help=f"§9: Min ink pixel ratio (default: {DEFAULT_INK_THRESHOLD}).",
    )
    parser.add_argument(
        "--save-crops", action="store_true",
        help=f"§9: Save bbox crop images to ./{OUTPUT_CROPS_DIR}/.",
    )
    parser.add_argument(
        "--tesseract", action="store_true",
        help="§10: Enable pytesseract re-OCR (requires pytesseract + Tesseract).",
    )
    parser.add_argument(
        "--pass-max-dist", type=float, default=DEFAULT_PASS_MAX_DIST,
        help=f"§11: PASS link max distance px (default: {DEFAULT_PASS_MAX_DIST}).",
    )
    parser.add_argument(
        "--fail-min-dist", type=float, default=DEFAULT_FAIL_MIN_DIST,
        help=f"§11: FAIL link min distance px (default: {DEFAULT_FAIL_MIN_DIST}).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


def extended_main() -> None:
    """
    §15 — Master CLI entrypoint.

    1. Parse arguments.
    2. Route --selftest immediately (no files needed).
    3. Validate positional args for all other modes.
    4. Build AssetBundle ONCE.
    5. Dispatch to section function.
    """
    parser = build_cli()
    args   = parser.parse_args()

    # Verbose logging
    if args.verbose:
        log.setLevel(logging.DEBUG)
        log.debug("DEBUG logging enabled.")

    log.info("CAD Compliance Audit System  v%s  — Starting", __version__)

    # ── Self-test: no files needed ────────────────────────────────
    if args.selftest:
        run_self_tests()
        return   # sys.exit() called inside

    # ── All other modes require positional args ───────────────────
    if not args.pdf_path or not args.json_path:
        parser.error(
            "pdf_path and json_path are required for all modes except --selftest."
        )

    # ── Apply CLI overrides to module-level constants ─────────────
    global DEFAULT_DPI, OUTPUT_OVERLAY, DEFAULT_PASS_MAX_DIST, DEFAULT_FAIL_MIN_DIST
    DEFAULT_DPI           = args.dpi
    OUTPUT_OVERLAY        = args.output
    DEFAULT_PASS_MAX_DIST = args.pass_max_dist
    DEFAULT_FAIL_MIN_DIST = args.fail_min_dist

    page_index = max(0, args.page - 1)   # Convert 1-based CLI arg to 0-based

    log.info("PDF        : %s", args.pdf_path)
    log.info("JSON       : %s", args.json_path)
    log.info("DPI        : %d", DEFAULT_DPI)
    log.info("Page       : %d  (index %d)", args.page, page_index)

    # ── Build AssetBundle once for all modes ─────────────────────
    bundle = build_asset_bundle(
        pdf_path    = args.pdf_path,
        json_path   = args.json_path,
        dpi         = DEFAULT_DPI,
        page_indices= [page_index],
    )

    # ── Dispatch ──────────────────────────────────────────────────

    if args.visual:
        run_visual_audit(bundle, output_path=OUTPUT_OVERLAY)
        sys.exit(0)

    if args.headless:
        run_headless_audit(bundle)
        # sys.exit() called internally — line below never reached
        return

    if args.pixel:
        result = run_pixel_fidelity_test(
            bundle,
            ink_threshold = args.ink_threshold,
            save_crops    = args.save_crops,
        )
        sys.exit(0 if result["overall"] == "PASS" else 1)

    if args.ocr:
        result = run_ocr_roundtrip_verifier(
            bundle,
            use_tesseract = args.tesseract,
        )
        # GAP-13 fix: WARN exits 0 (informational, non-blocking)
        sys.exit(0 if result["overall"] in ("PASS", "WARN") else 1)

    if args.math:
        result = run_geometric_math_verifier(
            bundle,
            pass_max_distance = DEFAULT_PASS_MAX_DIST,
            fail_min_distance = DEFAULT_FAIL_MIN_DIST,
        )
        sys.exit(0 if result["overall"] == "PASS" else 1)

    if args.fingerprint:
        result = run_hallucination_fingerprinter(bundle)
        sys.exit(0 if result["overall"] == "PASS" else 1)

    if args.regression:
        run_full_regression(
            bundle,
            use_tesseract = args.tesseract,
            save_crops    = args.save_crops,
            ink_threshold = args.ink_threshold,
        )
        # sys.exit() called internally — line below never reached
        return

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    extended_main()
