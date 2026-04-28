"""Constants and configuration for Logical Knowledge Graph.

Centralizes all engineering thresholds, tolerance values, and system parameters
to ensure consistency across all nodes.
"""

# ============================================================================
# SPATIAL CONSTANTS (Context Engineering)
# ============================================================================

# DPI settings for pixel-to-real-world conversion
DPI_STANDARD = 150
DPI_SUPPORTED = [150, 200, 300, 600]
PIXELS_PER_MM = 5.9055  # At 150 DPI

class GlobalCoordinateSync:
    """Utility to enforce 0.0-1.0 coordinate scaling to prevent geometric drift."""
    
    @staticmethod
    def to_global(x: float, y: float, w_px: int, h_px: int) -> tuple[float, float]:
        """Convert pixel coordinates to global 0.0-1.0 scale."""
        return (x / max(1, w_px), y / max(1, h_px))
        
    @staticmethod
    def to_local(gx: float, gy: float, w_px: int, h_px: int) -> tuple[float, float]:
        """Convert global 0.0-1.0 scale to pixel coordinates."""
        return (gx * w_px, gy * h_px)

# Spatial linking threshold (epsilon - ε)
# Maximum distance (global 0.0-1.0 scale) between geometry centroid and text bbox for binding
DISTANCE_THRESHOLD_GLOBAL = 0.05

# Tolerance threshold (tau - τ)
# Allowed variance between documented values and geometry properties
TOLERANCE_THRESHOLD = 0.1  # 10%

# Walker re-scan bounding box expansion (pixels)
WALKER_RESCAN_MARGIN = 200

# ============================================================================
# TOLERANCE THRESHOLDS (Spec-Driven Engineering)
# ============================================================================

# Dimensional tolerance for geometry-text matching
# Acceptable variance between table value and measured geometry
dimensional_tolerance_pct = 2.0  # ±2.0%

# Positional tolerance for alignment checks
positional_tolerance_px = 10.0

# OCR confidence threshold
ocr_confidence_min = 0.7

# ============================================================================
# Epsilon (ε) - Distance Threshold
# ============================================================================
EPSILON = DISTANCE_THRESHOLD_GLOBAL  # 0.05 global scale

# ============================================================================
# Tau (τ) - Engineering Tolerance
# ============================================================================
TAU_DIMENSIONAL = dimensional_tolerance_pct  # 2.0%
TAU_POSITIONAL = positional_tolerance_px    # 10 pixels

# ============================================================================
# SEMANTIC COLLAPSE (Ψ Operator)
# ============================================================================

# Token reduction target via semantic collapse
token_reduction_target_pct = 90  # Reduce tokens by 90%+ before Oracle call

# ============================================================================
# MODEL CONFIGURATION (Neuro Layer)
# ============================================================================

# Segment Anything Model (SAM)
SAM_MODEL_TYPE = "vit_h"  # vit_h, vit_l, vit_b
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

# U-Net (alternative)
UNET_CHANNELS = [3, 64, 128, 256, 512]
UNET_CLASSES = 3  # geometry, text, table

# OCR models
DOCTR_MODEL = "microsoft/table-transformer-detection"
PADDLEOCR_VERSION = "3.0"

# ============================================================================
# OPENCV CONFIGURATION (Symbolic Layer)
# ============================================================================

HOUGH_CIRCLE_PARAM1 = 50
HOUGH_CIRCLE_PARAM2 = 30  # Accumulator threshold (reduced 30% in Walker)
HOUGH_CIRCLE_MIN_RADIUS = 5
HOUGH_CIRCLE_MAX_RADIUS = 200

CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150

# Walker re-scan parameter adjustment
WALKER_PARAM2_REDUCTION = 0.7  # 30% reduction

# ============================================================================
# LLM ORACLE CONFIGURATION (Oracle Layer)
# ============================================================================

# LLM models
LLM_MODEL_CLAUDE = "claude-3-5-sonnet-20241022"
LLM_MODEL_LLAMA = "llama-3-70b"
LLM_MODEL_GEMINI = "gemini-1.5-flash"

# Temperature for compliance reasoning
LLM_TEMPERATURE = 0.0  # Deterministic, no creativity

# Max tokens for compliance report
LLM_MAX_TOKENS = 2000

# ============================================================================
# BUILDING CODE STANDARDS
# ============================================================================

INDIAN_STANDARDS = {
    "IS_456_2000": {
        "name": "IS 456:2000 - Plain and Reinforced Concrete",
        "min_column_size_mm": 225,
        "min_rebar_clear_cover_mm": 25,
        "max_spacing_stirrups_mm": 300,
    },
    "IS_800_2007": {
        "name": "IS 800:2007 - Steel Structures",
        "min_beam_width_mm": 200,
        "max_deflection_ratio": 1/325,
    },
}

# ============================================================================
# VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    "R001": "Geometry must have unique primitive_id",
    "R002": "Coordinates must be within page bounds",
    "R003": "Table cells must have non-empty text",
    "R004": "Hyperedge distance must be <= epsilon",
    "R005": "Variance must be <= tau for PASS status",
    "R006": "Axiom must reference valid hyperedge",
    "R007": "Compliance report must have valid status (PASS/FAIL/PENDING)",
    "R008": "All schemas must validate before node handoff",
}

# ============================================================================
# NODE CONFIGURATION
# ============================================================================

NODE_CONFIG = {
    "node_01_triage": {
        "model": "SAM",
        "output_masks": ["geometry", "text", "table"],
        "min_mask_density": 0.001,  # 0.1%
    },
    "node_02_vectorize": {
        "algorithm": "Hough + Canny",
        "require_unique_ids": True,
        "require_centroids": True,
    },
    "node_03_layout": {
        "model": "DocTR",
        "require_row_integrity": True,
        "column_mapping": ["Mark", "Size", "Reinforcement"],
    },
    "node_04_dhmot": {
        "epsilon": EPSILON,
        "tau": TAU_DIMENSIONAL,
        "walker_rescan_window": WALKER_RESCAN_MARGIN,
        "apply_psi": True,
    },
    "node_05_oracle": {
        "require_axioms": True,
        "rag_enabled": True,
        "token_reduction_target": token_reduction_target_pct,
    },
}

# ============================================================================
# ERROR CODES
# ============================================================================

ERROR_CODES = {
    1000: "VALIDATION_FAILED - Schema validation error",
    1001: "THRESHOLD_EXCEEDED - Distance or variance exceeds tolerance",
    1002: "WALKER_RESCAN_FAILED - Re-scan could not resolve mismatch",
    1003: "MODEL_ERROR - Neural network inference failed",
    1004: "ORACLE_ERROR - LLM API call failed",
    1005: "SCHEMA_MISMATCH - Node output incompatible with next node input",
}

# ============================================================================
# OUTPUT FORMATS
# ============================================================================

DEFAULT_JSON_INDENT = 2
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
REPORT_FORMAT = "json"
