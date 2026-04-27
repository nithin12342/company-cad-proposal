"""Formal JSON Schemas for Logical Knowledge Graph data contracts.

Defines strict type-validated schemas for all node inputs/outputs,
ensuring consistency and traceability across the entire pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import json
from datetime import datetime


# ============================================================================
# BASE SCHEMAS
# ============================================================================

@dataclass
class BaseNodeContext:
    """Base context structure for all LKG nodes.
    
    Attributes:
        node_id: Unique identifier for this node instance
        timestamp: ISO format timestamp of node execution
        dependencies: List of node_ids this node depends on
        input_schema: Reference to expected input schema version
        output_schema: Reference to produced output schema version
    """
    node_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    dependencies: List[str] = field(default_factory=list)
    input_schema: str = "v2.0"
    output_schema: str = "v2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "dependencies": self.dependencies,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        }


@dataclass
class BaseNodeSpecification:
    """Formal specification defining node behavior constraints.
    
    Attributes:
        node_type: Classification (neuro, symbolic, hybrid)
        algorithm: Primary algorithm or model name
        version: Algorithm/model version
        constraints: Dictionary of operational constraints
        validation_rules: List of validation rule identifiers
    """
    node_type: str
    algorithm: str
    version: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "algorithm": self.algorithm,
            "version": self.version,
            "constraints": self.constraints,
            "validation_rules": self.validation_rules
        }


@dataclass
class BaseNodeIntention:
    """Explicit purpose and expected outcome definition.
    
    Attributes:
        primary_goal: High-level objective
        expected_outcome: Concrete expected result
        success_criteria: List of measurable success conditions
        failure_modes: Known failure scenarios and mitigations
    """
    primary_goal: str
    expected_outcome: str
    success_criteria: List[str] = field(default_factory=list)
    failure_modes: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_goal": self.primary_goal,
            "expected_outcome": self.expected_outcome,
            "success_criteria": self.success_criteria,
            "failure_modes": self.failure_modes
        }


@dataclass
class NodeHarness:
    """Execution harness with code, compilation, run, and validation components.
    
    Attributes:
        source_module: Python module path for execution
        entry_function: Function to invoke
        compile_required: Whether compilation/preprocessing is needed
        validation_hooks: List of validation function names
        error_handling: Error handling strategy
    """
    source_module: str
    entry_function: str
    compile_required: bool = False
    validation_hooks: List[str] = field(default_factory=list)
    error_handling: str = "strict"  # strict, lenient, recoverable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_module": self.source_module,
            "entry_function": self.entry_function,
            "compile_required": self.compile_required,
            "validation_hooks": self.validation_hooks,
            "error_handling": self.error_handling
        }


# ============================================================================
# GEOMETRY SCHEMA (Phase 2 - Symbolic Layer)
# ============================================================================

@dataclass
class GeometryPrimitive:
    """Single geometric primitive from boundary representation (B-Rep).
    
    Attributes:
        primitive_id: Unique identifier (e.g., GEO_001)
        primitive_type: Type of geometry (rectangle, circle, line, arc, polyline)
        coordinates: Bounding box or defining coordinates
        centroid: Center point [x, y]
        properties: Metric properties (width, height, area, etc.)
    """
    primitive_id: str
    primitive_type: str
    coordinates: Dict[str, float]
    centroid: Tuple[float, float]
    properties: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "type": self.primitive_type,
            "coordinates": self.coordinates,
            "centroid": list(self.centroid),
            "properties": self.properties
        }

    @staticmethod
    def validate(geom: Dict[str, Any]) -> bool:
        """Validate geometry dictionary against schema."""
        required = ["primitive_id", "type", "coordinates", "centroid"]
        if not all(k in geom for k in required):
            return False
        if geom["type"] not in ["rectangle", "circle", "line", "arc", "polyline"]:
            return False
        return True


@dataclass
class GeometryBRepSchema:
    """Boundary Representation (B-Rep) schema for extracted geometry.
    
    Represents the complete geometric description of a page after
    pixel-to-mathematical conversion in Phase 2.
    
    Attributes:
        page_number: Source page number
        dpi_reference: DPI used for pixel-to-unit conversion
        scale_factor: Pixels per millimeter (for IS code conversions)
        geometries: List of GeometryPrimitive objects
        total_count: Number of primitives
        bounding_box: Overall page bounding box
    """
    page_number: int
    dpi_reference: int = 300
    scale_factor: float = 11.811  # pixels/mm at 300 DPI
    geometries: List[GeometryPrimitive] = field(default_factory=list)
    
    @property
    def total_count(self) -> int:
        return len(self.geometries)
    
    @property
    def bounding_box(self) -> Optional[Dict[str, float]]:
        """Calculate overall bounding box containing all primitives."""
        if not self.geometries:
            return None
        all_x = []
        all_y = []
        for geom in self.geometries:
            coords = geom.coordinates
            if "x1" in coords and "x2" in coords:
                all_x.extend([coords["x1"], coords["x2"]])
                all_y.extend([coords["y1"], coords["y2"]])
            elif "centroid" in coords:
                all_x.append(coords["centroid"][0])
                all_y.append(coords["centroid"][1])
        if not all_x:
            return None
        return {
            "x1": min(all_x), "y1": min(all_y),
            "x2": max(all_x), "y2": max(all_y)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "page_number": self.page_number,
            "dpi_reference": self.dpi_reference,
            "scale_factor": self.scale_factor,
            "geometries": [g.to_dict() for g in self.geometries],
            "total_count": self.total_count
        }
        bbox = self.bounding_box
        if bbox:
            result["bounding_box"] = bbox
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GeometryBRepSchema":
        """Deserialize from dictionary."""
        geometries = []
        for g in data.get("geometries", []):
            geom = GeometryPrimitive(
                primitive_id=g["primitive_id"],
                primitive_type=g["type"],
                coordinates=g["coordinates"],
                centroid=tuple(g["centroid"]),
                properties=g.get("properties", {})
            )
            geometries.append(geom)
        return GeometryBRepSchema(
            page_number=data["page_number"],
            dpi_reference=data.get("dpi_reference", 300),
            scale_factor=data.get("scale_factor", 11.811),
            geometries=geometries
        )

    @staticmethod
    def from_json(json_str: str) -> "GeometryBRepSchema":
        """Deserialize from JSON string."""
        return GeometryBRepSchema.from_dict(json.loads(json_str))

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate schema integrity.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        if self.page_number < 1:
            errors.append(f"Invalid page_number: {self.page_number}")
        if self.dpi_reference not in [150, 200, 300, 600]:
            errors.append(f"Unusual DPI: {self.dpi_reference}")
        for i, geom in enumerate(self.geometries):
            if not geom.primitive_id:
                errors.append(f"Geometry {i} missing primitive_id")
            if not GeometryPrimitive.validate(geom.to_dict()):
                errors.append(f"Geometry {i} ({geom.primitive_id}) invalid")
        return len(errors) == 0, errors


# ============================================================================
# TABLE SCHEMA (Phase 3 - Neuro-Symbolic Layer)
# ============================================================================

@dataclass
class TableCell:
    """Single table cell with OCR-extracted text and location.
    
    Attributes:
        column: Column header/type (e.g., "Mark", "Size", "Reinforcement")
        text: OCR-extracted text content
        bbox: Bounding box [x1, y1, x2, y2] in page coordinates
        confidence: OCR confidence score (0-1)
    """
    column: str
    text: str
    bbox: List[float]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence
        }


@dataclass
class TableRow:
    """Single row in a detected table.
    
    Attributes:
        row_index: Zero-based row index
        cells: List of TableCell objects
    """
    row_index: int
    cells: List[TableCell] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cells": [c.to_dict() for c in self.cells]
        }

    def get_cell_by_column(self, column: str) -> Optional[TableCell]:
        """Find cell by column name."""
        for cell in self.cells:
            if cell.column == column:
                return cell
        return None


@dataclass
class TableSchema:
    """Extracted table with structured rows and cells.
    
    Represents tabular data (e.g., reinforcement schedules, dimension tables)
    extracted via OCR in Phase 3.
    
    Attributes:
        table_id: Unique identifier (e.g., TBL_SCHEDULE_01)
        page_number: Source page number
        bounding_box: Table location [x1, y1, x2, y2]
        headers: Column header names
        rows: List of TableRow objects
    """
    table_id: str
    page_number: int
    bounding_box: List[float]
    headers: List[str] = field(default_factory=list)
    rows: List[TableRow] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "bounding_box": self.bounding_box,
            "headers": self.headers,
            "rows": [r.to_dict() for r in self.rows],
            "row_count": self.row_count
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TableSchema":
        """Deserialize from dictionary."""
        rows = []
        for r in data.get("rows", []):
            cells = []
            for c in r.get("cells", []):
                cell = TableCell(
                    column=c["column"],
                    text=c["text"],
                    bbox=c["bbox"],
                    confidence=c.get("confidence", 1.0)
                )
                cells.append(cell)
            row = TableRow(row_index=r["row_index"], cells=cells)
            rows.append(row)
        return TableSchema(
            table_id=data["table_id"],
            page_number=data["page_number"],
            bounding_box=data["bounding_box"],
            headers=data.get("headers", []),
            rows=rows
        )

    @staticmethod
    def from_json(json_str: str) -> "TableSchema":
        """Deserialize from JSON string."""
        return TableSchema.from_dict(json.loads(json_str))

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate table schema integrity.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        if not self.table_id:
            errors.append("Missing table_id")
        if self.page_number < 1:
            errors.append(f"Invalid page_number: {self.page_number}")
        if len(self.bounding_box) != 4:
            errors.append(f"Invalid bounding_box: {self.bounding_box}")
        for i, row in enumerate(self.rows):
            if row.row_index != i:
                errors.append(f"Row {i} has mismatched index: {row.row_index}")
        return len(errors) == 0, errors


# ============================================================================
# HYPEREDGE SCHEMA (Phase 4 - DHMoT Layer)
# ============================================================================

@dataclass
class HyperedgeBinding:
    """N-ary relational binding in the DHMoT hypergraph.
    
    Links geometric primitives with table data entries based on spatial proximity.
    Forms the core relational structure enabling cross-modal validation.
    
    Attributes:
        hyperedge_id: Unique identifier (e.g., HEDGE_001)
        geometry_id: Referenced geometry primitive ID
        table_id: Referenced table ID
        row_index: Table row index
        column: Column name being linked
        distance: Euclidean distance (pixels) between geometry centroid and text bbox
        within_threshold: Whether distance <= epsilon threshold
    """
    hyperedge_id: str
    geometry_id: str
    table_id: str
    row_index: int
    column: str
    distance: float
    within_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hyperedge_id": self.hyperedge_id,
            "geometry_id": self.geometry_id,
            "table_id": self.table_id,
            "row_index": self.row_index,
            "column": self.column,
            "distance": round(self.distance, 2),
            "within_threshold": self.within_threshold
        }


@dataclass
class ValidationResult:
    """Result of validating a hyperedge's geometry-text pairing.
    
    Attributes:
        hyperedge_id: Referenced hyperedge
        status: PASS, FAIL, or PENDING
        table_value: Value from table (e.g., "400x400")
        geometry_value: Measured value from geometry (e.g., 402.3)
        variance_pct: Percentage difference between values
        within_tolerance: Whether variance <= tolerance threshold
        details: Additional validation details
    """
    hyperedge_id: str
    status: str
    table_value: str
    geometry_value: float
    variance_pct: float
    within_tolerance: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hyperedge_id": self.hyperedge_id,
            "status": self.status,
            "table_value": self.table_value,
            "geometry_value": round(self.geometry_value, 2),
            "variance_pct": round(self.variance_pct, 2),
            "within_tolerance": self.within_tolerance,
            "details": self.details
        }


@dataclass
class AxiomManifest:
    """Collapsed semantic output after Psi operator.
    
    Replaces raw pixel/coordinate data with verified natural language facts.
    Dramatically reduces token count for downstream LLM processing.
    
    Attributes:
        axiom_id: Unique identifier (e.g., AXM_001)
        subject: Component being described (e.g., "Foundation Column C1")
        fact: Natural language fact string
        integrity: MATCHED, MISMATCHED, or UNCERTAIN
        variance_pct: Percentage difference (0.0 if perfect match)
        source_hyperedge: Referenced hyperedge ID
    """
    axiom_id: str
    subject: str
    fact: str
    integrity: str
    variance_pct: float
    source_hyperedge: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axiom_id": self.axiom_id,
            "subject": self.subject,
            "fact": self.fact,
            "integrity": self.integrity,
            "variance_pct": round(self.variance_pct, 2),
            "source_hyperedge": self.source_hyperedge
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AxiomManifest":
        """Deserialize from dictionary."""
        return AxiomManifest(
            axiom_id=data["axiom_id"],
            subject=data["subject"],
            fact=data["fact"],
            integrity=data["integrity"],
            variance_pct=data["variance_pct"],
            source_hyperedge=data["source_hyperedge"]
        )


@dataclass
class ComplianceReport:
    """Final regulatory compliance evaluation (Phase 5 - Oracle Layer).
    
    Output schema for LLM-based compliance checking against IS codes.
    
    Attributes:
        report_id: Unique report identifier
        document_id: Source document reference
        project_standards: List of applicable standards (e.g., ["IS 456:2000"])
        report_summary: Summary statistics
        compliance_details: Per-axiom evaluation details
        generated_at: Timestamp of report generation
    """
    report_id: str
    document_id: str
    project_standards: List[str]
    report_summary: Dict[str, Any]
    compliance_details: List[Dict[str, Any]]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "document_id": self.document_id,
            "project_standards": self.project_standards,
            "report_summary": self.report_summary,
            "compliance_details": self.compliance_details,
            "generated_at": self.generated_at
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
