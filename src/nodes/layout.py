"""Node 03: Layout Intelligence (Neuro-Symbolic Layer).

Third stage of the LKG pipeline. Uses OCR and table recognition to extract
structured data from table masks. Hybrid approach: neural network for text
detection, symbolic parsing for structure.

Engineering Principles:
- Context: Input text/table masks from Node 01
- Specification: DocTR/PaddleOCR, structured JSON schema
- Intention: Reconstruct engineering schedules and relationships
- Harness: Validation via row integrity and column mapping
"""

import numpy as np
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# Mock OCR for demonstration (would import DocTR/PaddleOCR in production)
try:
    # Production would use: from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = False
except ImportError:
    DOCTR_AVAILABLE = False

from ..core.node import LogicalKnowledgeNode, NodeOutput
from ..core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    TableSchema, TableRow, TableCell
)
from ..core.constants import DPI_STANDARD, NODE_CONFIG

logger = logging.getLogger(__name__)


class LayoutExtractionNode(LogicalKnowledgeNode):
    """Node 03: Layout Extraction - Neuro-Symbolic OCR.
    
    Processes text and table masks to extract structured tabular data.
    Uses OCR to read text and structural analysis to reconstruct tables.
    Output is schema-compliant JSON with physical coordinates.
    """

    def __init__(self, node_id: str, model_type: str = "DocTR", **kwargs):
        """Initialize Layout Extraction node.
        
        Args:
            node_id: Unique identifier (e.g., "node_03_layout")
            model_type: "DocTR" or "PaddleOCR"
            **kwargs: Additional configuration
        """
        self.model_type = model_type
        super().__init__(node_id, **kwargs)

    def _build_context(self) -> BaseNodeContext:
        """Build context for layout extraction.
        
        Takes text and table masks from Node 01.
        Produces structured table schemas.
        """
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["node_01_triage"],
            input_schema="BinaryMasks_v2.0/{text,table}",
            output_schema="TableSchema_v2.0"
        )

    def _build_specification(self) -> BaseNodeSpecification:
        """Build formal specification for layout extraction.
        
        DocTR/PaddleOCR for OCR, structured JSON output.
        """
        config = NODE_CONFIG["node_03_layout"]
        return BaseNodeSpecification(
            node_type="neuro_symbolic",
            algorithm=f"{self.model_type} + StructuralAnalysis",
            version="3.0",
            constraints={
                "ocr_model": self.model_type,
                "license": "Apache 2.0",
                "coordinate_system": f"{DPI_STANDARD} DPI",
                "required_columns": config["column_mapping"],
                "row_integrity": config["require_row_integrity"]
            },
            validation_rules=["R003", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        """Build intention for layout extraction.
        
        Goal: Reconstruct engineering schedules (e.g., reinforcement
        schedules) from tabular visual data into queryable structured data.
        """
        return BaseNodeIntention(
            primary_goal="Extract structured tabular data from visual schedules",
            expected_outcome="JSON tables with row/cell hierarchy and spatial coordinates",
            success_criteria=[
                "All table cells have text content",
                "Row integrity maintained (no missing cells)",
                "Column mapping correct (Mark, Size, Reinforcement)",
                "Bounding boxes in page coordinates"
            ],
            failure_modes=[
                {
                    "mode": "ocr_misread",
                    "mitigation": "Walker cross-validation with geometry"
                },
                {
                    "mode": "table_structure_misdetection",
                    "mitigation": "Manual table boundary adjustment"
                },
                {
                    "mode": "low_quality_scan",
                    "mitigation": "Higher DPI rescan or image enhancement"
                }
            ]
        )

    def _build_harness(self) -> NodeHarness:
        """Build execution harness for layout extraction.
        
        Validates OCR output quality and table structure integrity.
        """
        return NodeHarness(
            source_module="src.nodes.layout",
            entry_function="extract_tables",
            compile_required=False,
            validation_hooks=[
                "validate_row_integrity",
                "validate_column_mapping",
                "validate_ocr_confidence",
                "validate_spatial_consistency"
            ],
            error_handling="recoverable"  # Can attempt alternative OCR
        )

    def execute(self, 
                table_mask_path: str,
                text_mask_path: str,
                page_number: int = 1) -> Any:
        """Execute layout extraction on input masks.
        
        Args:
            table_mask_path: Path to table mask from Node 01
            text_mask_path: Path to text mask from Node 01
            page_number: Source page number
            
        Returns:
            NodeOutput with list of TableSchema objects
        """
        logger.info(f"Executing Layout Extraction on page {page_number}")

        # Validate inputs
        valid, errors = self.validate_input({
            "table_mask": table_mask_path,
            "text_mask": text_mask_path
        })
        if not valid:
            return NodeOutput(
                success=False,
                data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=errors
            )

        # Execute with harness guarantees
        harness_result = self._execute_with_harness(
            self._extract_tables,
            table_mask_path,
            text_mask_path,
            page_number
        )

        if harness_result["status"] == "failed":
            return NodeOutput(
                success=False,
                data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=harness_result["errors"]
            )

        table_schemas = harness_result.get("output")

        return NodeOutput(
            success=True,
            data=table_schemas,
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={
                "table_count": len(table_schemas),
                "model_type": self.model_type,
                "execution_trace": harness_result.get("execution_trace", [])
            }
        )

    def _extract_tables(self, 
                        table_mask_path: str,
                        text_mask_path: str,
                        page_number: int) -> List[TableSchema]:
        """Core extraction logic.
        
        Detects table structures and performs OCR to extract cell contents.
        
        Args:
            table_mask_path: Path to table region mask
            text_mask_path: Path to text region mask
            page_number: Page number for metadata
            
        Returns:
            List of TableSchema objects
        """
        if DOCTR_AVAILABLE:
            return self._extract_with_doctr(table_mask_path, text_mask_path, page_number)
        else:
            logger.warning("DocTR not available, using simulated extraction")
            return self._simulate_extraction(page_number)

    def _extract_with_doctr(self, table_mask_path: str, text_mask_path: str, page_number: int):
        """Production OCR extraction using DocTR."""
        # Would load image, run DocTR OCR model, parse table structure
        # This is a placeholder for actual implementation
        return self._simulate_extraction(page_number)

    def _simulate_extraction(self, page_number: int) -> List[TableSchema]:
        """Simulate table extraction (for testing without DocTR)."""
        import random
        
        tables = []
        
        # Simulate a reinforcement schedule table
        cells = []
        marks = ["C1", "C2", "C3", "C4"]
        sizes = ["400x400", "350x350", "300x300", "250x250"]
        rebars = ["8-T16", "6-T16", "4-T16", "8-T12"]
        
        for i, (mark, size, rebar) in enumerate(zip(marks, sizes, rebars)):
            y_base = 100 + i * 50
            cells.append(TableCell(
                column="Mark",
                text=mark,
                bbox=[60, y_base, 100, y_base + 20],
                confidence=random.uniform(0.85, 0.99)
            ))
            cells.append(TableCell(
                column="Size",
                text=size,
                bbox=[110, y_base, 200, y_base + 20],
                confidence=random.uniform(0.85, 0.99)
            ))
            cells.append(TableCell(
                column="Reinforcement",
                text=rebar,
                bbox=[210, y_base, 350, y_base + 20],
                confidence=random.uniform(0.85, 0.99)
            ))
        
        rows = []
        for i in range(len(marks)):
            row_cells = cells[i*3:(i+1)*3]
            rows.append(TableRow(row_index=i, cells=row_cells))
        
        table = TableSchema(
            table_id="TBL_REINFORCEMENT_01",
            page_number=page_number,
            bounding_box=[50, 80, 400, 300],
            headers=["Mark", "Size", "Reinforcement"],
            rows=rows
        )
        tables.append(table)
        
        return tables

    def validate_row_integrity(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        """Validation hook: Check row integrity (no missing cells)."""
        config = NODE_CONFIG["node_03_layout"]
        if not config["require_row_integrity"]:
            return True, []
        
        errors = []
        expected_cols = set(config["column_mapping"])
        
        for table in tables:
            for row in table.rows:
                present_cols = {cell.column for cell in row.cells}
                missing = expected_cols - present_cols
                if missing:
                    errors.append(
                        f"Table {table.table_id} row {row.row_index} "
                        f"missing columns: {missing}"
                    )
        
        return len(errors) == 0, errors

    def validate_column_mapping(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        """Validation hook: Check column mapping is correct."""
        config = NODE_CONFIG["node_03_layout"]
        errors = []
        
        for table in tables:
            for header in config["column_mapping"]:
                if header not in table.headers:
                    errors.append(
                        f"Table {table.table_id} missing expected column: {header}"
                    )
        
        return len(errors) == 0, errors

    def validate_ocr_confidence(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        """Validation hook: Check OCR confidence thresholds."""
        errors = []
        
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.confidence < 0.7:
                        errors.append(
                            f"Table {table.table_id} cell '{cell.text}' "
                            f"low confidence: {cell.confidence}"
                        )
        
        return len(errors) == 0, errors

    def validate_spatial_consistency(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        """Validation hook: Check spatial consistency of table elements."""
        errors = []
        
        for table in tables:
            bbox = table.bounding_box
            for row in table.rows:
                for cell in row.cells:
                    cell_bbox = cell.bbox
                    # Check cell is within table bounds
                    if not (bbox[0] <= cell_bbox[0] and cell_bbox[2] <= bbox[2] and
                            bbox[1] <= cell_bbox[1] and cell_bbox[3] <= bbox[3]):
                        errors.append(
                            f"Cell '{cell.text}' outside table bounds"
                        )
        
        return len(errors) == 0, errors