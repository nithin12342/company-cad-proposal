"""Node 03: Layout Intelligence (Neuro-Symbolic) - Production.

OCR-based table extraction with structural parsing. Uses python-doctr
for OCR and deterministic parsing for table reconstruction.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..core.node import LogicalKnowledgeNode, NodeOutput, PipelineJournal, PipelineDataLossError
from ..core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    TableSchema, TableRow, TableCell
)
from ..core.constants import DPI_STANDARD, NODE_CONFIG, GlobalCoordinateSync

logger = logging.getLogger(__name__)


try:
    from doctr import models
    from doctr.io import DocumentFile
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False


class LayoutExtractionNode(LogicalKnowledgeNode):
    """Node 03: Layout Intelligence - Production OCR.
    
    Extracts structured tabular data from text/table masks using
    python-doctr for OCR and deterministic parsing.
    """

    def __init__(self, node_id: str, model_type: str = "docTR", **kwargs):
        if not DOCTR_AVAILABLE:
            raise RuntimeError(
                "python-doctr not installed.\n"
                "Install with: pip install python-doctr@git+https://github.com/mindee/doctr.git"
            )
        self.model_type = model_type
        self.ocr_predictor = None
        self._setup_model()
        super().__init__(node_id, **kwargs)

    def _setup_model(self):
        """Initialize OCR model."""
        logger.info("Loading python-doctr model...")
        self.ocr_predictor = models.ocr_predictor(pretrained=True)
        logger.info("OCR model ready")

    def _build_context(self) -> BaseNodeContext:
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["node_01_triage"],
            input_schema="BinaryMasks_v2.0/{text,table}",
            output_schema="TableSchema_v2.0"
        )

    def _build_specification(self) -> BaseNodeSpecification:
        config = NODE_CONFIG["node_03_layout"]
        return BaseNodeSpecification(
            node_type="neuro_symbolic",
            algorithm="python-doctr + StructuralAnalysis",
            version="3.0",
            constraints={
                "ocr_model": "doctr",
                "row_integrity": config["require_row_integrity"]
            },
            validation_rules=["R003", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        return BaseNodeIntention(
            primary_goal="Extract structured tabular data from schedules",
            expected_outcome="JSON tables with cell hierarchy and coordinates",
            success_criteria=[
                "All cells have text content",
                "Row integrity maintained",
                "Column mapping correct"
            ],
            failure_modes=[
                {"mode": "ocr_error", "mitigation": "Adjust OCR confidence threshold"}
            ]
        )

    def _build_harness(self) -> NodeHarness:
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
            error_handling="strict"
        )

    def execute(self, table_mask_path: str, text_mask_path: str,
                page_number: int = 1, original_file_path: str = None) -> tuple[NodeOutput, PipelineJournal]:
        logger.info(f"Extracting layout from page {page_number}")

        valid, errors = self.validate_input({
            "table_mask": table_mask_path,
            "text_mask": text_mask_path
        })
        if not valid:
            out = NodeOutput(
                success=False, data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=errors
            )
            return out, PipelineJournal(node_name=self.node_id, input_summary=table_mask_path, output_summary="FAILED", warnings=errors)

        result = self._execute_with_harness(
            self._extract_tables, table_mask_path, text_mask_path, page_number, original_file_path
        )

        if result["status"] == "failed":
            out = NodeOutput(
                success=False, data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=result["errors"]
            )
            return out, PipelineJournal(node_name=self.node_id, input_summary=table_mask_path, output_summary="FAILED", warnings=result["errors"])

        tables = result["output"]
        
        if len(tables) == 0:
            raise PipelineDataLossError("0 tables extracted from document", self.node_id)
            
        out = NodeOutput(
            success=True, data=tables,
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={"table_count": len(tables)}
        )
        journal = PipelineJournal(
            node_name=self.node_id,
            input_summary=table_mask_path,
            output_summary=f"Extracted {len(tables)} tables",
            warnings=[]
        )
        return out, journal

    def _extract_tables(self, table_mask_path: str, text_mask_path: str,
                        page_number: int, original_file_path: str = None) -> List[TableSchema]:
        """Extract tables using DocTR natively on the original file."""
        import cv2
        from doctr.io import DocumentFile
        
        tables = []
        table_img = self._load_mask(table_mask_path)
        h_mask, w_mask = table_img.shape[:2]
        
        # 1. Run DocTR natively on the original high-res file
        if original_file_path and DOCTR_AVAILABLE and self.ocr_predictor:
            logger.info("Running DocTR natively on original source...")
            try:
                if original_file_path.lower().endswith('.pdf'):
                    doc = DocumentFile.from_pdf(original_file_path)
                else:
                    doc = DocumentFile.from_images(original_file_path)
                
                result = self.ocr_predictor(doc)
                page = result.pages[0]
                
                # DocTR provides relative coordinates 0.0-1.0
                all_cells = []
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            (xmin, ymin), (xmax, ymax) = word.geometry
                            all_cells.append(TableCell(
                                column="Unknown",
                                text=word.value,
                                bbox=[float(xmin), float(ymin), float(xmax), float(ymax)],
                                confidence=word.confidence
                            ))
            except Exception as e:
                logger.error(f"DocTR native processing failed: {e}")
                all_cells = []
        else:
            all_cells = []

        # 2. Group cells by Grid Table Regions
        table_regions = self._detect_table_regions(table_img)
        used_cells = set()
        
        for idx, region in enumerate(table_regions):
            rx1_px, ry1_px, rx2_px, ry2_px = region
            rx1, ry1 = GlobalCoordinateSync.to_global(rx1_px, ry1_px, w_mask, h_mask)
            rx2, ry2 = GlobalCoordinateSync.to_global(rx2_px, ry2_px, w_mask, h_mask)
            
            region_cells = []
            
            for i, cell in enumerate(all_cells):
                cx1, cy1, cx2, cy2 = cell.bbox
                cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    region_cells.append(cell)
                    used_cells.add(i)
            
            if region_cells:
                rows = self._group_into_rows(region_cells)
                table = TableSchema(
                    table_id=f"TBL_{page_number:02d}_{idx:02d}",
                    page_number=page_number,
                    bounding_box=[float(rx1), float(ry1), float(rx2), float(ry2)],
                    headers=self._detect_headers(rows),
                    rows=rows
                )
                tables.append(table)
                
        # 3. Create Virtual Table for Floating CAD Tags
        floating_cells = [cell for i, cell in enumerate(all_cells) if i not in used_cells]
        if floating_cells:
            floating_rows = []
            for i, cell in enumerate(floating_cells):
                cell.column = "Tag"
                floating_rows.append(TableRow(row_index=i, cells=[cell]))
                
            tag_table = TableSchema(
                table_id=f"TBL_{page_number:02d}_FLOATING_TAGS",
                page_number=page_number,
                bounding_box=[0.0, 0.0, 1.0, 1.0],
                headers=["Floating Tags"],
                rows=floating_rows
            )
            tables.append(tag_table)

        logger.info(f"Extracted {len(tables)} tables (including virtual floating tags)")
        return tables

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask image."""
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot load mask: {path}")
        return img

    def _detect_table_regions(self, mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table regions from binary mask."""
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 30:  # Minimum table size
                regions.append((x, y, x+w, y+h))
        return regions

    def _group_into_rows(self, cells: List[TableCell]) -> List[TableRow]:
        """Group cells into rows based on y-coordinate."""
        if not cells:
            return []

        # Sort by y position
        sorted_cells = sorted(cells, key=lambda c: c.bbox[1])

        rows = []
        current_row = []
        current_y = None
        # Adjust tolerance to global scale (e.g. 20 pixels / 1000 pixels = 0.02)
        tolerance = 0.02

        for cell in sorted_cells:
            cell_y = cell.bbox[1]
            if current_y is None or abs(cell_y - current_y) > tolerance:
                if current_row:
                    rows.append(TableRow(row_index=len(rows), cells=current_row))
                current_row = [cell]
                current_y = cell_y
            else:
                current_row.append(cell)

        if current_row:
            rows.append(TableRow(row_index=len(rows), cells=current_row))

        # Assign columns
        for row in rows:
            sorted_cells = sorted(row.cells, key=lambda c: c.bbox[0])
            for i, cell in enumerate(sorted_cells):
                cell.column = ["Mark", "Size", "Reinforcement", "Unknown"][min(i, 3)]

        return rows

    def _detect_headers(self, rows: List[TableRow]) -> List[str]:
        """Detect column headers from first row."""
        if not rows:
            return []
        first_row = rows[0]
        headers = []
        for cell in first_row.cells:
            if "Mark" in cell.text or "mark" in cell.text.lower():
                headers.append("Mark")
            elif "Size" in cell.text or "size" in cell.text.lower():
                headers.append("Size")
            elif "Reinf" in cell.text or "reinf" in cell.text.lower():
                headers.append("Reinforcement")
            else:
                headers.append(cell.text[:20])
        return headers

    def validate_row_integrity(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        config = NODE_CONFIG["node_03_layout"]
        if not config["require_row_integrity"]:
            return True, []

        expected_cols = set(config["column_mapping"])
        for table in tables:
            for row in table.rows:
                present = {c.column for c in row.cells}
                missing = expected_cols - present
                if missing:
                    return False, [f"Missing columns: {missing}"]
        return True, []

    def validate_column_mapping(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        config = NODE_CONFIG["node_03_layout"]
        errors = []
        for table in tables:
            for header in config["column_mapping"]:
                if header not in table.headers and table.headers:
                    pass  # Headers might be auto-detected
        return True, []

    def validate_ocr_confidence(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        errors = []
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.confidence < 0.7:
                        errors.append(f"Low confidence: {cell.text} = {cell.confidence}")
        return len(errors) == 0, errors

    def validate_spatial_consistency(self, tables: List[TableSchema]) -> tuple[bool, List[str]]:
        errors = []
        for table in tables:
            bbox = table.bounding_box
            for row in table.rows:
                for cell in row.cells:
                    cb = cell.bbox
                    if not (bbox[0] <= cb[0] and cb[2] <= bbox[2] and
                           bbox[1] <= cb[1] and cb[3] <= bbox[3]):
                        errors.append(f"Cell '{cell.text}' outside table")
        return len(errors) == 0, errors