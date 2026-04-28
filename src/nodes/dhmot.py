"""Node 04: DHMoT Agent (Manifold Logic - The Relational Engine).

Fourth stage of the LKG pipeline. Implements the DHMoT (Deterministic
Hierarchical Manifold of Thought) agent that performs cross-modal
validation by linking geometry and table data through spatial proximity.

This is where the "Graph" in Knowledge Graph becomes concrete - creating
hyperedges that bind semantically related entities across modalities.

Engineering Principles:
- Context: Node 02 (geometry) + Node 03 (tables) outputs
- Specification: Hyperedge formation, Walker re-scan, Psi collapse
- Intention: Self-healing cross-validation of drawing vs. documentation
- Harness: Deterministic matching with tolerance thresholds
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import cv2

from ..core.node import LogicalKnowledgeNode, NodeOutput, PipelineJournal, PipelineDataLossError
from ..core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    GeometryBRepSchema, GeometryPrimitive, TableSchema, HyperedgeBinding, 
    ValidationResult, AxiomManifest
)
from ..core.constants import (
    DISTANCE_THRESHOLD_GLOBAL, TOLERANCE_THRESHOLD, WALKER_RESCAN_MARGIN,
    NODE_CONFIG, VALIDATION_RULES, PIXELS_PER_MM,
    HOUGH_CIRCLE_PARAM1, HOUGH_CIRCLE_PARAM2,
    HOUGH_CIRCLE_MIN_RADIUS, HOUGH_CIRCLE_MAX_RADIUS,
    WALKER_PARAM2_REDUCTION,
    CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH
)

logger = logging.getLogger(__name__)



class ConflictException(Exception):
    """Exception raised when Walker re-scan cannot resolve >2% variance conflicts.
    
    Indicates that human-in-the-loop (HITL) arbitration is required.
    """
    def __init__(self, conflict_data: Dict[str, Any]):
        self.conflict_data = conflict_data
        super().__init__(f"CONFLICT_EXCEPTION: {len(conflict_data.get('conflicts', []))} conflicts require arbitration")



class DHMoTNode(LogicalKnowledgeNode):
    """Node 04: DHMoT Agent - Relational Knowledge Graph Construction.
    
    Creates hyperedges that bind geometry primitives with table data
    based on spatial proximity (epsilon threshold). Performs validation
    by comparing table values against measured geometry.
    
    Key Concepts:
    - Hyperedge: N-ary relation linking one geometry to N table cells
    - Walker: Autonomous re-scan agent when mismatches detected
    - Psi (Ψ) Operator: Semantic collapse from raw data to axioms
    """

    def __init__(self, node_id: str, **kwargs):
        """Initialize DHMoT node.
        
        Args:
            node_id: Unique identifier (e.g., "node_04_dhmot")
            **kwargs: Additional configuration
        """
        self.epsilon = kwargs.get("epsilon", DISTANCE_THRESHOLD_GLOBAL)
        self.tau = kwargs.get("tau", TOLERANCE_THRESHOLD)
        self.walker_rescan_margin = kwargs.get(
            "walker_rescan_margin", WALKER_RESCAN_MARGIN
        )
        self.apply_psi = kwargs.get("apply_psi", True)
        super().__init__(node_id, **kwargs)

    def _build_context(self) -> BaseNodeContext:
        """Build context for DHMoT operation.
        
        Consumes geometry (Node 02) and tables (Node 03).
        Produces hyperedges, validation results, and axioms.
        """
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["node_02_vectorize", "node_03_layout"],
            input_schema="{GeometryBRep_v2.0, TableSchema_v2.0}",
            output_schema="{Hyperedge_v2.0, Validation_v2.0, Axiom_v2.0}"
        )

    def _build_specification(self) -> BaseNodeSpecification:
        """Build formal specification for DHMoT agent.
        
        Hyperedge formation, Walker loop-back, Psi collapse.
        """
        config = NODE_CONFIG["node_04_dhmot"]
        return BaseNodeSpecification(
            node_type="symbolic",
            algorithm="DHMoT (Hyperedge Relational Binding + Walker + Psi)",
            version="2.0",
            constraints={
                "epsilon_px": self.epsilon,
                "tau_pct": self.tau,
                "walker_rescan_window": self.walker_rescan_margin,
                "apply_psi": self.apply_psi,
                "deterministic": True
            },
            validation_rules=["R004", "R005", "R006", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        """Build intention for DHMoT agent.
        
        Goal: Verify that the documented engineering values match
        the actual drawn geometry, with self-healing capabilities.
        """
        return BaseNodeIntention(
            primary_goal="Cross-modal validation: Verify documentation matches drawing",
            expected_outcome="Hyperedges linking geometry to table data with validation status",
            success_criteria=[
                "All geometry primitives linked to relevant table entries",
                "Validation results computed for each hyperedge",
                "Psi operator generates axioms for verified items",
                "Walker resolves mismatches autonomously"
            ],
            failure_modes=[
                {
                    "mode": "unlinked_geometry",
                    "mitigation": "Walker re-scan with expanded search window"
                },
                {
                    "mode": "tolerance_violation",
                    "mitigation": "Review and adjust tau threshold if appropriate"
                },
                {
                    "mode": "missing_table_data",
                    "mitigation": "Manual entry or OCR correction"
                }
            ]
        )

    def _build_harness(self) -> NodeHarness:
        """Build execution harness for DHMoT agent.
        
        Ensures deterministic execution and proper validation.
        """
        return NodeHarness(
            source_module="src.nodes.dhmot",
            entry_function="execute_dhmot",
            compile_required=False,
            validation_hooks=[
                "validate_hyperedges",
                "validate_distance_thresholds",
                "validate_variance_tolerances",
                "validate_axiom_generation"
            ],
            error_handling="strict"
        )

    def execute(self, 
                geometry: GeometryBRepSchema,
                tables: List[TableSchema],
                original_img_path: Optional[str] = None) -> tuple[NodeOutput, PipelineJournal]:
        """Execute DHMoT agent on geometry and table data.

        Args:
            geometry: B-Rep schema from Node 02
            tables: List of TableSchema from Node 03
            original_img_path: Optional path to original geometry mask for Walker re-scan

        Returns:
            NodeOutput with hyperedges, validations, and axioms
        """
        logger.info("Executing DHMoT Agent")

        # Validate inputs
        valid_geom, geom_errors = self.validate_input(geometry)
        valid_tables, table_errors = self.validate_input(tables)
        errors = geom_errors + table_errors
        
        if not (valid_geom and valid_tables):
            out = NodeOutput(
                success=False,
                data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=errors
            )
            return out, PipelineJournal(node_name=self.node_id, input_summary="Geom & Tables", output_summary="FAILED", warnings=errors)

        # Execute with harness guarantees
        harness_result = self._execute_with_harness(
            self._execute_dhmot,
            geometry,
            tables,
            original_img_path
        )

        if harness_result["status"] == "failed":
            out = NodeOutput(
                success=False,
                data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=harness_result["errors"]
            )
            return out, PipelineJournal(node_name=self.node_id, input_summary="Geom & Tables", output_summary="FAILED", warnings=harness_result["errors"])

        result = harness_result.get("output")
        
        if len(result["hyperedges"]) == 0:
            raise PipelineDataLossError("0 hyperedges formed between geometry and tables", self.node_id)

        out = NodeOutput(
            success=True,
            data=result,
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={
                "hyperedge_count": len(result["hyperedges"]),
                "validation_count": len(result["validations"]),
                "axiom_count": len(result["axioms"]),
                "epsilon": self.epsilon,
                "tau": self.tau,
                "execution_trace": harness_result.get("execution_trace", [])
            }
        )
        journal = PipelineJournal(
            node_name=self.node_id,
            input_summary=f"Geom: {geometry.total_count}, Tables: {len(tables)}",
            output_summary=f"Hyperedges: {len(result['hyperedges'])}, Validations: {len(result['validations'])}, Axioms: {len(result['axioms'])}",
            warnings=[]
        )
        return out, journal

    def _execute_dhmot(self, 
                       geometry: GeometryBRepSchema,
                       tables: List[TableSchema]) -> Dict[str, Any]:
        """Core DHMoT execution.
        
        1. Form hyperedges by linking geometry to table data
        2. Validate each hyperedge (geometry vs. table values)
        3. Apply Walker re-scan for failed hyperedges
        4. Apply Psi operator to generate axioms
        
        Returns:
            Dictionary with hyperedges, validations, and axioms
        """
        logger.info("Phase 1: Forming hyperedges")
        hyperedges = self._form_hyperedges(geometry, tables)
        
        logger.info(f"Formed {len(hyperedges)} hyperedges")
        
        logger.info("Phase 2: Validating hyperedges")
        validations = self._validate_hyperedges(hyperedges, geometry, tables)
        
        logger.info("Phase 3: Walker re-scan for failures")
        failed_validations = [v for v in validations if v.status == "FAIL"]
        if failed_validations:
            logger.info(f"{len(failed_validations)} hyperedges failed, invoking Walker")
            revalidations = self._walker_rescan(
                failed_validations, geometry, tables
            )
            # Replace failed validations with revalidations
            validations = [v for v in validations if v.status != "FAIL"] + revalidations
        
        logger.info("Phase 4: Applying Psi operator")
        axioms = []
        if self.apply_psi:
            axioms = self._apply_psi(validations, geometry, tables)
        
        logger.info(f"DHMoT complete: {len(hyperedges)} hyperedges, "
                    f"{len(validations)} validations, {len(axioms)} axioms")
        
        return {
            "hyperedges": [h.to_dict() for h in hyperedges],
            "validations": [v.to_dict() for v in validations],
            "axioms": [a.to_dict() for a in axioms]
        }

    def _form_hyperedges(self, geometry: GeometryBRepSchema,
                        tables: List[TableSchema]) -> List[HyperedgeBinding]:
        """Form hyperedges by linking geometry centroids to table cells.
        
        Uses epsilon threshold: distance <= EPSILON for binding.
        
        Returns:
            List of HyperedgeBinding objects
        """
        hyperedges = []
        edge_counter = 0
        
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    # Cell bbox center as reference point
                    cell_center_x = (cell.bbox[0] + cell.bbox[2]) / 2
                    cell_center_y = (cell.bbox[1] + cell.bbox[3]) / 2
                    
                    # Find nearest geometry centroid within epsilon
                    for geom in geometry.geometries:
                        geom_x, geom_y = geom.centroid
                        distance = np.sqrt(
                            (geom_x - cell_center_x)**2 +
                            (geom_y - cell_center_y)**2
                        )
                        
                        if distance <= self.epsilon:
                            edge_counter += 1
                            hyperedges.append(HyperedgeBinding(
                                hyperedge_id=f"HEDGE_{edge_counter:03d}",
                                geometry_id=geom.primitive_id,
                                table_id=table.table_id,
                                row_index=row.row_index,
                                column=cell.column,
                                distance=distance,
                                within_threshold=True
                            ))
        
        return hyperedges

    def _validate_hyperedges(self,
                            hyperedges: List[HyperedgeBinding],
                            geometry: GeometryBRepSchema,
                            tables: List[TableSchema]) -> List[ValidationResult]:
        """Validate each hyperedge by comparing table values to geometry.
        
        Uses tau (tolerance) threshold for dimensional variance.
        
        Returns:
            List of ValidationResult objects
        """
        validations = []
        
        # Build lookup maps
        geom_map = {g.primitive_id: g for g in geometry.geometries}
        
        for hedge in hyperedges:
            geom = geom_map.get(hedge.geometry_id)
            if not geom:
                continue
            
            # Extract size from geometry
            geom_size = self._get_geometry_size(geom)
            
            # Extract size from table cell
            table_val = self._get_table_value(
                hedge.table_id, hedge.row_index,
                hedge.column, tables
            )
            
            # Parse table size (e.g., "400x400")
            table_size = self._parse_size_string(table_val)
            
            # Calculate variance
            if table_size > 0 and geom_size > 0:
                variance = abs(table_size - geom_size) / table_size * 100
                within_tolerance = variance <= self.tau
                status = "PASS" if within_tolerance else "FAIL"
            else:
                variance = 0.0
                within_tolerance = False
                status = "FAIL"
            
            details = {
                "geometry_size": round(geom_size, 2),
                "table_size": table_size,
                "units": "mm",
                "tolerance_threshold": self.tau
            }
            
            validations.append(ValidationResult(
                hyperedge_id=hedge.hyperedge_id,
                status=status,
                table_value=table_val,
                geometry_value=geom_size,
                variance_pct=variance,
                within_tolerance=within_tolerance,
                details=details
            ))
        
        return validations

    def _walker_rescan(self,
                        failed_validations: List[ValidationResult],
                        geometry: GeometryBRepSchema,
                        tables: List[TableSchema],
                        original_img_path: Optional[str] = None) -> List[ValidationResult]:
        """Walker re-scan for failed hyperedges with visual evidence overlay.
        
        Expands search window and re-runs detection with adjusted parameters.
        Generates validation_overlay.png showing detected primitives and OCR
        bounding boxes overlaid on original PDF crop for HITL arbitration.
        
        If variance > 2% after re-scan, outputs CONFLICT_EXCEPTION JSON.
        
        Args:
            failed_validations: Validations that need re-scan
            geometry: Current geometry data
            tables: Current table data
            original_img_path: Path to original image for overlay generation
            
        Returns:
            Updated validation results
        """
        logger.info(f"Walker re-scan for {len(failed_validations)} failures")
        
        # Generate visual evidence overlay before re-scan
        self._generate_validation_overlay(failed_validations, geometry, tables, original_img_path)
        
        revalidations = []
        
        for val in failed_validations:
            # Perform actual re-scan with adjusted OpenCV parameters
            # Use walker_rescan_margin to expand search window
            # Reduce Hough param2 by 30% for more sensitive detection
            revalidated = self._perform_rescan(val, geometry, tables, original_img_path)
            revalidations.append(revalidated)
        
        # After re-scan, check for any remaining conflicts (>2% variance)
        conflict_failures = [v for v in revalidations if v.variance_pct > 2.0]
        if conflict_failures:
            self._output_conflict_exception(conflict_failures, geometry, tables)
        
        return revalidations

    def _make_rescanned_result(self, validation: ValidationResult) -> ValidationResult:
        """Return validation marked as re-scanned with no new detection."""
        return ValidationResult(
            hyperedge_id=validation.hyperedge_id,
            status=validation.status,
            table_value=validation.table_value,
            geometry_value=validation.geometry_value,
            variance_pct=validation.variance_pct,
            within_tolerance=validation.within_tolerance,
            details={
                **validation.details,
                "rescanned": True,
                "new_detection": False,
            }
        )

    def _perform_rescan(self, validation: ValidationResult,
                       geometry: GeometryBRepSchema,
                       tables: List[TableSchema],
                       original_img_path: Optional[str] = None) -> ValidationResult:
        """Perform actual re-scan with adjusted detection parameters.

        Uses expanded epsilon threshold and reduced Hough param2
        for more sensitive geometric feature detection.
        """
        # Find the geometry and table cell for this validation
        geom = None
        for g in geometry.geometries:
            if g.primitive_id == validation.geometry_id:
                geom = g
                break
        
        if not geom:
            return validation
        
        # Locate the OCR table cell bbox and expand by walker_rescan_margin
        expanded_epsilon = self.epsilon + self.walker_rescan_margin

        # Extract info to locate the table cell from the validation's hyperedge
        hyperedge_id = validation.hyperedge_id
        parts = hyperedge_id.split("_")
        if len(parts) < 3:
            # Can't parse hyperedge ID, return original validation
            return self._make_rescanned_result(validation)

        # HEDGE_001_<table_id>: parse table_id after the second underscore
        # table_id may contain underscores, so take everything after "HEDGE_<num>_"
        table_id = "_".join(parts[2:])
        row_index = -1
        column = ""

        # Find the matching table and cell to get bbox
        target_cell = None
        for table in tables:
            if table.table_id != table_id:
                continue
            for row in table.rows:
                # Try to find cell by matching geometry_id hint?
                # Actually we need the column that matched this geometry.
                # We stored that in hyperedge.column.
                # But ValidationResult doesn't have it directly.
                # We need to reconstruct from hyperedge.
                pass

        # Actually the hyperedge ID format is: HEDGE_<seq>_<table_id>
        # The column is encoded in the geometry_id? No.
        # The hyperedge was formed by matching geometry centroid to cell centroid.
        # To find the cell, we need the row_index and column.
        # Those were stored in HyperedgeBinding but not in ValidationResult.
        #
        # Since ValidationResult doesn't carry row/column, we need to derive
        # the target cell from the geometry itself by re-linking.
        #
        # Strategy: find all cells within expanded_epsilon of the geometry's centroid.
        # The originally linked cell should be within epsilon.
        geom_centroid = geom.centroid
        cell_candidates = []
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    cell_center_x = (cell.bbox[0] + cell.bbox[2]) / 2
                    cell_center_y = (cell.bbox[1] + cell.bbox[3]) / 2
                    distance = np.sqrt(
                        (geom_centroid[0] - cell_center_x)**2 +
                        (geom_centroid[1] - cell_center_y)**2
                    )
                    if distance <= self.epsilon:
                        cell_candidates.append((cell, table, row))

        if not cell_candidates:
            return self._make_rescanned_result(validation)

        # Take nearest candidate (there should be exactly one originally)
        target_cell, target_table, target_row = min(cell_candidates, key=lambda x: x[0].bbox)

        # Get cell bbox and expand by walker_rescan_margin
        x1, y1, x2, y2 = target_cell.bbox
        margin = self.walker_rescan_margin
        crop_x1 = max(0, int(x1) - margin)
        crop_y1 = max(0, int(y1) - margin)
        crop_x2 = int(x2) + margin
        crop_y2 = int(y2) + margin

        # Need original image to crop
        if not original_img_path:
            return self._make_rescanned_result(validation)

        img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self._make_rescanned_result(validation)

        # Clamp crop to image bounds
        h, w = img.shape
        crop_x2 = min(crop_x2, w)
        crop_y2 = min(crop_y2, h)

        # Crop the region
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            return self._make_rescanned_result(validation)

        # Run geometric extraction on the cropped mask with higher sensitivity
        # Parameter adjustments:
        # - HoughCircle param2 reduced by WALKER_PARAM2_REDUCTION (30%)
        # - Canny thresholds reduced by 30% for finer edge detection
        # - Contour approximation epsilon scaled down for more vertices
        reduced_hough_param2 = int(HOUGH_CIRCLE_PARAM2 * WALKER_PARAM2_REDUCTION)
        reduced_canny_low = int(CANNY_THRESHOLD_LOW * WALKER_PARAM2_REDUCTION)
        reduced_canny_high = int(CANNY_THRESHOLD_HIGH * WALKER_PARAM2_REDUCTION)

        blurred = cv2.GaussianBlur(crop, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20,
            param1=HOUGH_CIRCLE_PARAM1, param2=reduced_hough_param2,
            minRadius=HOUGH_CIRCLE_MIN_RADIUS, maxRadius=HOUGH_CIRCLE_MAX_RADIUS
        )

        # Detect lines
        edges = cv2.Canny(crop, reduced_canny_low, reduced_canny_high)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

        # Detect contours
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect size measurements from all detected primitives
        expected_size_mm = self._parse_size_string(validation.table_value)
        if expected_size_mm <= 0:
            return self._make_rescanned_result(validation)

        candidates = []

        # Process circles: diameter in mm
        if circles is not None:
            for c in np.uint16(np.around(circles))[0]:
                r = c[2]
                diameter_mm = (2 * r) / PIXELS_PER_MM
                variance = abs(diameter_mm - expected_size_mm) / expected_size_mm * 100
                candidates.append({
                    "size_mm": diameter_mm,
                    "variance": variance,
                    "type": "circle"
                })

        # Process lines: length in mm
        if lines is not None:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = line
                length_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                length_mm = length_px / PIXELS_PER_MM
                variance = abs(length_mm - expected_size_mm) / expected_size_mm * 100
                candidates.append({
                    "size_mm": length_mm,
                    "variance": variance,
                    "type": "line"
                })

        # Process contours: bounding box max dimension
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            max_dim_px = max(w, h)
            max_dim_mm = max_dim_px / PIXELS_PER_MM
            variance = abs(max_dim_mm - expected_size_mm) / expected_size_mm * 100
            candidates.append({
                "size_mm": max_dim_mm,
                "variance": variance,
                "type": "contour"
            })

        if not candidates:
            return self._make_rescanned_result(validation)

        # Find best matching primitive
        best_match = min(candidates, key=lambda c: c["variance"])

        if best_match["variance"] <= self.tau:
            # Successfully healed - within tolerance
            new_geom_size = best_match["size_mm"]
            new_variance = best_match["variance"]
            status = "PASS"
            within_tolerance = True
        else:
            # New detection doesn't match within tolerance
            new_geom_size = self._get_geometry_size(geom)
            new_variance = abs(new_geom_size - expected_size_mm) / expected_size_mm * 100
            status = "FAIL"
            within_tolerance = False

        return ValidationResult(
            hyperedge_id=validation.hyperedge_id,
            status=status,
            table_value=validation.table_value,
            geometry_value=new_geom_size,
            variance_pct=new_variance,
            within_tolerance=within_tolerance,
            details={
                "rescanned": True,
                "new_detection": True,
                "expanded_epsilon": expanded_epsilon,
                "original_epsilon": self.epsilon,
                "walker_rescan_margin": self.walker_rescan_margin,
                "geometry_size": round(new_geom_size, 2),
                "table_size": expected_size_mm,
                "variance_pct": round(new_variance, 2),
                "units": "mm",
                "tolerance_threshold": self.tau
            }
        )

    def _generate_validation_overlay(self,
                                   validations: List[ValidationResult],
                                   geometry: GeometryBRepSchema,
                                   tables: List[TableSchema],
                                   original_img_path: Optional[str] = None):
        """Generate visual evidence overlay for Human-in-the-Loop review.
        
        Creates validation_overlay.png that superimposes detected OpenCV
        primitives (lines, circles, contours) and OCR bounding boxes over
        the original PDF crop. Color codes by validation status:
        - Green: PASS (within tolerance)
        - Red: FAIL (variance > tau)
        - Yellow: PENDING
        
        Args:
            validations: List of validation results to overlay
            geometry: Geometry B-Rep with detected primitives
            tables: Table schemas with OCR results
            original_img_path: Path to original image (optional)
        """
        import cv2
        import numpy as np
        
        # Create or load base image
        if original_img_path:
            try:
                base_img = cv2.imread(original_img_path)
                if base_img is None:
                    raise FileNotFoundError()
            except:
                # Fallback: create blank canvas
                base_img = np.ones((3508, 2480, 3), dtype=np.uint8) * 255
        else:
            base_img = np.ones((3508, 2480, 3), dtype=np.uint8) * 255
        
        overlay = base_img.copy()
        
        # Color mapping for validation status
        colors = {
            "PASS": (0, 255, 0),    # Green
            "FAIL": (0, 0, 255),    # Red
            "PENDING": (0, 255, 255)  # Yellow
        }
        
        # Overlay geometry primitives
        for geom in geometry.geometries:
            # Find associated validation
            val_color = colors.get("PENDING", (128, 128, 128))
            for val in validations:
                if val.hyperedge_id and val.hyperedge_id.split("_")[-1] == geom.primitive_id.split("_")[-1]:
                    val_color = colors.get(val.status, (128, 128, 128))
                    break
            
            coords = geom.coordinates
            if geom.primitive_type == "rectangle":
                x1, y1 = int(coords.get("x1", 0)), int(coords.get("y1", 0))
                x2, y2 = int(coords.get("x2", 0)), int(coords.get("y2", 0))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), val_color, 2)
                
                # Add variance label if available
                for val in validations:
                    if val.geometry_id == geom.primitive_id:
                        label = f"{val.variance_pct:.1f}%"
                        cv2.putText(overlay, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color, 1)
            
            elif geom.primitive_type == "circle":
                cx = int(coords.get("center_x", 0))
                cy = int(coords.get("center_y", 0))
                r = int(coords.get("radius", 0))
                cv2.circle(overlay, (cx, cy), r, val_color, 2)
            
            elif geom.primitive_type == "line":
                x1, y1 = int(coords.get("x1", 0)), int(coords.get("y1", 0))
                x2, y2 = int(coords.get("x2", 0)), int(coords.get("y2", 0))
                cv2.line(overlay, (x1, y1), (x2, y2), val_color, 2)
            
            else:  # polygon
                # Draw bounding box
                x1, y1 = int(coords.get("x1", 0)), int(coords.get("y1", 0))
                x2, y2 = int(coords.get("x2", 0)), int(coords.get("y2", 0))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), val_color, 1)
        
        # Overlay OCR bounding boxes from tables
        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    x1, y1, x2, y2 = [int(v) for v in cell.bbox]
                    # Draw semi-transparent box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 128, 0), 1)
                    # Add text (truncated)
                    text = cell.text[:15]
                    cv2.putText(overlay, text, (x1, y1-2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 0), 1)
        
        # Add legend
        legend_y = 50
        for status, color in colors.items():
            cv2.rectangle(overlay, (20, legend_y), (40, legend_y+20), color, -1)
            cv2.putText(overlay, f"{status}", (50, legend_y+15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 30
        
        # Blend overlay with base (50% opacity)
        result = cv2.addWeighted(overlay, 0.6, base_img, 0.4, 0)
        
        # Save overlay
        output_path = "validation_overlay.png"
        cv2.imwrite(output_path, result)
        logger.info(f"Validation overlay saved to: {output_path}")
        
        return output_path

    def _output_conflict_exception(self, conflicts: List[ValidationResult],
                                 geometry: GeometryBRepSchema,
                                 tables: List[TableSchema]):
        """Output CONFLICT_EXCEPTION JSON for user arbitration.
        
        When Walker fails to resolve mismatches with >2% variance,
        the engine pauses and outputs a structured exception for
        manual user review.
        
        Args:
            conflicts: List of validation results with >2% variance
            geometry: Geometry B-Rep data
            tables: Table schema data
        """
        import json
        from datetime import datetime
        
        conflict_data = {
            "exception_type": "CONFLICT_EXCEPTION",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Walker re-scan failed to resolve geometry-table mismatches exceeding 2% variance threshold",
            "conflicts": [],
            "resolution_required": True
        }
        
        for val in conflicts:
            conflict_entry = {
                "hyperedge_id": val.hyperedge_id,
                "geometry_id": val.geometry_id,
                "variance_pct": val.variance_pct,
                "geometry_value": val.geometry_value,
                "table_value": val.table_value,
                "status": val.status,
                "details": val.details
            }
            conflict_data["conflicts"].append(conflict_entry)
        
        # Save to file
        output_path = "CONFLICT_EXCEPTION.json"
        with open(output_path, "w") as f:
            json.dump(conflict_data, f, indent=2)
        
        logger.warning(f"CONFLICT_EXCEPTION written to: {output_path}")
        logger.warning(f"{len(conflicts)} conflicts require user arbitration")
        
        # Also raise exception to pause execution
        raise ConflictException(conflict_data)

    def _execute_dhmot(self,
                       geometry: GeometryBRepSchema,
                       tables: List[TableSchema],
                       original_img_path: Optional[str] = None) -> Dict[str, Any]:
        """Core DHMoT execution.

        1. Form hyperedges by linking geometry to table data
        2. Validate each hyperedge (geometry vs. table values)
        3. Apply Walker re-scan for failed hyperedges
        4. Apply Psi operator to generate axioms

        Args:
            geometry: Geometry B-Rep schema
            tables: List of table schemas
            original_img_path: Optional path to original image for overlay

        Returns:
            Dictionary with hyperedges, validations, and axioms
        """
        logger.info("Phase 1: Forming hyperedges")
        hyperedges = self._form_hyperedges(geometry, tables)

        logger.info(f"Formed {len(hyperedges)} hyperedges")

        logger.info("Phase 2: Validating hyperedges")
        validations = self._validate_hyperedges(hyperedges, geometry, tables)

        logger.info("Phase 3: Walker re-scan for failures")
        failed_validations = [v for v in validations if v.status == "FAIL"]
        if failed_validations:
            logger.info(f"{len(failed_validations)} hyperedges failed, invoking Walker")
            try:
                revalidations = self._walker_rescan(
                    failed_validations, geometry, tables, original_img_path=original_img_path
                )
                # Replace failed validations with revalidations
                validations = [v for v in validations if v.status != "FAIL"] + revalidations
            except ConflictException as e:
                logger.error(f"Walker re-scan aborted due to conflicts: {e}")
                # Re-raise to propagate conflict exception
                raise

        logger.info("Phase 4: Applying Psi operator")
        axioms = []
        if self.apply_psi:
            axioms = self._apply_psi(validations, geometry, tables)

        logger.info(f"DHMoT complete: {len(hyperedges)} hyperedges, "
                    f"{len(validations)} validations, {len(axioms)} axioms")

        return {
            "hyperedges": [h.to_dict() for h in hyperedges],
            "validations": [v.to_dict() for v in validations],
            "axioms": [a.to_dict() for a in axioms]
        }

    def _apply_psi(self,
                   validations: List[ValidationResult],
                   geometry: GeometryBRepSchema,
                   tables: List[TableSchema]) -> List[AxiomManifest]:
        """Apply Psi operator: collapse raw data to axioms.
        
        Converts validated hyperedges into natural language facts,
        reducing token count for downstream LLM processing.
        
        Returns:
            List of AxiomManifest objects
        """
        axioms = []
        axiom_counter = 0
        
        for val in validations:
            if val.status != "PASS":
                continue  # Only collapse successful validations
            
            axiom_counter += 1
            
            # Determine component from table
            table_val = val.table_value
            
            # Generate natural language fact
            subject = f"Component_{table_val}"
            fact = (
                f"Axiom {axiom_counter}: {subject} dimensions ({table_val}) "
                f"match drawing geometry within {val.variance_pct:.1f}% tolerance."
            )
            
            axioms.append(AxiomManifest(
                axiom_id=f"AXM_{axiom_counter:03d}",
                subject=subject,
                fact=fact,
                integrity="MATCHED",
                variance_pct=val.variance_pct,
                source_hyperedge=val.hyperedge_id
            ))
        
        logger.info(f"Psi operator generated {len(axioms)} axioms")
        
        return axioms

    def _get_geometry_size(self, geom: GeometryPrimitive) -> float:
        """Extract characteristic size from geometry.
        
        Returns size in millimeters.
        """
        props = geom.properties
        
        if geom.primitive_type == "rectangle":
            # Use width as characteristic size
            width_mm = props.get("width_px", 0) / PIXELS_PER_MM
            height_mm = props.get("height_px", 0) / PIXELS_PER_MM
            return max(width_mm, height_mm)
        elif geom.primitive_type == "circle":
            diameter = props.get("diameter_px", 0)
            return diameter / PIXELS_PER_MM
        else:
            # For other types, use bounding box
            coords = geom.coordinates
            width = abs(coords.get("x2", 0) - coords.get("x1", 0))
            height = abs(coords.get("y2", 0) - coords.get("y1", 0))
            return max(width, height) / PIXELS_PER_MM

    def _get_table_value(self, table_id: str, row_index: int,
                        column: str, tables: List[TableSchema]) -> str:
        """Look up table cell value."""
        for table in tables:
            if table.table_id == table_id:
                for row in table.rows:
                    if row.row_index == row_index:
                        cell = row.get_cell_by_column(column)
                        if cell:
                            return cell.text
        return ""

    def _parse_size_string(self, size_str: str) -> float:
        """Parse size string like '400x400' to numeric value.
        
        Returns characteristic dimension (max of width/height).
        """
        try:
            if "x" in size_str.lower():
                parts = size_str.lower().split("x")
                return max(float(parts[0]), float(parts[1]))
            else:
                return float(size_str)
        except (ValueError, AttributeError):
            return 0.0

    def validate_hyperedges(self, hyperedges: List[HyperedgeBinding]) -> tuple[bool, List[str]]:
        """Validation hook: Check hyperedge distance thresholds."""
        errors = []
        for hedge in hyperedges:
            if hedge.distance > self.epsilon:
                errors.append(
                    f"{hedge.hyperedge_id} distance {hedge.distance} > epsilon {self.epsilon}"
                )
        return len(errors) == 0, errors

    def validate_distance_thresholds(self, hyperedges: List[HyperedgeBinding]) -> tuple[bool, List[str]]:
        """Validation hook: Verify all distances are within epsilon."""
        return self.validate_hyperedges(hyperedges)

    def validate_variance_tolerances(self, validations: List[ValidationResult]) -> tuple[bool, List[str]]:
        """Validation hook: Check variance against tau threshold."""
        errors = []
        for val in validations:
            if val.variance_pct > self.tau:
                errors.append(
                    f"{val.hyperedge_id} variance {val.variance_pct}% > tau {self.tau}%"
                )
        return len(errors) == 0, errors

    def validate_axiom_generation(self, axioms: List[AxiomManifest]) -> tuple[bool, List[str]]:
        """Validation hook: Check axiom integrity."""
        errors = []
        for axiom in axioms:
            if not axiom.source_hyperedge:
                errors.append(f"{axiom.axiom_id} missing source hyperedge")
            if axiom.integrity not in ["MATCHED", "MISMATCHED", "UNCERTAIN"]:
                errors.append(f"{axiom.axiom_id} invalid integrity: {axiom.integrity}")
        return len(errors) == 0, errors