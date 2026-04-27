"""Node 02: Geometric Extraction (Symbolic Layer) - Production.

Deterministic computer vision pipeline for extracting geometric primitives
from binary masks. Uses OpenCV for Hough transforms, Canny edge detection,
and contour analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

import cv2

from core.node import LogicalKnowledgeNode, NodeOutput
from core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    GeometryBRepSchema, GeometryPrimitive
)
from core.constants import DPI_STANDARD, PIXELS_PER_MM, NODE_CONFIG

logger = logging.getLogger(__name__)


class GeometricExtractionNode(LogicalKnowledgeNode):
    """Node 02: Geometric Extraction - Pure Computer Vision.
    
    Converts geometry masks to mathematical B-Rep using deterministic algorithms:
    - Hough Circle Transform for circles and holes
    - Probabilistic Hough Transform for line segments
    - Contour analysis for polygons and complex shapes
    """

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id, **kwargs)

    def _build_context(self) -> BaseNodeContext:
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["node_01_triage"],
            input_schema="BinaryMasks_v2.0/geometry",
            output_schema="GeometryBRep_v2.0"
        )

    def _build_specification(self) -> BaseNodeSpecification:
        config = NODE_CONFIG["node_02_vectorize"]
        return BaseNodeSpecification(
            node_type="symbolic",
            algorithm="HoughTransform + CannyEdgeDetection + ContourAnalysis",
            version="4.5+",
            constraints={
                "library": "OpenCV",
                "output_format": "JSON B-Rep",
                "deterministic": True,
                "no_ml": True
            },
            validation_rules=["R001", "R002", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        return BaseNodeIntention(
            primary_goal="Convert pixels to 100% deterministic equations",
            expected_outcome="Complete B-Rep with all shapes quantified",
            success_criteria=[
                "Every primitive has unique ID",
                "All primitives have centroid and bounds",
                "100% deterministic output"
            ],
            failure_modes=[
                {"mode": "low_contrast", "mitigation": "Walker re-scan"}
            ]
        )

    def _build_harness(self) -> NodeHarness:
        return NodeHarness(
            source_module="src.nodes.vectorize",
            entry_function="extract_geometry",
            compile_required=False,
            validation_hooks=[
                "validate_unique_ids",
                "validate_centroids",
                "validate_bounding_boxes",
                "validate_brep_integrity"
            ],
            error_handling="strict"
        )

    def execute(self, geometry_mask_path: str, page_number: int = 1) -> Any:
        logger.info(f"Extracting geometry from {geometry_mask_path}")

        valid, errors = self.validate_input(geometry_mask_path)
        if not valid:
            return NodeOutput(
                success=False, data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=errors
            )

        result = self._execute_with_harness(
            self._extract_geometry, geometry_mask_path, page_number
        )

        if result["status"] == "failed":
            return NodeOutput(
                success=False, data=None,
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                errors=result["errors"]
            )

        return NodeOutput(
            success=True, data=result["output"],
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={"primitive_count": result["output"].total_count}
        )

    def _extract_geometry(self, mask_path: str, page_number: int) -> GeometryBRepSchema:
        """Deterministic geometric primitive extraction."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        
        primitives = []
        pid = 0

        # Circles
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=200
        )
        if circles is not None:
            for c in np.uint16(np.around(circles))[0]:
                pid += 1
                primitives.append(self._circle_primitive(pid, c))

        # Lines
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines.reshape(-1, 4):
                pid += 1
                primitives.append(self._line_primitive(pid, line))

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pid += 1
            prim = self._contour_primitive(pid, contour)
            if prim:
                primitives.append(prim)

        # Filter duplicates
        primitives = self._filter_duplicates(primitives)

        brep = GeometryBRepSchema(
            page_number=page_number,
            dpi_reference=DPI_STANDARD,
            scale_factor=PIXELS_PER_MM,
            geometries=primitives
        )

        valid, errors = brep.validate()
        if not valid:
            raise RuntimeError(f"B-Rep validation failed: {errors}")

        logger.info(f"Extracted {len(primitives)} primitives")
        return brep

    def _circle_primitive(self, pid: int, c: List[float]) -> GeometryPrimitive:
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        return GeometryPrimitive(
            primitive_id=f"GEO_{pid:04d}",
            primitive_type="circle",
            coordinates={"center_x": cx, "center_y": cy, "radius": r},
            centroid=(cx, cy),
            properties={"radius_px": r, "diameter_px": 2*r, "area_px": float(np.pi*r*r)}
        )

    def _line_primitive(self, pid: int, line: List[float]) -> GeometryPrimitive:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return GeometryPrimitive(
            primitive_id=f"GEO_{pid:04d}",
            primitive_type="line",
            coordinates={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            centroid=(float((x1+x2)/2), float((y1+y2)/2)),
            properties={"length_px": length}
        )

    def _contour_primitive(self, pid: int, contour) -> Optional[GeometryPrimitive]:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        if vertices <= 2:
            return None
        types = {3: "triangle", 4: "rectangle", 5: "pentagon", 6: "hexagon"}
        ptype = types.get(vertices, "polygon")
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        return GeometryPrimitive(
            primitive_id=f"GEO_{pid:04d}",
            primitive_type=ptype,
            coordinates={"x1": float(x), "y1": float(y), "x2": float(x+w), "y2": float(y+h)},
            centroid=(float(x+w/2), float(y+h/2)),
            properties={"width_px": float(w), "height_px": float(h), "area_px": area, "vertices": vertices}
        )

    def _filter_duplicates(self, primitives: List[GeometryPrimitive]) -> List[GeometryPrimitive]:
        filtered = []
        for prim_i in primitives:
            duplicate = False
            for prim_j in filtered:
                if self._overlap(prim_i, prim_j):
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(prim_i)
        return filtered

    def _overlap(self, p1: GeometryPrimitive, p2: GeometryPrimitive) -> bool:
        c1, c2 = p1.coordinates, p2.coordinates
        if "x1" in c1 and "x1" in c2:
            return not (c1["x2"] < c2["x1"] or c2["x2"] < c1["x1"] or
                       c1["y2"] < c2["y1"] or c2["y2"] < c1["y1"])
        return False

    def validate_unique_ids(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        ids = [g.primitive_id for g in brep.geometries]
        return (True, []) if len(ids) == len(set(ids)) else (False, ["Duplicate IDs"])

    def validate_centroids(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        for g in brep.geometries:
            if not g.centroid:
                return False, [f"{g.primitive_id} missing centroid"]
        return True, []

    def validate_bounding_boxes(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        for g in brep.geometries:
            if "x1" not in g.coordinates:
                return False, [f"{g.primitive_id} missing bbox"]
        return True, []

    def validate_brep_integrity(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        return brep.validate()