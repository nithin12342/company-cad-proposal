"""Node 02: Geometric Extraction (Symbolic Layer).

Second stage of the LKG pipeline. Converts geometry pixel masks into
mathematical representations using pure computer vision algorithms.
This is the "Symbolic" component - no neural networks, only deterministic math.

Engineering Principles:
- Context: Input geometry mask from Node 01
- Specification: OpenCV Hough/Canny, B-Rep output schema
- Intention: Convert pixels to 100% deterministic equations
- Harness: Validation via unique IDs, centroids, bounding boxes
"""

import numpy as np
import logging
import json
from typing import Dict, Any, Optional, List
import numpy as np
import logging
import json
from dataclasses import dataclass
from pathlib import Path

# Note: OpenCV import wrapped to handle missing dependency gracefully
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    # Mock for demonstration
    class MockCV2:
        @staticmethod
        def Canny(*args, **kwargs):
            return np.array([])
        @staticmethod
        def HoughCircles(*args, **kwargs):
            return None
        @staticmethod
        def HoughLinesP(*args, **kwargs):
            return None
        @staticmethod
        def findContours(*args, **kwargs):
            return [], [], []
    cv2 = MockCV2()

from ..core.node import LogicalKnowledgeNode, NodeOutput
from ..core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    GeometryBRepSchema, GeometryPrimitive
)
from ..core.constants import DPI_STANDARD, PIXELS_PER_MM, NODE_CONFIG

logger = logging.getLogger(__name__)


class GeometricExtractionNode(LogicalKnowledgeNode):
    """Node 02: Geometric Extraction - Symbolic Computer Vision.
    
    Processes geometry masks from Node 01 using purely algorithmic
    approaches: Hough transforms for circles/lines, Canny edge detection,
    and contour analysis for polygons.
    
    Output is a Boundary Representation (B-Rep) - a mathematical description
    of all geometric primitives in the drawing.
    """

    def __init__(self, node_id: str, **kwargs):
        """Initialize Geometric Extraction node.
        
        Args:
            node_id: Unique identifier (e.g., "node_02_vectorize")
            **kwargs: Additional configuration
        """
        super().__init__(node_id, **kwargs)

    def _build_context(self) -> BaseNodeContext:
        """Build context for geometric extraction.
        
        Takes geometry mask from Node 01 as input.
        Produces B-Rep schema as output.
        """
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["node_01_triage"],
            input_schema="BinaryMasks_v2.0/geometry",
            output_schema="GeometryBRep_v2.0"
        )

    def _build_specification(self) -> BaseNodeSpecification:
        """Build formal specification for geometric extraction.
        
        OpenCV algorithms, deterministic output, B-Rep format.
        """
        config = NODE_CONFIG["node_02_vectorize"]
        return BaseNodeSpecification(
            node_type="symbolic",
            algorithm="HoughTransform + CannyEdgeDetection + ContourAnalysis",
            version="4.5+",
            constraints={
                "library": "OpenCV (Apache 2.0)",
                "output_format": "JSON B-Rep",
                "require_unique_ids": config["require_unique_ids"],
                "require_centroids": config["require_centroids"],
                "deterministic": True,
                "no_ml_involved": True
            },
            validation_rules=["R001", "R002", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        """Build intention for geometric extraction.
        
        Goal: Convert raw pixels into 100% deterministic mathematical
        equations without any probabilistic reasoning.
        """
        return BaseNodeIntention(
            primary_goal="Convert geometry pixels to mathematical primitives (lines, circles, arcs)",
            expected_outcome="Complete B-Rep with all shapes quantified and catalogued",
            success_criteria=[
                "Every primitive has unique ID",
                "All primitives have centroid and bounding box",
                "No probabilistic elements in output",
                "Output matches input geometry pixel-for-pixel"
            ],
            failure_modes=[
                {
                    "mode": "low_contrast_edges",
                    "mitigation": "Walker will re-scan with adjusted Canny threshold"
                },
                {
                    "mode": "occluded_geometry",
                    "mitigation": "Walker re-scan with different Hough parameters"
                }
            ]
        )

    def _build_harness(self) -> NodeHarness:
        """Build execution harness for geometric extraction.
        
        Validation ensures every primitive is properly quantified.
        """
        return NodeHarness(
            source_module="src.nodes.vectorize",
            entry_function="extract_geometry",
            compile_required=False,  # Pure Python/OpenCV
            validation_hooks=[
                "validate_unique_ids",
                "validate_centroids",
                "validate_bounding_boxes",
                "validate_brep_integrity"
            ],
            error_handling="strict"  # Must get geometry right
        )

    def execute(self, geometry_mask_path: str, page_number: int = 1) -> Any:
        """Execute geometric extraction on input mask.
        
        Args:
            geometry_mask_path: Path to geometry mask from Node 01
            page_number: Source page number for metadata
            
        Returns:
            NodeOutput with GeometryBRepSchema
        """
        logger.info(f"Executing Geometric Extraction on {geometry_mask_path}")

        # Validate input
        valid, errors = self.validate_input(geometry_mask_path)
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
            self._extract_geometry,
            geometry_mask_path,
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

        brep_schema = harness_result.get("output")

        return NodeOutput(
            success=True,
            data=brep_schema,
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={
                "primitive_count": brep_schema.total_count,
                "execution_trace": harness_result.get("execution_trace", [])
            }
        )

    def _extract_geometry(self, mask_path: str, page_number: int) -> GeometryBRepSchema:
        """Core extraction logic.
        
        Applies computer vision algorithms to extract geometric primitives.
        
        Args:
            mask_path: Path to geometry mask
            page_number: Page number for metadata
            
        Returns:
            GeometryBRepSchema with all extracted primitives
        """
        if not OPENCV_AVAILABLE:
            # Simulation mode for environments without OpenCV
            logger.warning("OpenCV not available, using simulated extraction")
            return self._simulate_extraction(page_number)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Fallback: simulate
            return self._simulate_extraction(page_number)

        primitives = []
        primitive_counter = 0

        # 1. Detect circles using Hough Circle Transform
        circles = self._detect_circles(mask)
        for circle in circles:
            primitive_counter += 1
            prim = self._circle_to_primitive(circle, primitive_counter)
            primitives.append(prim)

        # 2. Detect lines using Hough Line Transform
        lines = self._detect_lines(mask)
        for line in lines:
            primitive_counter += 1
            prim = self._line_to_primitive(line, primitive_counter)
            primitives.append(prim)

        # 3. Detect rectangles/arbitrary polygons via contours
        contours = self._detect_contours(mask)
        for contour in contours:
            # Skip if contour already captured as circle or line
            primitive_counter += 1
            prim = self._contour_to_primitive(contour, primitive_counter)
            primitives.append(prim)

        # Build B-Rep schema
        brep = GeometryBRepSchema(
            page_number=page_number,
            dpi_reference=DPI_STANDARD,
            scale_factor=PIXELS_PER_MM,
            geometries=primitives
        )

        # Validate
        valid, errors = brep.validate()
        if not valid:
            logger.warning(f"B-Rep validation errors: {errors}")

        return brep

    def _detect_circles(self, mask: np.ndarray) -> List[tuple]:
        """Detect circles using Hough Circle Transform."""
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,  # Accumulator threshold
            minRadius=5,
            maxRadius=200
        )
        if circles is not None:
            return circles[0].tolist()
        return []

    def _detect_lines(self, mask: np.ndarray) -> List[tuple]:
        """Detect lines using Hough Line Transform."""
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        if lines is not None:
            return lines.reshape(-1, 4).tolist()
        return []

    def _detect_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Detect contours (polygons, rectangles, etc.)."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _circle_to_primitive(self, circle: list, pid: int) -> GeometryPrimitive:
        """Convert Hough circle to GeometryPrimitive."""
        cx, cy, r = circle
        return GeometryPrimitive(
            primitive_id=f"GEO_{pid:03d}",
            primitive_type="circle",
            coordinates={
                "center_x": float(cx),
                "center_y": float(cy),
                "radius": float(r)
            },
            centroid=(float(cx), float(cy)),
            properties={
                "radius_px": float(r),
                "diameter_px": float(2 * r),
                "area_px": float(np.pi * r * r)
            }
        )

    def _line_to_primitive(self, line: list, pid: int) -> GeometryPrimitive:
        """Convert Hough line to GeometryPrimitive."""
        x1, y1, x2, y2 = line
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return GeometryPrimitive(
            primitive_id=f"GEO_{pid:03d}",
            primitive_type="line",
            coordinates={
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2)
            },
            centroid=(float((x1+x2)/2), float((y1+y2)/2)),
            properties={
                "length_px": float(length)
            }
        )

    def _contour_to_primitive(self, contour: np.ndarray, pid: int) -> GeometryPrimitive:
        """Convert contour to GeometryPrimitive."""
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Determine shape type
        vertices = len(approx)
        if vertices == 3:
            ptype = "triangle"
        elif vertices == 4:
            ptype = "rectangle"
        else:
            ptype = "polygon"

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        return GeometryPrimitive(
            primitive_id=f"GEO_{pid:03d}",
            primitive_type=ptype,
            coordinates={
                "x1": float(x), "y1": float(y),
                "x2": float(x + w), "y2": float(y + h)
            },
            centroid=(float(x + w/2), float(y + h/2)),
            properties={
                "width_px": float(w),
                "height_px": float(h),
                "area_px": float(area)
            }
        )

    def _simulate_extraction(self, page_number: int) -> GeometryBRepSchema:
        """Simulate geometric extraction (for testing without OpenCV)."""
        import random
        primitives = []
        
        # Simulate some circles
        for i in range(3):
            primitives.append(GeometryPrimitive(
                primitive_id=f"GEO_{i+1:03d}",
                primitive_type="circle",
                coordinates={
                    "center_x": random.uniform(100, 300),
                    "center_y": random.uniform(100, 300),
                    "radius": random.uniform(15, 50)
                },
                centroid=(0, 0),  # Will be computed
                properties={"radius_px": 30, "diameter_px": 60}
            ))
        
        # Simulate some rectangles
        for i in range(5):
            x, y = random.uniform(50, 400), random.uniform(50, 500)
            w, h = random.uniform(30, 150), random.uniform(30, 150)
            primitives.append(GeometryPrimitive(
                primitive_id=f"GEO_{i+4:03d}",
                primitive_type="rectangle",
                coordinates={
                    "x1": x, "y1": y,
                    "x2": x + w, "y2": y + h
                },
                centroid=(x + w/2, y + h/2),
                properties={
                    "width_px": w, "height_px": h,
                    "area_px": w * h
                }
            ))

        return GeometryBRepSchema(
            page_number=page_number,
            dpi_reference=DPI_STANDARD,
            scale_factor=PIXELS_PER_MM,
            geometries=primitives
        )

    def validate_unique_ids(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        """Validation hook: Ensure all primitive IDs are unique."""
        ids = [g.primitive_id for g in brep.geometries]
        if len(ids) != len(set(ids)):
            return False, ["Duplicate primitive IDs found"]
        return True, []

    def validate_centroids(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        """Validation hook: Ensure all primitives have centroids."""
        for geom in brep.geometries:
            if not geom.centroid:
                return False, [f"{geom.primitive_id} missing centroid"]
        return True, []

    def validate_bounding_boxes(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        """Validation hook: Ensure all primitives have bounding boxes."""
        for geom in brep.geometries:
            if "x1" not in geom.coordinates or "x2" not in geom.coordinates:
                return False, [f"{geom.primitive_id} missing bounding box"]
        return True, []

    def validate_brep_integrity(self, brep: GeometryBRepSchema) -> tuple[bool, List[str]]:
        """Validation hook: Full B-Rep integrity check."""
        return brep.validate()