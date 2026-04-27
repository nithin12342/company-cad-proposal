"""LKG Pipeline Executor.

Orchestrates execution of all 5 nodes in the Logical Knowledge Graph pipeline.
Ensures proper handoff between nodes with schema validation.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from ..nodes.triage import PixelTriageNode
from ..nodes.vectorize import GeometricExtractionNode
from ..nodes.layout import LayoutExtractionNode
from ..nodes.dhmot import DHMoTNode
from ..nodes.oracle import ComplianceOracleNode
from ..core.schemas import (
    GeometryBRepSchema, TableSchema, AxiomManifest
)

logger = logging.getLogger(__name__)


class LKGPipeline:
    """Logical Knowledge Graph Pipeline Executor.
    
    Orchestrates the 5-stage pipeline:
    1. Node 01: Pixel Triage (Neuro)
    2. Node 02: Geometric Extraction (Symbolic)
    3. Node 03: Layout Intelligence (Neuro-Symbolic)
    4. Node 04: DHMoT Agent (Relational)
    5. Node 05: Compliance Oracle (Oracle)
    
    Ensures error-free execution with proper context propagation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config or {}
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Initialize all pipeline nodes with proper IDs."""
        self.node_01 = PixelTriageNode(
            node_id="node_01_triage",
            **self.config.get("triage", {})
        )
        self.node_02 = GeometricExtractionNode(
            node_id="node_02_vectorize",
            **self.config.get("vectorize", {})
        )
        self.node_03 = LayoutExtractionNode(
            node_id="node_03_layout",
            **self.config.get("layout", {})
        )
        self.node_04 = DHMoTNode(
            node_id="node_04_dhmot",
            **self.config.get("dhmot", {})
        )
        self.node_05 = ComplianceOracleNode(
            node_id="node_05_oracle",
            **self.config.get("oracle", {})
        )

    def execute_full_pipeline(self, input_pdf: str) -> Dict[str, Any]:
        """Execute complete LKG pipeline on input PDF.
        
        Args:
            input_pdf: Path to scanned PDF document
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        logger.info("="*60)
        logger.info("Starting LKG Pipeline Execution")
        logger.info(f"Input: {input_pdf}")
        logger.info("="*60)

        results = {
            "pipeline_status": "running",
            "nodes": {},
            "final_report": None,
            "errors": []
        }

        try:
            # Stage 1: Pixel Triage
            logger.info("\n[Stage 1/5] Pixel Triage (Neuro)")
            stage1 = self.node_01.execute(input_pdf)
            results["nodes"]["triage"] = self._format_node_result(stage1)
            if not stage1.success:
                raise PipelineError("Node 01 failed", stage1.errors)

            # Stage 2: Geometric Extraction
            logger.info("\n[Stage 2/5] Geometric Extraction (Symbolic)")
            stage2 = self.node_02.execute(
                stage1.data.geometry_mask_path,
                page_number=1
            )
            results["nodes"]["vectorize"] = self._format_node_result(stage2)
            if not stage2.success:
                raise PipelineError("Node 02 failed", stage2.errors)

            # Stage 3: Layout Extraction
            logger.info("\n[Stage 3/5] Layout Intelligence (Neuro-Symbolic)")
            stage3 = self.node_03.execute(
                stage1.data.table_mask_path,
                stage1.data.text_mask_path,
                page_number=1
            )
            results["nodes"]["layout"] = self._format_node_result(stage3)
            if not stage3.success:
                raise PipelineError("Node 03 failed", stage3.errors)

            # Stage 4: DHMoT Processing
            logger.info("\n[Stage 4/5] DHMoT Agent (Relational)")
            geometry: GeometryBRepSchema = stage2.data
            tables: list[TableSchema] = stage3.data
            stage4 = self.node_04.execute(
                geometry,
                tables,
                original_img_path=stage1.data.geometry_mask_path
            )
            results["nodes"]["dhmot"] = self._format_node_result(stage4)
            if not stage4.success:
                raise PipelineError("Node 04 failed", stage4.errors)

            # Stage 5: Compliance Oracle
            logger.info("\n[Stage 5/5] Compliance Oracle (Oracle)")
            axioms = stage4.data["axioms"]
            axiom_objs = [AxiomManifest.from_dict(a) for a in axioms]
            document_id = Path(input_pdf).stem
            stage5 = self.node_05.execute(axiom_objs, document_id)
            results["nodes"]["oracle"] = self._format_node_result(stage5)
            if not stage5.success:
                raise PipelineError("Node 05 failed", stage5.errors)

            # Pipeline complete
            results["pipeline_status"] = "completed"
            results["final_report"] = stage5.data.to_dict()

            logger.info("\n" + "="*60)
            logger.info("Pipeline Execution Complete")
            logger.info("="*60)

        except PipelineError as e:
            logger.error(f"Pipeline failed: {e}")
            results["pipeline_status"] = "failed"
            results["errors"].append({
                "stage": e.stage,
                "errors": e.errors
            })
        except Exception as e:
            logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
            results["pipeline_status"] = "error"
            results["errors"].append({
                "stage": "unknown",
                "error": str(e)
            })

        return results

    def _format_node_result(self, node_output) -> Dict[str, Any]:
        """Format node output for results dictionary."""
        return {
            "success": node_output.success,
            "node_id": node_output.data.__class__.__name__ if node_output.success else None,
            "errors": node_output.errors,
            "metadata": node_output.metadata
        }


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""

    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.stage = message.split()[0] if message else "unknown"  # e.g., "Node 01"
        self.errors = errors or []