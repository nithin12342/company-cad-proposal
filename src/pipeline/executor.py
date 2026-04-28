"""LKG Pipeline Executor.

Orchestrates execution of all 5 nodes in the Logical Knowledge Graph pipeline.
Ensures proper handoff between nodes with schema validation.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from ..core.node import LogicalKnowledgeNode, PipelineDataLossError
from ..core.schemas import (
    GeometryBRepSchema, TableSchema, AxiomManifest
)

logger = logging.getLogger(__name__)


class LKGPipeline:
    """Logical Knowledge Graph Pipeline Executor."""

    def __init__(self, nodes: Dict[str, LogicalKnowledgeNode]):
        """Initialize pipeline with injected nodes.
        
        Args:
            nodes: Dictionary of node objects mapping to their stages.
        """
        self.nodes = nodes

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
            stage1, j1 = self.nodes["triage"].execute(input_pdf)
            results["nodes"]["triage"] = self._format_node_result(stage1, j1)
            if not stage1.success:
                raise PipelineError("Node 01 failed", stage1.errors)

            # Stage 2: Geometric Extraction
            logger.info("\n[Stage 2/5] Geometric Extraction (Symbolic)")
            stage2, j2 = self.nodes["vectorize"].execute(
                stage1.data.geometry_mask_path,
                page_number=1
            )
            results["nodes"]["vectorize"] = self._format_node_result(stage2, j2)
            if not stage2.success:
                raise PipelineError("Node 02 failed", stage2.errors)

            # Stage 3: Layout Extraction
            logger.info("\n[Stage 3/5] Layout Intelligence (Neuro-Symbolic)")
            stage3, j3 = self.nodes["layout"].execute(
                stage1.data.table_mask_path,
                stage1.data.text_mask_path,
                page_number=1,
                original_file_path=input_pdf
            )
            results["nodes"]["layout"] = self._format_node_result(stage3, j3)
            if not stage3.success:
                raise PipelineError("Node 03 failed", stage3.errors)

            # Stage 4: DHMoT Processing
            logger.info("\n[Stage 4/5] DHMoT Agent (Relational)")
            geometry: GeometryBRepSchema = stage2.data
            tables: list[TableSchema] = stage3.data
            stage4, j4 = self.nodes["dhmot"].execute(
                geometry,
                tables,
                original_img_path=stage1.data.geometry_mask_path
            )
            results["nodes"]["dhmot"] = self._format_node_result(stage4, j4)
            if not stage4.success:
                raise PipelineError("Node 04 failed", stage4.errors)

            # Stage 5: Compliance Oracle
            logger.info("\n[Stage 5/5] Compliance Oracle (Oracle)")
            axioms = stage4.data["axioms"]
            axiom_objs = [AxiomManifest.from_dict(a) for a in axioms]
            document_id = Path(input_pdf).stem
            stage5, j5 = self.nodes["oracle"].execute(axiom_objs, document_id)
            results["nodes"]["oracle"] = self._format_node_result(stage5, j5)
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

    def _format_node_result(self, node_output, journal) -> Dict[str, Any]:
        """Format node output for results dictionary."""
        return {
            "success": node_output.success,
            "node_id": node_output.data.__class__.__name__ if node_output.success else None,
            "errors": node_output.errors,
            "metadata": node_output.metadata,
            "journal": journal.dict() if journal else None
        }


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""

    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.stage = message.split()[0] if message else "unknown"  # e.g., "Node 01"
        self.errors = errors or []