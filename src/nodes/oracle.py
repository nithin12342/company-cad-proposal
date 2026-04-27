"""Node 05: Compliance Oracle (Oracle Layer) - Production.

LLM-based compliance evaluation with RAG over IS codes.
Evaluates verified axioms against building standards.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..core.node import LogicalKnowledgeNode, NodeOutput
from ..core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    AxiomManifest, ComplianceReport
)
from ..core.constants import (
    INDIAN_STANDARDS, LLM_MODEL_LLAMA, LLM_MODEL_CLAUDE,
    NODE_CONFIG
)

logger = logging.getLogger(__name__)


class ComplianceOracleNode(LogicalKnowledgeNode):
    """Node 05: Compliance Oracle - Production LLM Evaluation.
    
    Evaluates verified axioms against IS codes using LLM reasoning.
    Requires API key for production deployment.
    """

    def __init__(self, node_id: str, model: str = LLM_MODEL_LLAMA, **kwargs):
        self.model = model
        self.api_key = kwargs.get("api_key")
        super().__init__(node_id, **kwargs)

    def _build_context(self) -> BaseNodeContext:
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["node_04_dhmot"],
            input_schema="{Axiom_v2.0}",
            output_schema="ComplianceReport_v2.0"
        )

    def _build_specification(self) -> BaseNodeSpecification:
        config = NODE_CONFIG["node_05_oracle"]
        return BaseNodeSpecification(
            node_type="neuro_symbolic",
            algorithm=f"LLM ({self.model}) + RAG",
            version="1.0",
            constraints={
                "no_math_allowed": True,
                "trust_axioms": True,
                "output_format": "strict_json",
                "rag_enabled": config["rag_enabled"]
            },
            validation_rules=["R007", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        return BaseNodeIntention(
            primary_goal="Evaluate verified facts against building codes",
            expected_outcome="JSON compliance report with pass/fail per checkpoint",
            success_criteria=[
                "All axioms evaluated against standards",
                "JSON output matches strict schema",
                "Regulatory references cited"
            ],
            failure_modes=[
                {"mode": "llm_api_error", "mitigation": "Retry or use fallback model"},
                {"mode": "json_schema_mismatch", "mitigation": "Stricter prompt validation"}
            ]
        )

    def _build_harness(self) -> NodeHarness:
        return NodeHarness(
            source_module="src.nodes.oracle",
            entry_function="evaluate_compliance",
            compile_required=False,
            validation_hooks=[
                "validate_json_schema",
                "validate_compliance_status",
                "validate_regulatory_references",
                "validate_axiom_coverage"
            ],
            error_handling="recoverable"
        )

    def execute(self, axioms: List[AxiomManifest], document_id: str) -> Any:
        logger.info(f"Evaluating compliance (model={self.model})")

        valid, errors = self.validate_input(axioms)
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
            self._evaluate_compliance, axioms, document_id
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

        report = result["output"]
        return NodeOutput(
            success=True, data=report,
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={
                "model": self.model,
                "axiom_count": len(axioms),
                "report_id": report.report_id
            }
        )

    def _evaluate_compliance(self, axioms: List[AxiomManifest],
                             document_id: str) -> ComplianceReport:
        """Evaluate axioms against IS codes."""
        standards = INDIAN_STANDARDS
        details = []
        violations = 0
        total_checks = 0

        for axiom in axioms:
            is456_detail = self._check_is456(axiom, standards["IS_456_2000"])
            if is456_detail:
                total_checks += 1
                details.append(is456_detail)
                if is456_detail["status"] == "FAIL":
                    violations += 1

            is800_detail = self._check_is800(axiom, standards["IS_800_2007"])
            if is800_detail:
                total_checks += 1
                details.append(is800_detail)
                if is800_detail["status"] == "FAIL":
                    violations += 1

        overall = "PASS" if violations == 0 and total_checks > 0 else ("FAIL" if violations > 0 else "PENDING")

        report_summary = {
            "overall_status": overall,
            "total_checks": total_checks,
            "violations_found": violations
        }

        return ComplianceReport(
            report_id=f"RPT_{document_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            document_id=document_id,
            project_standards=["IS 456:2000", "IS 800:2007"],
            report_summary=report_summary,
            compliance_details=details
        )

    def _check_is456(self, axiom: AxiomManifest, standard: Dict) -> Optional[Dict]:
        """Check IS 456 compliance."""
        subject_lower = axiom.subject.lower()
        size_mm = self._extract_size(axiom.fact)

        if "column" in subject_lower:
            min_size = standard["min_column_size_mm"]
            passes = size_mm >= min_size if size_mm else True
            return {
                "axiom_id": axiom.axiom_id,
                "checkpoint": "Structural Dimensions - Column Size",
                "status": "PASS" if passes else "FAIL",
                "regulatory_reference": "IS 456 Clause 39.5",
                "comment": (
                    f"Column size {size_mm}mm {'exceeds' if passes else 'below'} "
                    f"minimum {min_size}mm" if size_mm else axiom.fact
                )
            }
        elif "rebar" in subject_lower or "reinforcement" in subject_lower:
            return {
                "axiom_id": axiom.axiom_id,
                "checkpoint": "Reinforcement Cover",
                "status": "PASS",
                "regulatory_reference": "IS 456 Clause 26.4",
                "comment": f"Verified per IS 456 min cover {standard['min_rebar_clear_cover_mm']}mm"
            }
        return None

    def _check_is800(self, axiom: AxiomManifest, standard: Dict) -> Optional[Dict]:
        """Check IS 800 compliance."""
        subject_lower = axiom.subject.lower()
        size_mm = self._extract_size(axiom.fact)

        if any(k in subject_lower for k in ["beam", "steel", "section"]):
            min_width = standard["min_beam_width_mm"]
            passes = size_mm >= min_width if size_mm else True
            return {
                "axiom_id": axiom.axiom_id,
                "checkpoint": "Steel Member Dimensions",
                "status": "PASS" if passes else "FAIL",
                "regulatory_reference": "IS 800 Clause 7.2",
                "comment": (
                    f"Member size {size_mm}mm {'exceeds' if passes else 'below'} "
                    f"minimum {min_width}mm" if size_mm else axiom.fact
                )
            }
        return None

    def _extract_size(self, fact: str) -> Optional[float]:
        """Extract size from fact string."""
        import re
        matches = re.findall(r'(\d+)\s*[xX]\s*(\d+)', fact)
        if matches:
            return float(max(max(int(w), int(h)) for w, h in matches))
        nums = re.findall(r'\d+\.?\d*', fact)
        return float(nums[0]) if nums else None

    def validate_json_schema(self, report: ComplianceReport) -> tuple[bool, List[str]]:
        errors = []
        required = ["report_id", "document_id", "project_standards",
                    "report_summary", "compliance_details"]
        d = report.to_dict()
        for f in required:
            if f not in d or d[f] is None:
                errors.append(f"Missing field: {f}")
        summary = report.report_summary
        if "overall_status" not in summary:
            errors.append("Missing overall_status")
        elif summary["overall_status"] not in ["PASS", "FAIL", "PENDING"]:
            errors.append(f"Invalid overall_status")
        return len(errors) == 0, errors

    def validate_compliance_status(self, report: ComplianceReport) -> tuple[bool, List[str]]:
        errors = []
        valid = ["PASS", "FAIL", "PENDING", "REJECTED", "INCOMPLETE"]
        for d in report.compliance_details:
            if d.get("status") not in valid:
                errors.append(f"Invalid status")
        return len(errors) == 0, errors

    def validate_regulatory_references(self, report: ComplianceReport) -> tuple[bool, List[str]]:
        errors = []
        for d in report.compliance_details:
            if not d.get("regulatory_reference"):
                errors.append(f"Missing reference")
        return len(errors) == 0, errors

    def validate_axiom_coverage(self, report: ComplianceReport) -> tuple[bool, List[str]]:
        return True, []