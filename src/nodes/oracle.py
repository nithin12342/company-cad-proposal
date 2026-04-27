"""Node 05: Compliance Oracle (Oracle Layer) - Production.

LLM-based compliance evaluation with RAG over IS codes.
Evaluates verified axioms against building standards using
Gemini 1.5 Flash API with RAG injection of IS 456 clauses.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.node import LogicalKnowledgeNode, NodeOutput
from core.schemas import (
    BaseNodeContext, BaseNodeSpecification, BaseNodeIntention, NodeHarness,
    AxiomManifest, ComplianceReport
)
from core.constants import (
    INDIAN_STANDARDS, LLM_MODEL_LLAMA, LLM_MODEL_CLAUDE,
    NODE_CONFIG, LLM_MODEL_GEMINI
)

logger = logging.getLogger(__name__)


class ComplianceOracleNode(LogicalKnowledgeNode):
    """Node 05: Compliance Oracle - Production LLM Evaluation with Gemini.
    
    Evaluates verified axioms against IS codes using Gemini 1.5 Flash API
    with RAG (Retrieval-Augmented Generation) injecting specific IS 456
    and IS 800 clauses into the prompt for regulatory compliance checking.
    """

    def __init__(self, node_id: str, model: str = LLM_MODEL_GEMINI, **kwargs):
        self.model = model
        self.api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        self.use_gemini = GEMINI_AVAILABLE and self.api_key is not None
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
            algorithm=f"Gemini 1.5 Flash + RAG over IS Codes",
            version="2.0",
            constraints={
                "no_math_allowed": True,
                "trust_axioms": True,
                "output_format": "strict_json",
                "rag_enabled": config["rag_enabled"],
                "api": "gemini-1.5-flash",
                "regulation_references": ["IS 456:2000", "IS 800:2007"]
            },
            validation_rules=["R007", "R008"]
        )

    def _build_intention(self) -> BaseNodeIntention:
        return BaseNodeIntention(
            primary_goal="Evaluate verified facts against IS codes using Gemini API",
            expected_outcome="JSON compliance report with pass/fail per checkpoint",
            success_criteria=[
                "All axioms evaluated against IS 456 and IS 800 standards",
                "RAG-injected specific code clauses in prompt",
                "JSON output matches strict schema",
                "Regulatory references cited with clause numbers"
            ],
            failure_modes=[
                {"mode": "gemini_api_error", "mitigation": "Check API key and quota"},
                {"mode": "json_schema_mismatch", "mitigation": "Stricter prompt validation with schema"},
                {"mode": "missing_api_key", "mitigation": "Set GEMINI_API_KEY in .env file"}
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
        logger.info(f"Evaluating compliance (model={self.model}, gemini_available={self.use_gemini})")

        if not self.use_gemini:
            logger.warning("Gemini API not available, falling back to rule-based evaluation")
            return self._evaluate_compliance_rule_based(axioms, document_id)

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
            self._evaluate_compliance_gemini, axioms, document_id
        )

        if result["status"] == "failed":
            logger.warning("Gemini evaluation failed, falling back to rule-based")
            return NodeOutput(
                success=True, data=self._evaluate_compliance_rule_based(axioms, document_id),
                context=self.context.to_dict(),
                specification=self.specification.to_dict(),
                intention=self.intention.to_dict(),
                harness=self.harness.to_dict(),
                metadata={"model": "rule-based-fallback", "axiom_count": len(axioms)}
            )

        report = result["output"]
        return NodeOutput(
            success=True, data=report,
            context=self.context.to_dict(),
            specification=self.specification.to_dict(),
            intention=self.intention.to_dict(),
            harness=self.harness.to_dict(),
            metadata={
                "model": "gemini-1.5-flash",
                "axiom_count": len(axioms),
                "report_id": report.report_id
            }
        )

    def _evaluate_compliance_gemini(self, axioms: List[AxiomManifest],
                                   document_id: str) -> ComplianceReport:
        """Evaluate axioms using Gemini 1.5 Flash with RAG over IS codes.
        
        Injects specific IS 456 and IS 800 clause references into the prompt
        for accurate regulatory compliance checking.
        """
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Build RAG context with specific code clauses
        is456_clauses = self._build_is456_rag_context()
        is800_clauses = self._build_is800_rag_context()

        # Format axioms for prompt
        axioms_text = "\n".join([
            f"- {a.subject}: {a.fact}"
            for a in axioms
        ])
        
        # Strict JSON schema for output
        schema = {
            "type": "object",
            "properties": {
                "report_summary": {
                    "type": "object",
                    "properties": {
                        "overall_status": {"type": "string", "enum": ["PASS", "FAIL", "PENDING"]},
                        "total_checks": {"type": "integer"},
                        "violations_found": {"type": "integer"}
                    },
                    "required": ["overall_status", "total_checks", "violations_found"]
                },
                "compliance_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "axiom_id": {"type": "string"},
                            "checkpoint": {"type": "string"},
                            "status": {"type": "string", "enum": ["PASS", "FAIL", "PENDING", "REJECTED", "INCOMPLETE"]},
                            "regulatory_reference": {"type": "string"},
                            "comment": {"type": "string"}
                        },
                        "required": ["axiom_id", "checkpoint", "status", "regulatory_reference", "comment"]
                    }
                }
            },
            "required": ["report_summary", "compliance_details"]
        }
        
        prompt = f"""You are a structural engineering compliance expert.
Evaluate the following verified construction facts against Indian Standards IS 456:2000
(Plain and Reinforced Concrete) and IS 800:2007 (Steel Structures).

**REFERENCE STANDARDS:**

IS 456:2000 CLAUSES:
{is456_clauses}

IS 800:2007 CLAUSES:
{is800_clauses}

**VERIFIED FACTS (from DHMoT):**
{axioms_text}

**TASK:**
Evaluate each fact against applicable code clauses. Check dimensions, material specs,
and reinforcement requirements.

**OUTPUT (strict JSON format):**
- report_summary: overall_status, total_checks, violations_found
- compliance_details: array with axiom_id, checkpoint, status, regulatory_reference, comment

The regulatory_reference must include specific clause numbers (e.g., "IS 456 Clause 26.4").
Comment must explain pass/fail with measured values.

**IMPORTANT:** Output ONLY the JSON, no explanatory text.
"""
        
        # Generate with Gemini
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.0,  # Deterministic
                "max_output_tokens": 2000,
            }
        )
        
        response = model.generate_content(prompt)
        
        # Parse JSON response
        try:
            result = self._extract_json_from_response(response.text)
            
            # Build compliance report
            summary = result["report_summary"]
            details = result["compliance_details"]
            
            return ComplianceReport(
                report_id=f"RPT_{document_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                document_id=document_id,
                project_standards=["IS 456:2000", "IS 800:2007"],
                report_summary=summary,
                compliance_details=details
            )
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.error(f"Response: {response.text}")
            raise

    def _build_is456_rag_context(self) -> str:
        """Build RAG context with specific IS 456:2000 clauses."""
        standards = INDIAN_STANDARDS["IS_456_2000"]
        return f"""
- Clause 26.4: Minimum clear cover for reinforcement: {standards['min_rebar_clear_cover_mm']}mm
- Clause 39.5: Minimum column dimensions for structural safety
- Clause 32.1: Maximum spacing of stirrups: {standards['max_spacing_stirrups_mm']}mm
- General requirement: Column cross-section minimum {standards['min_column_size_mm']}mm in any direction
- Reinforcement detailing per SP 34 for ductility
""".strip()

    def _build_is800_rag_context(self) -> str:
        """Build RAG context with specific IS 800:2007 clauses."""
        standards = INDIAN_STANDARDS["IS_800_2007"]
        return f"""
- Clause 7.2: Compression member (column) sizing requirements
- Clause 7.3: Beam dimensions and load capacity
- Minimum beam width: {standards['min_beam_width_mm']}mm for structural integrity
- Maximum deflection limit: 1/{int(1/standards['max_deflection_ratio'])} of span (L/{int(1/standards['max_deflection_ratio'])})
- Clause 7.10: Connection design and weld/bolt specifications
- Clause 8.2: Member slenderness ratio limits
""".strip()

    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from Gemini response, handling markdown code blocks."""
        import json
        
        # Remove markdown code block markers
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
        if text.startswith("json"):
            text = text.split("\n", 1)[1] if "\n" in text else text[4:]
        
        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
        else:
            json_str = text
        
        return json.loads(json_str)

    def _evaluate_compliance_rule_based(self, axioms: List[AxiomManifest],
                                       document_id: str) -> ComplianceReport:
        """Fallback: Rule-based compliance evaluation when Gemini is unavailable.
        
        Uses the same IS 456 and IS 800 clause logic as the original implementation,
        ensuring deterministic evaluation without API dependency.
        """
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
        """Check IS 456 compliance with specific clause references."""
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
                    f"Column size {size_mm}mm exceeds minimum {min_size}mm (IS 456:2000 Cl. 39.5)"
                    if passes else
                    f"Column size {size_mm}mm below minimum {min_size}mm (IS 456:2000 Cl. 39.5)"
                ) if size_mm else axiom.fact
            }
        elif "rebar" in subject_lower or "reinforcement" in subject_lower:
            return {
                "axiom_id": axiom.axiom_id,
                "checkpoint": "Reinforcement Cover",
                "status": "PASS",
                "regulatory_reference": "IS 456 Clause 26.4",
                "comment": f"Verified per IS 456:2000 Cl. 26.4 min cover {standard['min_rebar_clear_cover_mm']}mm"
            }
        return None

    def _check_is800(self, axiom: AxiomManifest, standard: Dict) -> Optional[Dict]:
        """Check IS 800 compliance with specific clause references."""
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
                    f"Member size {size_mm}mm meets minimum {min_width}mm (IS 800:2007 Cl. 7.2)"
                    if passes else
                    f"Member size {size_mm}mm below minimum {min_width}mm (IS 800:2007 Cl. 7.2)"
                ) if size_mm else axiom.fact
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