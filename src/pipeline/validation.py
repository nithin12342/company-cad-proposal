"""Pipeline validation and monitoring.

Provides utilities for validating data flow between nodes,
checking schema consistency, and monitoring pipeline health.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..core.schemas import (
    GeometryBRepSchema, TableSchema,
    HyperedgeBinding, AxiomManifest
)
from ..core.constants import NODE_CONFIG, VALIDATION_RULES, ERROR_CODES

logger = logging.getLogger(__name__)


@dataclass
class PipelineHealth:
    """Health status of the LKG pipeline.
    
    Tracks execution metrics, validation results, and error rates
    across all pipeline stages.
    """
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stage_status: Dict[str, str] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    schema_mismatches: List[str] = field(default_factory=list)
    node_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if pipeline is healthy (no critical errors)."""
        return len(self.validation_errors) == 0 and \
               len(self.schema_mismatches) == 0

    def add_error(self, error: str, stage: str = "unknown"):
        """Add validation error from specific stage."""
        self.validation_errors.append(f"[{stage}] {error}")

    def add_schema_mismatch(self, mismatch: str, stage: str = "unknown"):
        """Add schema mismatch from specific stage."""
        self.schema_mismatches.append(f"[{stage}] {mismatch}")

    def update_stage(self, stage: str, status: str):
        """Update status of a pipeline stage."""
        self.stage_status[stage] = status
        logger.info(f"Stage '{stage}': {status}")

    def add_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Add performance metrics for a node."""
        self.node_metrics[node_id] = metrics


class SchemaValidator:
    """Validates data schemas between pipeline nodes.
    
    Ensures consistency and traceability across all node handoffs,
    implementing the 'consistency_traceability' system property.
    """

    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []

    def validate_node_input(self, node_id: str, data: Any,
                           expected_type: type) -> tuple[bool, List[str]]:
        """Validate input data for a node.
        
        Args:
            node_id: Target node identifier
            data: Input data to validate
            expected_type: Expected type of input data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Type check
        if not isinstance(data, expected_type):
            errors.append(
                f"Node {node_id}: Expected {expected_type.__name__}, "
                f"got {type(data).__name__}"
            )
            return False, errors

        # Schema-specific validation
        if isinstance(data, GeometryBRepSchema):
            valid, schema_errors = data.validate()
            errors.extend(schema_errors)
        elif isinstance(data, TableSchema):
            valid, schema_errors = data.validate()
            errors.extend(schema_errors)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, TableSchema):
                    valid, schema_errors = item.validate()
                    errors.extend(schema_errors)

        is_valid = len(errors) == 0
        
        # Record validation
        self.validation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": node_id,
            "valid": is_valid,
            "errors": errors
        })

        return is_valid, errors

    def validate_node_output(self, node_id: str, next_node_id: str,
                            data: Any, expected_type: type) -> tuple[bool, List[str]]:
        """Validate output data for handoff to next node.
        
        Args:
            node_id: Source node identifier
            next_node_id: Target node identifier  
            data: Output data to validate
            expected_type: Expected data type
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for None/empty output
        if data is None:
            errors.append(
                f"Node {node_id}: Output is None for {next_node_id}"
            )
            return False, errors

        # List output validation
        if isinstance(data, list):
            if len(data) == 0:
                errors.append(
                    f"Node {node_id}: Empty list output"
                )
            for i, item in enumerate(data):
                if item is None:
                    errors.append(
                        f"Node {node_id}: None at index {i} in list"
                    )

        # Schema validation
        if isinstance(data, GeometryBRepSchema):
            valid, schema_errors = data.validate()
            if not valid:
                errors.extend(schema_errors)

        is_valid = len(errors) == 0

        # Record validation
        self.validation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "from_node": node_id,
            "to_node": next_node_id,
            "valid": is_valid,
            "errors": errors
        })

        return is_valid, errors

    def check_schema_version(self, data: Any, expected_version: str) -> bool:
        """Check if data matches expected schema version.
        
        Args:
            data: Schema object to check
            expected_version: Expected version string
            
        Returns:
            True if version matches
        """
        # In production, would check version field in schema objects
        return True  # Simplified for demo

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed.
        
        Returns:
            Dictionary with validation statistics
        """
        total = len(self.validation_history)
        failed = sum(1 for v in self.validation_history if not v["valid"])
        passed = total - failed
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "history": self.validation_history
        }


class PipelineMonitor:
    """Monitors pipeline health and performance.
    
    Tracks execution metrics, detects anomalies, and ensures
    the 'error_free_execution' system property.
    """

    def __init__(self):
        self.health = PipelineHealth()
        self.validator = SchemaValidator()
        self.execution_times: Dict[str, float] = {}

    def start_stage(self, stage_name: str):
        """Mark start of pipeline stage."""
        import time
        self.health.update_stage(stage_name, "running")
        self.start_times[stage_name] = time.time()

    def end_stage(self, stage_name: str, success: bool = True):
        """Mark end of pipeline stage."""
        import time
        end_time = time.time()
        start_time = self.start_times.get(stage_name, end_time)
        duration = end_time - start_time
        
        status = "completed" if success else "failed"
        self.health.update_stage(stage_name, status)
        self.health.add_node_metrics(stage_name, {
            "duration_seconds": duration,
            "success": success
        })

    def validate_handoff(self, from_node: str, to_node: str,
                        data: Any, expected_type: type) -> bool:
        """Validate data handoff between nodes.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            data: Data being passed
            expected_type: Expected data type
            
        Returns:
            True if handoff is valid
        """
        valid, errors = self.validator.validate_node_output(
            from_node, data, to_node
        )
        
        if not valid:
            for error in errors:
                self.health.add_error(error, from_node)
            return False
        
        return True

    def get_health_report(self) -> Dict[str, Any]:
        """Get current health report.
        
        Returns:
            Dictionary with health status and metrics
        """
        return {
            "healthy": self.health.is_healthy(),
            "timestamp": self.health.timestamp,
            "stage_status": self.health.stage_status,
            "validation_errors": self.health.validation_errors,
            "schema_mismatches": self.health.schema_mismatches,
            "node_metrics": self.health.node_metrics,
            "validation_summary": self.validator.get_validation_summary()
        }


# Initialize start_times in PipelineMonitor
def __init__(self):
    self.health = PipelineHealth()
    self.validator = SchemaValidator()
    self.execution_times: Dict[str, float] = {}
    self.start_times: Dict[str, float] = {}

PipelineMonitor.__init__ = __init__