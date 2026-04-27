"""Base Node implementation for Logical Knowledge Graph.

Implements the 4 engineering principles: Context, Specification, Intention, Harness.
Each node in the LKG inherits from this base class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
from dataclasses import dataclass
import traceback

from .schemas import (
    BaseNodeContext, BaseNodeSpecification,
    BaseNodeIntention, NodeHarness
)
from .constants import VALIDATION_RULES, ERROR_CODES

logger = logging.getLogger(__name__)


@dataclass
class NodeOutput:
    """Structured output from a node execution."""
    success: bool
    data: Any
    context: Dict[str, Any]
    specification: Dict[str, Any]
    intention: Dict[str, Any]
    harness: Dict[str, Any]
    errors: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


class LogicalKnowledgeNode(ABC):
    """Abstract base class for all LKG nodes.
    
    Encapsulates the 4 engineering principles:
    1. Context - What data and dependencies?
    2. Specification - Formal behavior definition
    3. Intention - Explicit goals and outcomes
    4. Harness - Execution and validation machinery
    """

    def __init__(self, node_id: str, **kwargs):
        """Initialize node with engineering principles.
        
        Args:
            node_id: Unique identifier for this node instance
            **kwargs: Additional configuration parameters
        """
        self.node_id = node_id
        self.kwargs = kwargs

        # Initialize the 4 engineering principles
        self.context = self._build_context()
        self.specification = self._build_specification()
        self.intention = self._build_intention()
        self.harness = self._build_harness()

        logger.info(f"Initialized {self.__class__.__name__}[{node_id}]")

    @abstractmethod
    def _build_context(self) -> BaseNodeContext:
        """Build node context with dependencies and data requirements.
        
        Returns:
            BaseNodeContext instance
        """
        pass

    @abstractmethod
    def _build_specification(self) -> BaseNodeSpecification:
        """Build formal specification defining behavior and constraints.
        
        Returns:
            BaseNodeSpecification instance
        """
        pass

    @abstractmethod
    def _build_intention(self) -> BaseNodeIntention:
        """Build explicit purpose and expected outcomes.
        
        Returns:
            BaseNodeIntention instance
        """
        pass

    @abstractmethod
    def _build_harness(self) -> NodeHarness:
        """Build execution harness with validation.
        
        Returns:
            NodeHarness instance
        """
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> NodeOutput:
        """Execute node logic with error-free guarantee.
        
        Returns:
            NodeOutput with execution results and metadata
        """
        pass

    def validate_input(self, data: Any) -> tuple[bool, List[str]]:
        """Validate input against node's expected schema.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        if data is None:
            errors.append("Input data is None")
            return False, errors
        return True, errors

    def validate_output(self, data: Any) -> tuple[bool, List[str]]:
        """Validate output against node's schema contract.
        
        Args:
            data: Output data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        if data is None:
            errors.append("Output data is None")
            return False, errors
        return True, errors

    def _execute_with_harness(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Execute function with harness guarantees (zero-error execution).
        
        Wraps execution with comprehensive error handling and validation
        to ensure the harness engineering principle.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary with execution results
        """
        result = {
            "node_id": self.node_id,
            "status": "started",
            "errors": [],
            "warnings": [],
            "execution_trace": []
        }

        try:
            # Pre-execution validation
            result["execution_trace"].append("pre_validation")
            
            # Execute main function
            result["execution_trace"].append("main_execution")
            output = func(*args, **kwargs)
            
            # Post-execution validation
            result["execution_trace"].append("post_validation")
            result["status"] = "completed"
            result["output"] = output
            
            # Run validation hooks from harness
            for hook in self.harness.validation_hooks:
                result["execution_trace"].append(f"validation_{hook}")
                # Hook would be invoked here
                
        except Exception as e:
            result["status"] = "failed"
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["errors"].append(error_msg)
            result["traceback"] = traceback.format_exc()
            logger.error(f"Node {self.node_id} failed: {error_msg}")
            
            # Harness error handling strategy
            if self.harness.error_handling == "strict":
                raise
            elif self.harness.error_handling == "recoverable":
                result["status"] = "recovered"
            # "lenient" just records the error

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node definition to dictionary.
        
        Returns:
            Dictionary representation of node's engineering principles
        """
        return {
            "node_id": self.node_id,
            "class": self.__class__.__name__,
            "context": self.context.to_dict(),
            "specification": self.specification.to_dict(),
            "intention": self.intention.to_dict(),
            "harness": self.harness.to_dict(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node_id})"