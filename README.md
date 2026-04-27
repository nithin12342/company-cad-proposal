"""# Logical Knowledge Graph (LKG)

A production-ready implementation of a **Logical Knowledge Graph** for hybrid neuro-symbolic CAD compliance verification. This system processes scanned architectural drawings through a 5-stage pipeline combining neural networks for perception with symbolic algorithms for deterministic reasoning.

## Overview

The LKG maximizes **development speed** and ensures **high-quality output** through four core engineering principles:

1. **Context Engineering**: Structured context propagation across all nodes
2. **Spec-Driven Engineering**: Formal JSON schemas guide design and validation  
3. **Intention Engineering**: Explicit goals and outcomes for each component
4. **Harness Engineering**: Zero-error execution with comprehensive validation

## Architecture

```
Scanned PDF (300 DPI)
    │
    ▼
┌─────────────────┐    ┌───────────────────────┐
│  Node 01        │    │  Engineering Principle │
│  Pixel Triage   │───▶│  Context Engineering   │
│  (Neuro)        │    └───────────────────────┘
│  • SAM/U-Net    │
│  • 3 masks      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌───────────────────────┐
│  Node 02        │    │  Engineering Principle │
│  Geometric      │───▶│  Spec-Driven           │
│  Extraction     │    └───────────────────────┘
│  (Symbolic)     │
│  • OpenCV       │
│  • B-Rep        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌───────────────────────┐
│  Node 03        │───▶│  Intention Engineering │
│  Layout Intel   │    └───────────────────────┘
│  (Neuro-Symbolic)
│  • DocTR OCR    │
│  • Tables       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌───────────────────────┐
│  Node 04        │───▶│  Harness Engineering   │
│  DHMoT Agent    │    └───────────────────────┘
│  (Relational)   │
│  • Hyperedges   │
│  • Walker       │
│  • Ψ collapse   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Node 05        │
│  Oracle         │
│  (Oracle)       │
│  • LLM eval     │
│  • Compliance   │
└────────┬────────┘
         │
         ▼
    JSON Report
```

## Pipeline Stages

### Stage 1: Pixel Triage (Neuro)
- **Algorithm**: Segment Anything Model (SAM) or U-Net
- **Input**: 300 DPI scanned PDF
- **Output**: 3 binary masks (geometry, text, table)
- **Key Feature**: Apache 2.0 licensed models only
- **Validation**: Mask density > 0.1%, dimension matching

### Stage 2: Geometric Extraction (Symbolic)
- **Algorithm**: OpenCV Hough transforms + Canny edge detection
- **Input**: Geometry mask
- **Output**: B-Rep JSON with primitives
- **Key Feature**: 100% deterministic (no neural networks)
- **Validation**: Unique IDs, centroids, bounding boxes

### Stage 3: Layout Intelligence (Neuro-Symbolic)
- **Algorithm**: DocTR/PaddleOCR + structural parsing
- **Input**: Text and table masks
- **Output**: TableSchema JSON with rows/cells
- **Key Feature**: Row integrity, column mapping
- **Validation**: OCR confidence ≥ 0.7, spatial consistency

### Stage 4: DHMoT Agent (Relational)
- **Algorithm**: Hyperedge binding + Walker + Ψ collapse
- **Input**: B-Rep + Tables
- **Output**: Hyperedges, validations, axioms
- **Key Feature**: 
  - ε = 50px distance threshold (epsilon)
  - τ = 2% variance tolerance (tau)
  - Walker re-scan: ±200px window
- **Validation**: Distance, variance, axiom integrity

### Stage 5: Compliance Oracle (Oracle)
- **Algorithm**: LLM (Claude 3.5 Sonnet / Llama 3 70B) + RAG
- **Input**: Axiom manifest
- **Output**: Compliance report JSON
- **Key Feature**: 
  - Trust axioms as absolute truth
  - NO mathematical re-calculation
  - IS 456 & IS 800 standards
- **Validation**: Strict JSON schema, valid status, citations

## Quick Start

### Installation

```bash
# Clone the repository
cd /workspaces/company-cad-proposal

# Install dependencies (if any)
pip install -r requirements.txt

# Or use the provided environment
```

### Running the Pipeline

```python
from src.pipeline.executor import LKGPipeline

# Initialize pipeline
pipeline = LKGPipeline(config={
    "triage": {"model_type": "SAM"},
    "layout": {"model_type": "DocTR"},
    "oracle": {"model": "llama-3-70b"}
})

# Execute on a scanned PDF
results = pipeline.execute_full_pipeline("drawing.pdf")

# Check results
if results["pipeline_status"] == "completed":
    report = results["final_report"]
    print(f"Status: {report['report_summary']['overall_status']}")
    print(f"Violations: {report['report_summary']['violations_found']}")
```

### Command-Line Interface

```bash
# Basic usage
python -m src.cli input.pdf

# With options
python -m src.cli input.pdf \
    --output ./results \
    --model doctr \
    --llama \
    --verbose \
    --save-intermediate
```

## Core Components

### Base Classes

#### `LogicalKnowledgeNode` (src/core/node.py)

Abstract base class for all LKG nodes. Implements the 4 engineering principles:

```python
from src.core.node import LogicalKnowledgeNode

class MyNode(LogicalKnowledgeNode):
    def _build_context(self):
        return BaseNodeContext(
            node_id=self.node_id,
            dependencies=["prev_node"],
            input_schema="PrevSchema_v2.0",
            output_schema="MySchema_v2.0"
        )
    
    def _build_specification(self):
        return BaseNodeSpecification(
            node_type="symbolic",
            algorithm="MyAlgorithm",
            version="1.0",
            constraints={...},
            validation_rules=["R001", "R002"]
        )
    
    def _build_intention(self):
        return BaseNodeIntention(
            primary_goal="My goal",
            expected_outcome="My outcome",
            success_criteria=[...],
            failure_modes=[...]
        )
    
    def _build_harness(self):
        return NodeHarness(
            source_module="src.nodes.mynode",
            entry_function="execute",
            compile_required=False,
            validation_hooks=["validate_output"],
            error_handling="strict"
        )
    
    def execute(self, input_data):
        # Your execution logic
        return NodeOutput(...)
```

### Schemas (src/core/schemas.py)

Type-validated data classes for all pipeline data:

```python
from src.core.schemas import (
    GeometryBRepSchema,  # Node 02 output
    TableSchema,         # Node 03 output
    HyperedgeBinding,    # Node 04: hyperedges
    ValidationResult,    # Node 04: validations
    AxiomManifest,       # Node 04: axioms
    ComplianceReport     # Node 05 output
)

# Example: Creating a geometry primitive
geom = GeometryPrimitive(
    primitive_id="GEO_001",
    primitive_type="rectangle",
    coordinates={"x1": 100, "y1": 200, "x2": 150, "y2": 250},
    centroid=(125, 225),
    properties={"width_px": 50, "height_px": 50}
)

# Validation
valid, errors = geom.validate()
```

### Constants (src/core/constants.py)

Centralized configuration:

```python
from src.core.constants import (
    EPSILON,           # 50px (ε) - distance threshold
    TAU_DIMENSIONAL,   # 2.0% (τ) - variance tolerance
    PIXELS_PER_MM,     # 11.811 at 300 DPI
    DISTANCE_THRESHOLD_PX,
    WALKER_RESCAN_MARGIN
)
```

## Engineering Principles in Practice

### Context Engineering

Each node explicitly declares its context:

```python
def _build_context(self):
    return BaseNodeContext(
        node_id=self.node_id,
        dependencies=["node_01_triage"],  # What we need
        input_schema="GeometryBRep_v2.0",  # What we expect
        output_schema="Hyperedge_v2.0"     # What we produce
    )
```

**Benefits:**
- Clear dependency graph
- Automatic validation at handoffs
- Easy to trace data flow

### Spec-Driven Engineering

Every node has formal specifications:

```python
def _build_specification(self):
    return BaseNodeSpecification(
        node_type="symbolic",
        algorithm="HoughTransform + Canny",
        version="4.5+",
        constraints={
            "deterministic": True,
            "no_ml_involved": True,
            "library": "OpenCV (Apache 2.0)"
        },
        validation_rules=["R001", "R002", "R008"]
    )
```

**Benefits:**
- Self-documenting code
- Automated validation
- Version tracking

### Intention Engineering

Explicit goals and outcomes:

```python
def _build_intention(self):
    return BaseNodeIntention(
        primary_goal="Convert pixels to equations",
        expected_outcome="Complete B-Rep with all shapes",
        success_criteria=[
            "Every primitive has unique ID",
            "All have centroids and bounds"
        ],
        failure_modes=[
            {
                "mode": "low_contrast",
                "mitigation": "Walker re-scan"
            }
        ]
    )
```

**Benefits:**
- Clear success criteria
- Known failure modes
- Easier debugging

### Harness Engineering

Execution with guarantees:

```python
def _build_harness(self):
    return NodeHarness(
        source_module="src.nodes.vectorize",
        entry_function="extract_geometry",
        compile_required=False,
        validation_hooks=[
            "validate_unique_ids",
            "validate_centroids",
            "validate_bounding_boxes"
        ],
        error_handling="strict"  # Must succeed
    )
```

**Benefits:**
- Automated validation
- Error handling strategy
- Zero-error guarantee

## Configuration

Edit `src/core/constants.py` to customize:

```python
# Distance threshold (epsilon - ε)
DISTANCE_THRESHOLD_PX = 50.0  # pixels

# Tolerance threshold (tau - τ)
dimensional_tolerance_pct = 2.0  # percent

# Walker re-scan
WALKER_RESCAN_MARGIN = 200  # pixels

# LLM settings
LLM_TEMPERATURE = 0.0  # deterministic
LLM_MAX_TOKENS = 2000
```

Or override per-pipeline:

```python
pipeline = LKGPipeline(config={
    "dhmot": {
        "epsilon": 75,  # Override default
        "tau": 1.5,
        "walker_rescan_margin": 300
    }
})
```

## Validation System

### Automatic Validation

Every node handoff is validated:

```python
# Node 02 → Node 04 handoff
monitor.validate_handoff(
    from_node="node_02_vectorize",
    to_node="node_04_dhmot",
    data=geometry_brep,
    expected_type=GeometryBRepSchema
)
```

### Schema Validation

All schemas have built-in validation:

```python
brep = GeometryBRepSchema(...)
valid, errors = brep.validate()
if not valid:
    print(f"Validation failed: {errors}")
```

### Health Monitoring

Track pipeline health:

```python
from src.pipeline.validation import PipelineMonitor

monitor = PipelineMonitor()
monitor.start_stage("vectorization")
# ... execute ...
monitor.end_stage("vectorization", success=True)

report = monitor.get_health_report()
print(f"Healthy: {report['healthy']}")
print(f"Errors: {report['validation_errors']}")
```

## Error Codes

| Code | Meaning |
|------|---------|
| 1000 | Schema validation failed |
| 1001 | Threshold exceeded |
| 1002 | Walker re-scan failed |
| 1003 | Model inference failed |
| 1004 | LLM API error |
| 1005 | Schema mismatch |

## Performance Optimizations

### Token Reduction (Ψ Operator)

The DHMoT agent applies semantic collapse to reduce LLM token count:

**Before Ψ:**
- Raw geometry: ~500 bytes/hyperedge
- 50 hyperedges: 25KB

**After Ψ:**
- Axioms: ~150 bytes/axiom
- 50 axioms: 7.5KB

**Savings: 70% reduction** (90%+ including avoided geometry tokens)

### Parallel Execution

Stages 2 (geometry) and 3 (layout) can run in parallel:

```python
# Parallelize independent stages
with concurrent.futures.ThreadPoolExecutor() as executor:
    geo_future = executor.submit(node_02.execute, geometry_mask)
    layout_future = executor.submit(node_03.execute, table_mask, text_mask)
    
    geometry = geo_future.result()
    tables = layout_future.result()
```

## Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Examples

### Basic CAD Compliance

```python
pipeline = LKGPipeline()
results = pipeline.execute_full_pipeline("foundation_drawing.pdf")

report = results["final_report"]
if report["report_summary"]["overall_status"] == "PASS":
    print("✅ Drawing complies with IS codes")
else:
    print("❌ Violations found:")
    for detail in report["compliance_details"]:
        if detail["status"] == "FAIL":
            print(f"  - {detail['checkpoint']}: {detail['comment']}")
```

### Custom Validation

```python
from src.nodes.dhmot import DHMoTNode

# Custom tolerance
dhmot = DHMoTNode(
    node_id="custom_dhmot",
    epsilon=75,      # More lenient distance
    tau=1.0,         # Stricter tolerance
    apply_psi=True
)
```

### Batch Processing

```python
# Process multiple drawings
for pdf in Path("drawings/").glob("*.pdf"):
    results = pipeline.execute_full_pipeline(str(pdf))
    
    # Save report
    report_path = f"reports/{pdf.stem}_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
```

## License

This project uses Apache 2.0 licensed models and libraries:
- Segment Anything Model (SAM) - Apache 2.0
- OpenCV - Apache 2.0
- DocTR - Apache 2.0

LLM APIs may have separate licensing terms.

## Contributing

Contributions should:
1. Follow the 4 engineering principles
2. Include tests with > 90% coverage
3. Maintain type hints
4. Document new nodes/schemas

See `CONTRIBUTING.md` for details.

## References

- [Segment Anything Model](https://ai.facebook.com/blog/segment-anything/)
- [OpenCV](https://opencv.org/)
- [DocTR](https://github.com/mindee/doctr)
- [IS 456:2000](https://www.bis.org.in/)
- [IS 800:2007](https://www.bis.org.in/)

## Support

For issues or questions:
- GitHub Issues: https://github.com/company-cad-proposal/issues
- Documentation: See `docs/` directory