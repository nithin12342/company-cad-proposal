"""LKG Usage Examples

This document provides practical examples for using the Logical Knowledge Graph.
"""

import json
from pathlib import Path
from src.pipeline.executor import LKGPipeline
from src.nodes.dhmot import DHMoTNode

# ============================================================================
# QUICK START
# ============================================================================

"""
Process a CAD drawing through the complete pipeline:

1. Pixel Triage: Segment into geometry/text/table masks
2. Geometric Extraction: Convert pixels to B-Rep
3. Layout Intelligence: Extract table data
4. DHMoT Agent: Cross-validate geometry vs. documentation
5. Compliance Oracle: Evaluate against IS codes
"""

pipeline = LKGPipeline()
results = pipeline.execute_full_pipeline("drawing.pdf")

if results["pipeline_status"] == "completed":
    report = results["final_report"]
    print(f"Status: {report['report_summary']['overall_status']}")
    print(f"Violations: {report['report_summary']['violations_found']}")

# ============================================================================
# CUSTOM CONFIGURATION
# ============================================================================

"""
Customize pipeline behavior by overriding defaults:
"""

pipeline = LKGPipeline(config={
    # Node 01: Pixel Triage
    "triage": {
        "model_type": "SAM",  # or "UNet"
    },
    # Node 02: Geometric Extraction (inherits defaults)
    "vectorize": {},
    # Node 03: Layout Intelligence
    "layout": {
        "model_type": "DocTR",  # or "PaddleOCR"
    },
    # Node 04: DHMoT Agent
    "dhmot": {
        "epsilon": 50.0,        # Distance threshold (px)
        "tau": 2.0,             # Variance tolerance (%)
        "walker_rescan_margin": 200,  # Re-scan window (px)
        "apply_psi": True,      # Enable semantic collapse
    },
    # Node 05: Compliance Oracle
    "oracle": {
        "model": "llama-3-70b",  # or "claude-3-5-sonnet-20241022"
        "api_key": None,         # API key (None = mock)
    },
})

results = pipeline.execute_full_pipeline("drawing.pdf")

# ============================================================================
# ACCESSING INTERMEDIATE RESULTS
# ============================================================================

"""
Each node's output is available in results['nodes']:
"""

triage_result = results["nodes"]["triage"]
print(f"Triage successful: {triage_result['success']}")
print(f"Processing time: {triage_result['metadata']['processing_time_ms']}ms")

geometry_result = results["nodes"]["vectorize"]
print(f"Primitives found: {geometry_result['metadata']['primitive_count']}")

layout_result = results["nodes"]["layout"]
print(f"Tables found: {layout_result['metadata']['table_count']}")

dhmot_result = results["nodes"]["dhmot"]
print(f"Hyperedges formed: {dhmot_result['metadata']['hyperedge_count']}")
print(f"Validations performed: {dhmot_result['metadata']['validation_count']}")
print(f"Axioms generated: {dhmot_result['metadata']['axiom_count']}")

# ============================================================================
# DHMoT AGENT - DIRECT USAGE
# ============================================================================

"""
Use DHMoT node independently for cross-modal validation:
"""

from src.nodes.dhmot import DHMoTNode
from src.core.schemas import GeometryBRepSchema, TableSchema

# Create geometry (in practice: from Node 02 output)
from src.core.schemas import GeometryPrimitive

geom = GeometryPrimitive(
    primitive_id="GEO_001",
    primitive_type="rectangle",
    coordinates={"x1": 100, "y1": 200, "x2": 150, "y2": 250},
    centroid=(125, 225),
    properties={"width_px": 50, "height_px": 50, "area_px": 2500}
)

geometry = GeometryBRepSchema(page_number=1, geometries=[geom])

# Create table (in practice: from Node 03 output)
from src.core.schemas import TableRow, TableCell

cell = TableCell(column="Size", text="400x400", bbox=[110, 60, 200, 80])
row = TableRow(row_index=0, cells=[cell])
table = TableSchema(
    table_id="TBL_001",
    page_number=1,
    bounding_box=[50, 50, 200, 200],
    rows=[row]
)

# Create and run DHMoT agent
dhmot = DHMoTNode(
    node_id="custom_dhmot",
    epsilon=50,      # Distance threshold
    tau=2.0,         # Variance tolerance (%)
    apply_psi=True
)

result = dhmot.execute(geometry, [table])

print(f"Hyperedges: {len(result.data['hyperedges'])}")
print(f"Validations: {len(result.data['validations'])}")
print(f"Axioms: {len(result.data['axioms'])}")

for val in result.data["validations"]:
    print(f"  Status: {val['status']}")
    print(f"  Variance: {val['variance_pct']}%")

# ============================================================================
# SAVING AND LOADING
# ============================================================================

"""
Save pipeline output to JSON file:
"""

if results["pipeline_status"] == "completed":
    output_path = Path("output/compliance_report.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Report saved to {output_path}")

# ============================================================================
# BATCH PROCESSING
# ============================================================================

"""
Process multiple drawings:
"""

from concurrent.futures import ThreadPoolExecutor

def process_drawing(pdf_path: str) -> dict:
    """Process single drawing."""
    pipeline = LKGPipeline()
    return pipeline.execute_full_pipeline(pdf_path)

pdf_files = ["drawing1.pdf", "drawing2.pdf", "drawing3.pdf"]

# Sequential processing
all_results = {}
for pdf in pdf_files:
    all_results[pdf] = process_drawing(pdf)

# Parallel processing (faster)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_drawing, pdf): pdf for pdf in pdf_files}
    all_results = {futures[f]: f.result() for f in futures}

# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

"""
Validate schemas before pipeline execution:
"""

from src.pipeline.validation import SchemaValidator
from src.core.schemas import GeometryBRepSchema

validator = SchemaValidator()

# Validate node output
valid, errors = validator.validate_node_output(
    from_node="node_02",
    to_node="node_04",
    data=geometry,
    expected_type=GeometryBRepSchema
)

if not valid:
    print(f"Schema validation failed: {errors}")

# Get validation summary
summary = validator.get_validation_summary()
print(f"Success rate: {summary['success_rate']:.1%}")

# ============================================================================
# ERROR HANDLING
# ============================================================================

"""
Handle pipeline errors gracefully:
"""

try:
    results = pipeline.execute_full_pipeline("missing_file.pdf")
except Exception as e:
    print(f"Pipeline execution failed: {e}")

# Check results for partial completion
if results["pipeline_status"] in ("failed", "error"):
    for error in results["errors"]:
        print(f"Error: {error}")
        # Log error, send alert, etc.

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Common issues and solutions:

1. "ModuleNotFoundError: No module named 'nodes'"
   → Ensure running from workspace root, not src/

2. "NameError: name 'PIXELS_PER_MM' is not defined"
   → Check all imports in node files are correct

3. Pipeline fails at Node 01/02
   → Verify OpenCV/Dependencies installed

4. Pipeline fails at Node 03
   → DocTR/PaddleOCR not available (uses mock in demo)

5. Pipeline fails at Node 05
   → LLM API key not set (uses mock if None)
"""