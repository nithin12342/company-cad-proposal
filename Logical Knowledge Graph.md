# Logical Knowledge Graph: Hybrid Neuro-Symbolic CAD Compliance System

## Node 01: Pixel Triage (Neuro Layer)
* **Context:** Input: 300 DPI Scanned PDF. Constraints: Apache 2.0 models only. Environment: Python 3.10+ / Torch.
* **Spec-Driven:** Model: Segment Anything (SAM) or U-Net. Output: 3 distinct binary masks (Geometry, Text, Table) as PNGs.
* **Intention:** Isolate domain-specific pixel clusters to eliminate physical scanner entropy before mathematical processing.
* **Harness:** `python src/triage.py --input drawings.pdf` | Verification: Assert mask pixel density > 0.1% to prevent blank output.

## Node 02: Geometric Extraction (Symbolic Layer)
* **Context:** Input: Geometry Mask. Domain: Euclidean Geometry. Library: OpenCV 4.5+.
* **Spec-Driven:** Algorithm: Hough Line/Circle Transform, Canny Edge Detection. Output: JSON B-Rep (Boundary Representation).
* **Intention:** Convert raw pixel data into 100% deterministic mathematical equations (Lines, Circles, Arcs).
* **Harness:** `python src/vectorize.py --mask ./masks/geometry.png` | Verification: Assert every primitive has a unique ID, centroid, and bounding box.

## Node 03: Layout Intelligence (Neuro-Symbolic)
* **Context:** Input: Text & Table Masks. Model: DocTR or PaddleOCR.
* **Spec-Driven:** JSON Schema: Row/Cell hierarchy with physical bounding boxes mapped to 300 DPI coordinates.
* **Intention:** Reconstruct structural relationships and engineering schedules from visual grid patterns.
* **Harness:** `python src/parse_tables.py --mask ./masks/tables.png` | Verification: Map 'Mark' columns to 'Size' cells with 100% row integrity.

## Node 04: DHMoT Agent (Manifold Logic)
* **Context:** Input: Node 02 & Node 03 JSON. Linking Threshold (ε): 50px distance.
* **Spec-Driven:** Hyperedge formation. Variance Tolerance (τ): ±2.0%. Walker Logic: Re-scan window = ±200px.
* **Intention:** Implement self-healing cross-modal validation. Verify if the drawing matches the text documentation.
* **Harness:** `python src/dhmot_engine.py` | Verification: If mismatch detected, trigger `walker_rescan()`. Execute Ψ (Collapse) on success.

## Node 05: Regulatory Oracle (Oracle Layer)
* **Context:** Input: Axiom Manifest. Knowledge: IS 456, IS 800 via RAG.
* **Spec-Driven:** API: Claude 3.5 Sonnet or Llama 3 70B. Schema: Pydantic Compliance Report.
* **Intention:** Apply linguistic reasoning to verify axioms against building codes.
* **Harness:** `curl -X POST api.oracle.v1/report` | Verification: Assert JSON output contains 'Status' and 'Regulatory_Reference'.

## Global Guardrails
* **Privacy:** All data remains local until Phase 5 to minimize cloud egress.
* **Reliability:** Strict enforcement of v1.2 JSON schemas between all nodes.
* **Efficiency:** Semantic Collapse (Ψ) must reduce token footprint by >90% before Oracle call.