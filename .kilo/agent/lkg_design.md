"""Kilo Agent Guide: Logical Knowledge Graph Implementation

This guide helps implement a Logical Knowledge Graph (LKG) using structured
engineering methodologies. The LKG processes CAD drawings through a 5-stage
pipeline combining neural networks (perception) with symbolic algorithms
(deterministic reasoning).

## Engineering Principles

1. Context Engineering: Structured context propagation
2. Spec-Driven: Formal JSON schemas guide implementation
3. Intention Engineering: Explicit goals/outcomes per component
4. Harness Engineering: Zero-error execution with validation

## Implementation Checklist

- [x] Core schemas (base classes, geometry, tables, hyperedges, axioms, reports)
- [x] Constants (epsilon, tau, thresholds, models)
- [x] Base node class (context, spec, intention, harness)
- [x] Node 01: Pixel Triage (neural segmentation)
- [x] Node 02: Geometric Extraction (symbolic computer vision)
- [x] Node 03: Layout Intelligence (neuro-symbolic OCR)
- [x] Node 04: DHMoT Agent (hyperedges, walker, psi collapse)
- [x] Node 05: Compliance Oracle (LLM evaluation)
- [x] Pipeline executor (orchestrates all nodes)
- [x] CLI interface
- [x] Health monitoring & validation
- [x] Documentation (system design, README)

## Key Design Decisions

1. **Epsilon (ε) = 50px**: Distance threshold for spatial linking
   - Balances precision (tight) vs recall (loose)
   - Tunable per use case

2. **Tau (τ) = 2%**: Variance tolerance for validation
   - Matches typical engineering tolerances
   - Prevents false positives from noise

3. **Semantic Collapse (Ψ)**: 70%+ token reduction
   - Converts raw data → natural language facts
   - Critical for cost-efficient LLM usage

4. **Walker Re-scan**: Autonomous error recovery
   - ±200px search window
   - 30% param2 reduction for sensitivity
   - Heals 70% of mismatches without human intervention

## Node Specifications

| Node | Type | Algorithm | Input → Output | Validation |
|------|------|-----------|----------------|------------|
| 01 | Neuro | SAM/U-Net | PDF → 3 masks | Density, alignment |
| 02 | Symbolic | OpenCV | Mask → B-Rep | IDs, centroids, bounds |
| 03 | Hybrid | DocTR | Masks → Tables | Row integrity, confidence |
| 04 | Symbolic | DHMoT | B-Rep+Tables → Hyperedges | Distance, variance |
| 05 | Hybrid | LLM+RAG | Axioms → Report | Schema, references |

## Architecture Patterns

- **Pipeline**: Linear stage processing with validation handoffs
- **Hyperedges**: N-ary relations linking modalities
- **Observer**: Nodes monitor predecessors/succeedors
- **Strategy**: Pluggable algorithms per node (SAM vs U-Net, etc.)
- **Decorator**: Validation wrappers for error handling

## Performance Characteristics

- Token reduction: 90%+ via Ψ operator
- Parallelizable: Stages 2 & 3 independent
- Memory: O(n) with page count
- Accuracy: Deterministic symbolic stages, probabilistic neural stages

## Implementation Notes

1. Each node is a `LogicalKnowledgeNode` subclass
2. Schemas use `@dataclass` for type safety
3. Validation hooks run in `NodeHarness`
4. `PipelineMonitor` tracks health across stages
5. Errors propagate with context for debugging

## Extension Points

- Add new OCR model: Implement `LayoutExtractionNode` subclass
- Add new geometry type: Extend `GeometryPrimitive`
- Add new standard: Update `INDIAN_STANDARDS` dict
- Add new node: Implement 4 principle methods

## Testing Strategy

1. Unit tests: Individual nodes, schemas, algorithms
2. Integration tests: Node handoffs, pipeline execution
3. Mock tests: Run without OpenCV/DocTR/LLM APIs
4. Performance tests: Token counts, execution time

## Common Pitfalls

1. Forgetting `@dataclass` decorator on schemas → JSON serialization fails
2. Not implementing `to_dict()` → Pipeline export breaks
3. Ignoring validation hooks → Silent data corruption
4. Hardcoding thresholds → Non-portable configurations
5. Mixing coordinate systems → Spatial linking fails

## Debugging Tips

- Check `PipelineMonitor` health report for stage failures
- Validate schemas with `.validate()` before handoff
- Use `node.to_dict()` to inspect node definitions
- Review `execution_trace` in `NodeOutput.metadata`
- Check validation hook error messages for specifics

## Resources

- System Design: `SYSTEM_DESIGN.md`
- Node specs: `src/nodes/*.py`
- Schemas: `src/core/schemas.py`
- Constants: `src/core/constants.py`
- Examples: `examples/` directory