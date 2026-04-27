"""CLI for Logical Knowledge Graph System.

Provides command-line interface for running the LKG pipeline
on scanned CAD documents.
"""

import argparse
import json
import logging
from pathlib import Path

from .pipeline.executor import LKGPipeline


def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Logical Knowledge Graph - CAD Compliance Pipeline"
    )
    parser.add_argument(
        "input",
        help="Path to scanned PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results (default: ./output)",
        default="./output"
    )
    parser.add_argument(
        "--model",
        help="OCR model to use (doctr|paddle) [default: doctr]",
        choices=["doctr", "paddle"],
        default="doctr"
    )
    parser.add_argument(
        "--llm",
        help="LLM model for compliance (claude|llama) [default: llama]",
        choices=["claude", "llama"],
        default="llama"
    )
    parser.add_argument(
        "--api-key",
        help="LLM API key (optional, uses mock if not provided)"
    )
    parser.add_argument(
        "--verbose", "-v",
        help="Enable verbose logging",
        action="store_true"
    )
    parser.add_argument(
        "--save-intermediate",
        help="Save intermediate node outputs",
        action="store_true"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build pipeline configuration
    config = {
        "triage": {"model_type": "SAM"},
        "layout": {"model_type": args.model.upper()},
        "oracle": {
            "model": "claude-3-5-sonnet-20241022" if args.llm == "claude"
                     else "llama-3-70b",
            "api_key": args.api_key
        }
    }

    print("="*60)
    print("Logical Knowledge Graph - CAD Compliance Pipeline")
    print("="*60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"OCR:    {args.model.upper()}")
    print(f"LLM:    {args.llm.upper()}")
    print("="*60)

    # Run pipeline
    pipeline = LKGPipeline(config=config)
    results = pipeline.execute_full_pipeline(str(input_path))

    # Save results
    report_path = output_dir / "compliance_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")

    # Print summary
    if results["pipeline_status"] == "completed":
        report = results["final_report"]
        print("\n" + "="*60)
        print("COMPLIANCE REPORT SUMMARY")
        print("="*60)
        print(f"Overall Status: {report['report_summary']['overall_status']}")
        print(f"Total Checks:   {report['report_summary']['total_checks']}")
        print(f"Violations:     {report['report_summary']['violations_found']}")
        print("\nDetails:")
        for detail in report["compliance_details"]:
            status_icon = "✓" if detail["status"] == "PASS" else "✗"
            print(f"  {status_icon} {detail['checkpoint']}")
            print(f"    {detail['comment']}")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("PIPELINE FAILED")
        print("="*60)
        for error in results["errors"]:
            print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    exit(main())