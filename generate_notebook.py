#!/usr/bin/env python3
"""
Generate an educational Jupyter notebook from a research paper PDF.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_notebook.py paper.pdf
    python generate_notebook.py paper.pdf -o my_notebook.ipynb
    python generate_notebook.py paper.pdf --provider llama --model meta-llama/Meta-Llama-3.1-70B-Instruct --base-url https://api.together.xyz/v1
"""

import argparse
import os
import sys
from pathlib import Path

from config import OPENAI_DEFAULT_MODEL, LLAMA_DEFAULT_MODEL, MAX_PDF_SIZE_MB, DEFAULT_PROVIDER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a toy-implementation Jupyter notebook from a research paper PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_notebook.py paper.pdf
  python generate_notebook.py paper.pdf -o output.ipynb
  python generate_notebook.py paper.pdf --provider openai --model gpt-4o-mini
  python generate_notebook.py paper.pdf --provider llama --base-url https://api.together.xyz/v1
        """,
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the research paper PDF file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output notebook path (default: <pdf_stem>_notebook.ipynb)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=["openai", "llama"],
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model ID (default: {OPENAI_DEFAULT_MODEL} for openai, {LLAMA_DEFAULT_MODEL} for llama)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for Llama-compatible providers (e.g., https://api.together.xyz/v1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate pipeline outputs to console",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: File must be a PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Check file size
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_PDF_SIZE_MB:
        print(
            f"ERROR: PDF is {file_size_mb:.1f}MB. Max supported is {MAX_PDF_SIZE_MB}MB.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: API key not set. Set OPENAI_API_KEY or LLM_API_KEY environment variable.", file=sys.stderr)
        print("  export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        sys.exit(1)

    # Determine model
    provider = args.provider
    model = args.model
    if not model:
        model = OPENAI_DEFAULT_MODEL if provider == "openai" else LLAMA_DEFAULT_MODEL

    # Determine output path
    output_path = args.output or f"{pdf_path.stem}_notebook.ipynb"

    print(f"Research Paper -> Toy Implementation Notebook")
    print(f"  Input:    {pdf_path}")
    print(f"  Output:   {output_path}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    if args.base_url:
        print(f"  Base URL: {args.base_url}")

    # Run pipeline
    from pipeline import run_pipeline

    run_pipeline(
        pdf_path=str(pdf_path),
        output_path=output_path,
        model=model,
        verbose=args.verbose,
        provider=provider,
        base_url=args.base_url,
        api_key=api_key,
    )

    print(f"\nDone! Open with: jupyter notebook {output_path}")


if __name__ == "__main__":
    main()
