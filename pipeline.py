"""Multi-step pipeline: PDF → analysis → design → cells → validated notebook."""

import json
import sys

from config import (
    OPENAI_DEFAULT_MODEL,
    MAX_TOKENS_ANALYSIS,
    MAX_TOKENS_DESIGN,
    MAX_TOKENS_GENERATE,
    MAX_TOKENS_VALIDATE,
)
from llm import call_llm_with_retry, load_pdf_as_text, parse_llm_json
from notebook_builder import build_notebook, save_notebook
from prompts import (
    ANALYSIS_PROMPT,
    DESIGN_PROMPT_TEMPLATE,
    GENERATE_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    VALIDATE_PROMPT_TEMPLATE,
)


def _print_step(step_num: int, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {step_num}/4: {name}")
    print(f"{'='*60}")


def run_pipeline(
    pdf_path: str,
    output_path: str,
    model: str = OPENAI_DEFAULT_MODEL,
    verbose: bool = False,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Run the full paper-to-notebook pipeline.

    Args:
        pdf_path: Path to the research paper PDF.
        output_path: Where to save the generated .ipynb.
        model: Model ID to use.
        verbose: If True, print intermediate JSON outputs.
        provider: LLM provider ('openai' or 'llama').
        base_url: Custom base URL for Llama-compatible providers.
        api_key: API key for the provider.
    """
    # Load PDF as text (instead of Gemini's native PDF upload)
    print(f"Loading PDF: {pdf_path}")
    pdf_text = load_pdf_as_text(pdf_path)

    # Common kwargs for all LLM calls
    llm_kwargs = dict(
        model=model,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
    )

    # ------------------------------------------------------------------
    # Step 1: Paper Analysis
    # ------------------------------------------------------------------
    _print_step(1, "Analyzing paper")
    print("  Extracting title, algorithms, baselines, metrics...")

    analysis_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[pdf_text, ANALYSIS_PROMPT],
        max_tokens=MAX_TOKENS_ANALYSIS,
        **llm_kwargs,
    )

    analysis = parse_llm_json(analysis_raw, "paper_analysis", model, api_key=api_key,
                              provider=provider, base_url=base_url)
    title = analysis.get("title", "Unknown Paper")
    num_algos = len(analysis.get("algorithms", []))
    print(f"  Paper: {title}")
    print(f"  Found {num_algos} algorithm(s)")

    if verbose:
        print("\n  --- Analysis JSON ---")
        print(json.dumps(analysis, indent=2))

    # ------------------------------------------------------------------
    # Step 2: Toy Design Plan
    # ------------------------------------------------------------------
    _print_step(2, "Designing toy implementation")
    print("  Planning synthetic data, mock models, experiment loop...")

    design_prompt = DESIGN_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2)
    )

    design_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[pdf_text, design_prompt],
        max_tokens=MAX_TOKENS_DESIGN,
        **llm_kwargs,
    )

    design = parse_llm_json(design_raw, "toy_design", model, api_key=api_key,
                            provider=provider, base_url=base_url)
    num_mocks = len(design.get("mock_models", []))
    num_viz = len(design.get("visualizations", []))
    print(f"  Mock components: {num_mocks}")
    print(f"  Planned visualizations: {num_viz}")

    if verbose:
        print("\n  --- Design JSON ---")
        print(json.dumps(design, indent=2))

    # ------------------------------------------------------------------
    # Step 3: Generate Notebook Cells
    # ------------------------------------------------------------------
    _print_step(3, "Generating notebook cells")
    print("  Writing code and markdown for all 11 sections...")

    generate_prompt = GENERATE_PROMPT_TEMPLATE.format(
        analysis_json=json.dumps(analysis, indent=2),
        design_json=json.dumps(design, indent=2),
    )

    cells_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[pdf_text, generate_prompt],
        max_tokens=MAX_TOKENS_GENERATE,
        **llm_kwargs,
    )

    cells = parse_llm_json(cells_raw, "generate_cells", model, api_key=api_key,
                           provider=provider, base_url=base_url)
    num_cells = len(cells)
    code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    md_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")
    print(f"  Generated {num_cells} cells ({code_cells} code, {md_cells} markdown)")

    if verbose:
        print("\n  --- First 3 cells ---")
        for c in cells[:3]:
            ctype = c.get("cell_type", "?")
            src = c.get("source", "")[:200]
            print(f"  [{ctype}] {src}...")

    # ------------------------------------------------------------------
    # Step 4: Validate & Repair
    # ------------------------------------------------------------------
    _print_step(4, "Validating notebook")
    print("  Checking for undefined variables, missing imports, placeholders...")

    validate_prompt = VALIDATE_PROMPT_TEMPLATE.format(
        cells_json=json.dumps(cells, indent=2)
    )

    validated_raw = call_llm_with_retry(
        system_prompt=SYSTEM_PROMPT,
        user_content=[validate_prompt],
        max_tokens=MAX_TOKENS_VALIDATE,
        **llm_kwargs,
    )

    validated_cells = parse_llm_json(validated_raw, "validate", model, api_key=api_key,
                                     provider=provider, base_url=base_url)
    num_validated = len(validated_cells)
    print(f"  Validated: {num_validated} cells")

    # ------------------------------------------------------------------
    # Build and save notebook
    # ------------------------------------------------------------------
    print(f"\nBuilding notebook...")
    nb = build_notebook(validated_cells)
    save_notebook(nb, output_path)
    print(f"Saved to: {output_path}")
