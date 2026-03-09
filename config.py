"""Configuration constants for the paper-to-notebook tool."""

# --- Provider configuration ---
# Supported providers: "openai", "llama", "uf-navigator"
DEFAULT_PROVIDER = "uf-navigator"

# Default models per provider (user can override in the UI)
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
LLAMA_DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
UF_NAVIGATOR_DEFAULT_MODEL = "gpt-oss-120b"
UF_NAVIGATOR_BASE_URL = "https://api.ai.it.ufl.edu/v1/"

# Legacy Gemini config (commented out)
# DEFAULT_MODEL = "gemini-2.0-flash"

# Token limits per pipeline step (larger for real PyTorch implementations)
MAX_TOKENS_ANALYSIS = 8192
MAX_TOKENS_DESIGN = 8192
MAX_TOKENS_GENERATE = 65536
MAX_TOKENS_VALIDATE = 65536
MAX_TOKENS_FIX = 65536

# Notebook execution
EXECUTE_TIMEOUT = 300  # seconds per cell
MAX_FIX_ATTEMPTS = 2   # max times to retry fixing errors

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]  # seconds

# PDF constraints
MAX_PDF_SIZE_MB = 30
MAX_PDF_PAGES = 100

# Required notebook sections (in order)
REQUIRED_SECTIONS = [
    "Title & Paper Overview",
    "Problem Intuition",
    "Imports & Setup",
    "Dataset & Tokenization",
    "Model Architecture",
    "Loss Function & Training Utilities",
    "Baseline Implementation",
    "Paper's Main Algorithm — Training",
    "Inference / Generation",
    "Full Experiment & Evaluation",
    "Visualizations",
    "Summary & Next Steps",
]
