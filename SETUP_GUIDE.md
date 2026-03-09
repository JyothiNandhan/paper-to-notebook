# Paper to Notebook — Setup & Run Guide

## Prerequisites
- Python 3.10+ installed
- A PDF research paper to convert
- An API key from one of:
  - **UF Navigator** (GPT OSS) — get yours at https://navigator.ai.it.ufl.edu
  - **OpenAI** — https://platform.openai.com/api-keys
  - **Llama provider** (Together AI, Groq, etc.)

---

## Step 1: Open Terminal & Navigate to the Project

```bash
cd ~/Downloads/paper\ to\ code
```

## Step 2: Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

## Step 3: Install Dependencies (first time only)

```bash
pip install -r requirements.web.txt
```

## Step 4: Start the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

## Step 5: Open the Web UI

Open your browser and go to:
```
http://localhost:8000
```

## Step 6: Configure & Generate

1. **Provider** — Select your LLM provider (default: UF Navigator)
2. **Model** — Leave blank for default, or type a model name:
   - UF Navigator: `gpt-oss-120b` or `gpt-oss-20b`
   - OpenAI: `gpt-4o-mini`, `gpt-4o`, etc.
3. **API Key** — Paste your API key
4. **Upload PDF** — Drop or browse for your research paper
5. Click **Generate Notebook**

---

## Quick Start (Copy-Paste)

Run all these commands in order:

```bash
cd ~/Downloads/paper\ to\ code
source venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000 in your browser.

---

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

## Deactivating the Virtual Environment

```bash
deactivate
```

---

## CLI Usage (Alternative to Web UI)

You can also generate notebooks from the command line:

```bash
cd ~/Downloads/paper\ to\ code
source venv/bin/activate
export OPENAI_API_KEY="your-api-key-here"
python generate_notebook.py paper.pdf --provider uf-navigator --model gpt-oss-120b
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o output.ipynb` | Output file name | `<paper>_notebook.ipynb` |
| `--provider` | `openai`, `llama`, or `uf-navigator` | `uf-navigator` |
| `--model` | Model ID | Provider default |
| `--base-url` | Custom API endpoint (for Llama) | Provider default |
| `--verbose` | Print pipeline details | Off |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `command not found: uvicorn` | Make sure venv is activated: `source venv/bin/activate` |
| `Address already in use` | Kill old server: `kill $(lsof -t -i:8000)` then restart |
| 401 Invalid API key | Double-check your key; clear browser cache with Ctrl+Shift+R |
| UI looks outdated | Hard refresh: Ctrl+Shift+R, or clear localStorage in DevTools |
| Connection timeout | UF Navigator requires campus network/VPN |
