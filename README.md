# Docling Streamlit Converter

Convert local documents to Markdown, HTML, DocTags, and lossless JSON using Docling — all in a simple Streamlit app.

## Features
- Upload common formats: PDF, DOCX, PPTX, XLSX, HTML, images, CSV, VTT, MD
- View and download outputs in:
  - Markdown (`.md`)
  - HTML (`.html`)
  - DocTags (`.doctags.txt`)
  - Lossless JSON (`.json`)
- Fill a custom JSON template using AI (OpenRouter) or a fast heuristic fallback
- Windows-friendly defaults for Hugging Face Hub caching

## Requirements
- Python 3.10+
- Packages in `requirements.txt`

## Quick start (Windows PowerShell)

```powershell
# 1) (Optional) Create a virtual environment
python -m venv .venv ; .venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the printed local URL (e.g., http://localhost:8501) in your browser, upload a file, and explore the outputs.

## AI template filling (OpenRouter)

The Template Fill tab lets you populate your own `master.json`-style template using data extracted by Docling. By default, the app will use AI via OpenRouter if an API key is available; otherwise it falls back to a simple label-based heuristic.

1) Create a local `.env` file (not committed):

```env
OPENROUTER_API_KEY=sk-or-...
# Optional overrides
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# OPENROUTER_MODEL=openrouter/auto
# OPENROUTER_HTTP_REFERER=https://github.com/Unigalactix/Docling-Demo
# OPENROUTER_X_TITLE=Docling Demo
```

2) Restart the app so it can load the `.env`.

3) In the app's Template Fill tab:
- Choose your template source (project `master.json` or upload a custom template)
- Pick the method: "AI (OpenRouter)" or "Heuristic (regex)"
- Download the filled JSON

Behavior and guarantees:
- The AI is instructed to preserve the exact structure and keys of your template
- It replaces only leaf values equal to the literal string `"string"`
- If a value cannot be found in Docling's lossless JSON, it sets an empty string
- It should not invent information; it uses Docling JSON (and plain text as a last resort)

Notes on privacy and cost:
- Using AI sends the document-derived JSON (and a truncated plain-text view) to OpenRouter and the selected model provider. Review your data policies before enabling.
- API usage may incur costs depending on the model you select via `OPENROUTER_MODEL`.

## Notes
- On the first conversion, Docling may download models and OCR assets; this can take a few minutes.
- If you see Windows permission errors or Hugging Face Hub symlink warnings, try one of:
  - Enable Windows Developer Mode (Settings → For developers → Developer Mode), or
  - Run the terminal as Administrator, or
  - Keep the defaults in `app.py` which set `HF_HUB_DISABLE_HARDLINKS=1` to avoid hardlink issues.

## References
- Docling GitHub: https://github.com/docling-project/docling
- Docling Docs: https://docling-project.github.io/docling/
- Docling Website: https://www.docling.ai/