# Docling Streamlit Converter

Convert local documents to Markdown, HTML, DocTags, and lossless JSON using Docling — all in a simple Streamlit app.

## Features
- Upload common formats: PDF, DOCX, PPTX, XLSX, HTML, images, CSV, VTT, MD
- View and download outputs in:
  - Markdown (`.md`)
  - HTML (`.html`)
  - DocTags (`.doctags.txt`)
  - Lossless JSON (`.json`)
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