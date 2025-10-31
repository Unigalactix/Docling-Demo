import os
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

# Robust HF Hub settings for Windows (avoid symlink/hardlink issues)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_HARDLINKS", "1")

from dotenv import load_dotenv
import requests
import streamlit as st
from docling.document_converter import DocumentConverter, InputFormat


st.set_page_config(page_title="Docling Converter", layout="wide")


@st.cache_resource(show_spinner=False)
def get_converter():
    # Initialize once and reuse across reruns
    return DocumentConverter()


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def convert_file(path: Path):
    converter = get_converter()
    result = converter.convert(str(path))
    return result


def doc_to_outputs(doc):
    # Build outputs in four formats
    md = doc.export_to_markdown()
    html = doc.export_to_html()
    doctags = doc.export_to_doctags()
    # Lossless JSON via export_to_dict
    js = doc.export_to_dict()
    json_text = json.dumps(js, ensure_ascii=False, indent=2)
    return md, html, doctags, json_text


def _truncate(text: str, max_chars: int = 120_000) -> str:
    if text is None:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n... [truncated]"


def _openrouter_endpoint() -> str:
    base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return base.rstrip("/") + "/chat/completions"


def _openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openrouter/auto")


def _openrouter_headers() -> Dict[str, str]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    headers = {
        "Content-Type": "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    # Optional but recommended headers per OpenRouter docs
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.getenv("OPENROUTER_X_TITLE", "")
    if title:
        headers["X-Title"] = title
    return headers


def ai_fill_template_via_openrouter_from_markdown(template: Dict[str, Any], markdown_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Use an LLM (via OpenRouter) to map values from the document's Markdown into the template structure.
    Rules:
      - Preserve all keys and nesting from the template.
      - Replace any leaf value equal to the literal string "string" OR null with best-matched value from the Markdown.
      - If not found, set the value to null.
      - Do not invent content not present in the Markdown; prefer exact strings/numbers.
    """

    endpoint = _openrouter_endpoint()
    model = _openrouter_model()
    headers = _openrouter_headers()
    if "Authorization" not in headers:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment/.env")

    # Prepare compact inputs for the prompt
    tpl_str = json.dumps(template, ensure_ascii=False, indent=2)
    md_str = _truncate(markdown_text or "")

    system_prompt = (
        "You are a precise JSON transformation assistant. "
        "Your job is to fill a target JSON template using ONLY values present in a provided MARKDOWN content derived from the document. "
        "Preserve every key and structure from the template. Replace only leaf values equal to the exact string 'string' or null. "
        "If a value cannot be determined from the markdown, set it to null. "
        "Do not hallucinate or invent values. Return strictly valid JSON with no commentary."
    )

    user_prompt = (
        "TARGET TEMPLATE (preserve structure, replace 'string' or null leaves):\n" +
        f"```json\n{tpl_str}\n```\n\n" +
        "SOURCE MARKDOWN (extracted by Docling; use exact values only):\n" +
        f"```markdown\n{md_str}\n```\n\n" +
        "Return ONLY the filled JSON object."
    )

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()

    # OpenRouter returns OpenAI-compatible schema
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Empty response from AI")

    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON between code fences
    fence_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
    if fence_match:
        fenced = fence_match.group(1)
        return json.loads(fenced)

    # Last attempt: find first JSON object substring
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return json.loads(content[brace_start:brace_end+1])

    raise RuntimeError("AI response was not valid JSON")


def camel_to_words(name: str) -> str:
    # Turn CamelCase or PascalCase into spaced words
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


def extract_value_from_text(label: str, text: str) -> str | None:
    # Try several label variants and capture the value after separators
    candidates = [label, camel_to_words(label)]
    # De-duplicate and keep order
    seen = set()
    variants = []
    for c in candidates:
        c_norm = c.strip()
        if c_norm.lower() not in seen:
            variants.append(c_norm)
            seen.add(c_norm.lower())

    # Patterns: Label : value | Label - value | Label = value
    for v in variants:
        # Escape special regex chars in label
        v_esc = re.escape(v)
        for sep in [":", "-", "="]:
            pattern = rf"(?i)\b{v_esc}\b\s*{re.escape(sep)}\s*([^\r\n|]+)"
            m = re.search(pattern, text)
            if m:
                val = m.group(1).strip()
                # Cleanup trailing artifacts
                val = re.sub(r"\s+$", "", val)
                val = val.strip(" ")
                return val
    return None


def fill_template_from_text(obj, text: str):
    # Recursively fill any leaf value equal to the literal string "string" OR null.
    # If a value cannot be extracted, keep null (None) instead of an empty string.
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            is_placeholder = (isinstance(v, str) and v == "string") or (v is None)
            if is_placeholder:
                found = extract_value_from_text(k, text)
                out[k] = found if found is not None else None
            else:
                out[k] = fill_template_from_text(v, text)
        return out
    elif isinstance(obj, list):
        return [fill_template_from_text(it, text) for it in obj]
    else:
        # Keep original for numbers, dates, other strings
        return obj


st.title("📄 Docling Streamlit Converter")
st.caption(
    "Upload a document (PDF, DOCX, PPTX, XLSX, HTML, images, etc.) and export to Markdown, HTML, DocTags, or lossless JSON."
)

# Load environment variables from .env (e.g., OPENROUTER_API_KEY)
load_dotenv()

with st.expander("Advanced (optional)"):
    col1, col2 = st.columns(2)
    with col1:
        max_pages = st.number_input("Max pages to process (0 = no limit)", min_value=0, value=0, step=1)
    with col2:
        page_range = st.text_input("Page range (e.g. 1-3,5). Leave empty for all.", value="")
    st.caption("Leave defaults if unsure. Advanced options are applied best-effort.")

uploaded = st.file_uploader(
    "Choose a file",
    type=[
        "pdf", "docx", "pptx", "xlsx", "html", "htm", "md",
        "png", "jpg", "jpeg", "tif", "tiff", "csv", "vtt"
    ],
)

if uploaded is not None:
    tmp_path = save_upload_to_temp(uploaded)
    st.info(f"Saved upload to temporary file: {tmp_path.name}")

    with st.spinner("Converting with Docling… (downloads models on first run)"):
        try:
            # Apply simple page controls if provided (Docling supports page_range in convert())
            # For simplicity, we call convert() without custom options; advanced tuning can be added later.
            result = convert_file(tmp_path)
        except Exception as e:
            st.error(f"Conversion failed: {type(e).__name__}: {e}")
            st.stop()

    if result.errors:
        with st.expander("Warnings/Errors during conversion"):
            for err in result.errors:
                st.write(str(err))

    doc = result.document
    st.success("Conversion complete.")
    st.write(f"Pages detected: {len(result.pages)} | Status: {result.status}")

    md, html, doctags, json_text = doc_to_outputs(doc)

    tabs = st.tabs(["Markdown", "HTML", "DocTags", "JSON", "Template Fill"])

    with tabs[0]:
        st.download_button(
            "Download .md",
            data=md.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}.md",
            mime="text/markdown",
        )
        st.markdown(md)

    with tabs[1]:
        st.download_button(
            "Download .html",
            data=html.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}.html",
            mime="text/html",
        )
        # Render a preview inside an iframe-like container
        st.components.v1.html(html, height=600, scrolling=True)

    with tabs[2]:
        st.download_button(
            "Download .doctags.txt",
            data=doctags.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}.doctags.txt",
            mime="text/plain",
        )
        st.text_area("DocTags preview", doctags, height=300)

    with tabs[3]:
        st.download_button(
            "Download .json",
            data=json_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}.json",
            mime="application/json",
        )
        st.code(json_text, language="json")

    # Template-based JSON filling (AI via OpenRouter, with heuristic fallback)
    with tabs[4]:
        st.subheader("Fill JSON from Template (master.json)")

        # Template source selection
        template_choice = st.radio(
            "Template source",
            ["Use project master.json", "Upload custom template"],
            horizontal=True,
        )
        template_bytes = None
        template_path = Path(__file__).resolve().parent / "master.json"
        if template_choice == "Use project master.json":
            if not template_path.exists():
                st.error("master.json not found in project directory.")
                st.stop()
            template_bytes = template_path.read_bytes()
        else:
            t_upload = st.file_uploader("Upload a JSON template", type=["json"], key="tpl_upload")
            if t_upload is None:
                st.info("Upload a template JSON to proceed.")
                st.stop()
            template_bytes = t_upload.getvalue()

        try:
            template = json.loads(template_bytes.decode("utf-8"))
        except Exception as e:
            st.error(f"Invalid template JSON: {e}")
            st.stop()

        # Prepare source: use generated Markdown for extraction
        md_source = md

        method_options = ["AI (OpenRouter)", "Heuristic (regex)"]
        has_key = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
        default_index = 0 if has_key else 1
        method = st.radio("Filling method", method_options, index=default_index, horizontal=True)

        if method == "AI (OpenRouter)":
            if not has_key:
                st.warning("OPENROUTER_API_KEY not set. Using heuristic fallback.")
                filled = fill_template_from_text(template, md_source)
            else:
                with st.spinner("Asking AI to map Markdown to template via OpenRouter…"):
                    try:
                        filled = ai_fill_template_via_openrouter_from_markdown(template, md_source)
                    except Exception as e:
                        st.error(f"AI filling failed: {type(e).__name__}: {e}")
                        st.info("Falling back to heuristic method.")
                        filled = fill_template_from_text(template, md_source)
        else:
            filled = fill_template_from_text(template, md_source)

        filled_text = json.dumps(filled, ensure_ascii=False, indent=2)

        st.download_button(
            "Download filled template .json",
            data=filled_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_filled.json",
            mime="application/json",
        )
        st.code(filled_text, language="json")

    st.caption(
        "Tip: If you see Windows cache warnings or permission errors from Hugging Face Hub, enable Windows Developer Mode or run Streamlit as Administrator."
    )
else:
    st.info("Upload a file to begin.")
