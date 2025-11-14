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


def convert_file(path: Path, page_range: str | None = "", max_pages: int = 0):
    converter = get_converter()
    kwargs: Dict[str, Any] = {}
    if page_range:
        kwargs["page_range"] = page_range
    if isinstance(max_pages, int) and max_pages > 0:
        kwargs["max_pages"] = max_pages

    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            try:
                return converter.convert(str(path), input_format=InputFormat.PDF, **kwargs)
            except TypeError:
                # Older versions may not accept input_format; fall back
                pass
        return converter.convert(str(path), **kwargs)
    except Exception:
        # Retry once with no kwargs as a safety net
        return converter.convert(str(path))


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


def _chat_system_prompt() -> str:
    return (
        "You are a precise assistant for question answering grounded ONLY in the provided DOCUMENT CONTEXT. "
        "Use the context verbatim; if the answer is not present, say you don't know. "
        "Cite short quotes from the context when helpful. Be concise and accurate."
    )


def ai_answer_from_context_via_openai(context_md: str, question: str, temperature: float = 0.0) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    sys_prompt = _chat_system_prompt()
    ctx = _truncate(context_md or "")
    user = (
        "DOCUMENT CONTEXT (Markdown):\n" +
        f"```markdown\n{ctx}\n```\n\n" +
        f"QUESTION: {question}\n\n" +
        "Answer strictly using the context above."
    )
    payload = {
        "model": openai_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Empty response from OpenAI")
    return content


def ai_answer_from_context_via_openrouter(context_md: str, question: str, temperature: float = 0.0) -> str:
    endpoint = _openrouter_endpoint()
    model = _openrouter_model()
    headers = _openrouter_headers()
    if "Authorization" not in headers:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment/.env")
    sys_prompt = _chat_system_prompt()
    ctx = _truncate(context_md or "")
    user = (
        "DOCUMENT CONTEXT (Markdown):\n" +
        f"```markdown\n{ctx}\n```\n\n" +
        f"QUESTION: {question}\n\n" +
        "Answer strictly using the context above."
    )
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def ai_chat_answer_with_fallback(context_md: str, question: str, temperature: float = 0.0) -> tuple[str, str]:
    try:
        return ai_answer_from_context_via_openai(context_md, question, temperature), "openai"
    except Exception as e:
        print(f"OpenAI chat failed: {e}; trying OpenRouter…")
        return ai_answer_from_context_via_openrouter(context_md, question, temperature), "openrouter"


def ai_fill_template_via_openai_from_markdown(template: Dict[str, Any], markdown_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Use OpenAI API to fill template from markdown. Returns filled template or raises Exception.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
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
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    payload = {
        "model": openai_model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Empty response from OpenAI")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    fence_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", content)
    if fence_match:
        fenced = fence_match.group(1)
        return json.loads(fenced)
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return json.loads(content[brace_start:brace_end+1])
    raise RuntimeError("OpenAI response was not valid JSON")

def ai_fill_template_with_fallback(template: Dict[str, Any], markdown_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Maintained for backward-compat: returns only the filled JSON.
    """
    filled, _ = ai_fill_template_with_fallback_and_provider(template, markdown_text, temperature)
    return filled


def ai_fill_template_with_fallback_and_provider(template: Dict[str, Any], markdown_text: str, temperature: float = 0.0) -> tuple[Dict[str, Any], str]:
    """
    Try OpenAI first, fallback to OpenRouter; returns (filled_json, provider_used).
    provider_used in {"openai", "openrouter"}.
    """
    try:
        return ai_fill_template_via_openai_from_markdown(template, markdown_text, temperature), "openai"
    except Exception as e:
        print(f"OpenAI failed: {e}. Trying OpenRouter...")
        try:
            return ai_fill_template_via_openrouter_from_markdown(template, markdown_text, temperature), "openrouter"
        except Exception as e2:
            print(f"OpenRouter also failed: {e2}")
            raise RuntimeError(f"Both OpenAI and OpenRouter failed: {e2}")


def ai_fill_template_via_openrouter_from_markdown(template: Dict[str, Any], markdown_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Use an LLM (via OpenRouter) to map values from the document's Markdown into the template structure. Rules: Preserve all keys and nesting from the template. Replace any leaf value equal to the literal string 'string' OR null with best-matched value from the Markdown. If not found, set the value to null. Do not invent content not present in the Markdown; prefer exact strings/numbers.
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


st.title("📄 Docling Chat with Documents")
st.caption("Upload documents (PDF, DOCX, PPTX, images, etc.). Docling extracts text; then ask questions in chat.")

# Load environment variables from .env (e.g., OPENROUTER_API_KEY)
load_dotenv()

with st.sidebar:
    st.subheader("Upload")
    max_pages = st.number_input("Max pages (0 = all)", min_value=0, value=0, step=1, help="Process limit for very large files")
    page_range = st.text_input("Page range (e.g. 1-3,5)", value="")
    st.caption("Leave defaults if unsure.")

uploaded = st.sidebar.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "png", "jpg", "jpeg", "tif", "tiff", "csv", "vtt"],
)
process_btn = st.sidebar.button("Process Document", type="primary", disabled=(uploaded is None))

if uploaded is not None and process_btn:
    tmp_path = save_upload_to_temp(uploaded)
    st.info(f"Saved upload to temporary file: {tmp_path.name}")

    with st.spinner("Converting with Docling… (downloads models on first run)"):
        try:
            # Apply page controls when provided (best-effort depending on Docling version)
            result = convert_file(tmp_path, page_range=page_range, max_pages=max_pages)
        except Exception as e:
            st.error(f"Conversion failed: {type(e).__name__}: {e}")
            st.stop()

    if result.errors:
        with st.expander("Warnings/Errors during conversion"):
            for err in result.errors:
                st.write(str(err))

    doc = result.document
    md, html, doctags, json_text = doc_to_outputs(doc)

    st.session_state["doc_md"] = md
    st.session_state["doc_name"] = Path(uploaded.name).stem
    st.success(f"Document processed. Pages: {len(result.pages)} | Status: {result.status}")
    with st.expander("Preview extracted Markdown"):
        st.markdown(md)

# Chat section
st.header("💬 Chat With Your Document")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state.get("doc_md"):
    # Show a tiny status line
    used_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if os.getenv("OPENAI_API_KEY") else os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    st.caption(f"Context: {st.session_state.get('doc_name','(no name)')} — Model: {used_model}")

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    temperature_chat = st.slider("Chat temperature", 0.0, 1.0, 0.0, 0.1, key="chat_temp")
    prompt = st.chat_input("Ask a question about the uploaded document…")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer, provider = ai_chat_answer_with_fallback(st.session_state["doc_md"], prompt, temperature_chat)
                    st.session_state["messages"].append({"role": "assistant", "content": answer + f"\n\n_(provider: {provider})_"})
                    st.markdown(answer + f"\n\n_(provider: {provider})_")
                except Exception as e:
                    err = f"Chat failed: {type(e).__name__}: {e}"
                    st.session_state["messages"].append({"role": "assistant", "content": err})
                    st.error(err)
    st.button("Clear chat", on_click=lambda: st.session_state.update({"messages": []}))
else:
    st.info("Upload and process a document from the sidebar to start chatting.")

st.caption(
    "Tip: For very large documents, use lower temperature and rely on direct citations to keep answers grounded."
)
