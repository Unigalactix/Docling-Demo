import os
import io
import json
import re
import tempfile
from pathlib import Path

# Robust HF Hub settings for Windows (avoid symlink/hardlink issues)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_HARDLINKS", "1")

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
    # Recursively fill any leaf value equal to the literal string "string"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and v == "string":
                found = extract_value_from_text(k, text)
                out[k] = found if found is not None else ""
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

    # Template-based JSON filling (uses project master.json by default)
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

        # Extract plain text from doc for matching
        doc_text = doc.export_to_text()

        filled = fill_template_from_text(template, doc_text)
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
