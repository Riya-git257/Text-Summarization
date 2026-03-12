# ── Imports ───────────────────────────────────────────────────────────────────
import re
import validators
import streamlit as st # type: ignore
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="URL Summarizer", page_icon="🔗")
st.title("🔗 URL Summarizer")
st.write("Summarize any YouTube video or website using Groq + LangChain.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("[Get a Groq API key](https://console.groq.com/)")

# ── Main Input ────────────────────────────────────────────────────────────────
generic_url = st.text_input("Enter a URL", placeholder="https://youtube.com/watch?v=... or any website")

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="Provide a clear and concise summary of the following content in 300 words.\n\nContent:{text}",
    input_variables=["text"],
)

# ── Helper: Extract YouTube Video ID ─────────────────────────────────────────
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",      # youtube.com/watch?v=
        r"youtu\.be/([a-zA-Z0-9_-]{11})",  # youtu.be/
        r"shorts/([a-zA-Z0-9_-]{11})",     # youtube.com/shorts/
        r"embed/([a-zA-Z0-9_-]{11})",      # youtube.com/embed/
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# ── Helper: Load YouTube Transcript ──────────────────────────────────────────
def load_youtube(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ Could not extract a valid YouTube video ID from that URL.")
        st.stop()

    st.caption(f"Video ID detected: `{video_id}`")

    loader = YoutubeLoader(
        video_id=video_id,
        add_video_info=False,
        language=["en", "en-US", "en-IN", "hi"],
    )
    docs = loader.load()

    if not docs or not docs[0].page_content.strip():
        st.error("❌ No transcript found. The video may have captions disabled, be private, or region-locked.")
        st.stop()

    return docs

# ── Helper: Load Website ──────────────────────────────────────────────────────
def load_website(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()

    if not docs or not docs[0].page_content.strip():
        st.error("❌ No content could be extracted from that URL.")
        st.stop()

    return docs

# ── Summarize Button ──────────────────────────────────────────────────────────
if st.button("Summarize", type="primary"):

    # Validate inputs
    if not groq_api_key.strip():
        st.error("❌ Please enter your Groq API key in the sidebar.")
        st.stop()
    if not generic_url.strip():
        st.error("❌ Please enter a URL.")
        st.stop()
    if not validators.url(generic_url):
        st.error("❌ That doesn't look like a valid URL. Make sure it starts with https://")
        st.stop()

    try:
        with st.spinner("Fetching content and summarizing..."):

            # 1. Load content
            is_youtube = "youtube.com" in generic_url or "youtu.be" in generic_url
            docs = load_youtube(generic_url) if is_youtube else load_website(generic_url)

            # 2. Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            # 3. Summarize
            llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
            chain = load_summarize_chain(llm, chain_type="map_reduce", combine_prompt=prompt)
            summary = chain.run(chunks)

        # 4. Display result
        st.subheader("Summary")
        st.write(summary)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        import traceback
        st.code(traceback.format_exc())