# ── Imports ───────────────────────────────────────────────────────────────────
import validators
import streamlit as st 
from components.loader import load_website, load_youtube
from components.splitter import split_documents
from components.embedding import get_embedding_model
from components.vectorestore import create_vectorstore
from components.retriever import get_retriever
from components.llm import get_llm
from components.rag_chain import build_rag_chain



# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="URL Summarizer", page_icon="🔗")
st.title("🔗 URL Summarizer")
st.write("Summarize any YouTube video or website using Groq + LangChain.")

# -- Seesion State -------------------------------------------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "url_loaded" not in st.session_state:
    st.session_state.url_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.header("Settings 🪦")

    groq_api_key = st.text_input("Groq API Key 🔑", type="password", placeholder="gsk_...")

    st.markdown("[Get a Groq API key](https://console.groq.com/)")

# -- Main Input -----------------------------------------------------------------
url = st.text_input(
    "Enter URL 🔍",
    placeholder = "https://...."
)


#---load button -----------------------------------------------------------------

if st.button("Load url 🍵"):
    
    if not groq_api_key:
        st.error("Please enter your Groq API key")
        st.stop()

    if not url:

        st.error("Please enter a URL")
        st.stop()

    if not validators.url(url):

        st.error("Invalid URL")
        st.stop()

    with st.spinner("Loading document...."):

        try:
            if "youtube.com" in url or "youtu.be" in url:

                st.info("Loading YouTube transcript...")

                docs = load_youtube(url)

            else:

                st.info("Loading Website...")

                docs = load_website(url)

            chunks = split_documents(docs)

            embedding = get_embedding_model()

            vectorstore = create_vectorstore(
                chunks,
                embedding
            )

            retriever = get_retriever(vectorstore)

            st.session_state.retriever = retriever
            st.session_state.url_loaded = True

            st.success("Document Loaded Successfully")

        except Exception as e:

            st.error(e)

#----Ask Query-----------------------------------------------------------------

if st.session_state.url_loaded:

    st.divider()

    st.subheader("Ask Questions")

    user_query = st.text_input(
        "Ask anything about the document",
        placeholder="Summarize the document..."
    )

    if user_query:

        llm = get_llm(groq_api_key)

        rag_chain = build_rag_chain(
            st.session_state.retriever,
            llm
        )

        with st.spinner("Generating answer..."):

            answer = rag_chain.invoke(user_query)

        st.subheader("Answer")

        st.write(answer)
