# 🔗 AI URL Summarizer with RAG

An intelligent **Retrieval-Augmented Generation (RAG)** application that summarizes and answers questions about **web pages** using **LangChain, FAISS, Hugging Face Embeddings, Groq LLM, and Streamlit**.

Instead of sending the entire document to the LLM, the application retrieves only the most relevant content before generating a response, resulting in more accurate and context-aware answers.

---

## 🌐 Live Demo

🚀 **Try the deployed application here:**

👉 **[Open AI URL Summarizer](https://text-summarization-iv9dtagak9gjkjkpdqtkl3.streamlit.app/)**

---

## ✨ Features

* 🌐 Summarize content from any public website.
* 🤖 Ask questions about the loaded document.
* 🧠 Retrieval-Augmented Generation (RAG) pipeline.
* 🔍 Semantic search using Hugging Face embeddings.
* 📚 FAISS vector database for efficient retrieval.
* ⚡ Groq Llama 3.3 70B for fast inference.
* 🎯 Modular and maintainable project structure.
* 💻 Interactive Streamlit interface.

> **Note:** Website RAG is fully functional. YouTube support is currently under development.

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Groq API
* Hugging Face Embeddings
* FAISS
* BeautifulSoup
* Requests

---

## 📂 Project Structure

```text
.
├── app.py
├── components
│   ├── loader.py
│   ├── splitter.py
│   ├── embedding.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── llm.py
│   ├── prompts.py
│   └── rag_chain.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Riya-git257/Text-Summarization.git
cd your-repository
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🚀 How It Works

1. Enter a website URL.
2. The document is loaded and cleaned.
3. The content is split into manageable chunks.
4. Each chunk is converted into vector embeddings.
5. FAISS creates a searchable vector index.
6. The retriever finds the most relevant chunks for the user's query.
7. Groq Llama generates a response using the retrieved context.

---

## 🏗️ RAG Architecture

```text
Website URL
      │
      ▼
Document Loader
      │
      ▼
Text Splitter
      │
      ▼
Hugging Face Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
Groq Llama 3.3 70B
      │
      ▼
Generated Answer
```

---

## 🔮 Future Enhancements

* YouTube transcript support
* PDF document support
* Chat history
* Source citations
* Flashcard generation
* Quiz generation
* Export summaries to PDF/DOCX
* Multi-document RAG

---

## 🤝 Contributions

Contributions, feature requests, and suggestions are welcome. Feel free to open an issue or submit a pull request.

---

## 📜 License

This project is intended for educational and learning purposes.

