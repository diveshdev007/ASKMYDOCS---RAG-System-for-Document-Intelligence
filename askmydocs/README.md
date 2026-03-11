# ASKMYDOCS: RAG System for Document Intelligence 📄

ASKMYDOCS is a Retrieval-Augmented Generation (RAG) prototype allowing users to upload private PDF documents and interrogate them via a conversational interface. Built on **Streamlit** and **LangChain**, it integrates **OpenAI Embeddings**, **FAISS**, and the **GPT-3.5-turbo** language model to provide answers exclusively sourced from uploaded document context, minimizing hallucination risks.

## Features

- **Document Upload & Processing**: Seamless UI for PDF uploads, extracting text via `PyPDF`.
- **Configurable Chunking Strategies**: Toggle between 500, 1000, and 1500 token chunk sizes via UI to observe retrieval quality VS speed tradeoffs.
- **Semantic Search via FAISS**: Efficient matching of user queries to document snippets using OpenAI embeddings.
- **Strict Anti-Hallucination Guardrails**: Prompts are meticulously structured to answer only from context, defaulting to "I don't know" when information is absent.
- **Temperature Control**: Slider to dynamically adjust model temperature (0.3 to 0.7) for deterministic vs. creative responses.
- **Explainable AI**: Direct insight into the RAG process by surfacing the exact document chunks utilized and their corresponding L2 similarity scores.

## Architecture

1. **PDF Upload** → 2. **Document Loader** (PyPDF) → 3. **Semantic Text Splitter** (RecursiveCharacterTextSplitter) → 4. **Embedding Generation** (OpenAIEmbeddings) → 5. **Vector Database** (FAISS) → 6. **RetrievalQA** (Similarity Search) → 7. **LLM Synthesis** (ChatOpenAI)

## Installation Prerequisites

- Python 3.8+
- An [OpenAI API Key](https://platform.openai.com/)

## Local Run Instructions

1. **Clone or download** this repository.
2. **Navigate** to the project directory:
   ```bash
   cd askmydocs
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**:
   Create a `.env` file in the root directory by copying `.env.example`, or input your key directly into the Streamlit UI.
   ```bash
   cp .env.example .env
   ```
5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
6. **Usage**:
   - Access the UI typically at `http://localhost:8501`.
   - Ensure the OpenAI API Key is submitted via the sidebar.
   - Adjust "Chunk Size" and "Temperature" according to your testing needs.
   - Upload any readable PDF document.
   - Type your questions into the main interface!
