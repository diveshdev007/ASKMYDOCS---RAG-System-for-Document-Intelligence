import streamlit as st
import tempfile
import os
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="ASKMYDOCS", page_icon="📄", layout="wide")

st.title("ASKMYDOCS: RAG System for Document Intelligence 📄")
st.markdown("Upload a private PDF document and ask questions about it. The system retrieves relevant chunks and generates an answer without hallucinations.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Pipeline Configuration")
    
    api_key = st.text_input("Google API Key", type="password", help="Enter your Google API Key for the Gemini LLM.")
    
    chunk_size = st.selectbox(
        "Chunk Size (Tokens)", 
        options=[500, 1000, 1500],
        index=1,
        help="Affects how much context is retrieved per chunk. Compare this to balance accuracy and speed."
    )
    
    temperature = st.slider(
        "Model Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Adjust the creativity. 0.3 is strict, 0.7 is more creative. Higher values may increase hallucination risk."
    )

    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

# --- Main Application Logic ---
if "rag" not in st.session_state:
    st.session_state.rag = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if uploaded_file and api_key:
    if not st.session_state.doc_processed or st.sidebar.button("Reprocess Document"):
        with st.spinner("Embedding Document locally and Processing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                st.session_state.rag = RAGPipeline(api_key=api_key)
                st.session_state.chunk_count = st.session_state.rag.process_document(
                    file_path=tmp_file_path, 
                    chunk_size=chunk_size
                )
                st.session_state.rag.setup_qa_chain(temperature=temperature)
                
                st.session_state.doc_processed = True
                os.remove(tmp_file_path)
                st.success(f"Document processed successfully! Splitting resulted in {st.session_state.chunk_count} chunks.")
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.session_state.doc_processed = False

elif not api_key:
    st.info("Please enter your Google API Key in the sidebar to proceed.")
elif not uploaded_file:
    st.info("Please upload a PDF document in the sidebar.")

# --- Chat Interface ---
if st.session_state.doc_processed:
    st.markdown("### Ask Questions")
    
    question = st.text_input("What would you like to know from the document?")
    
    if st.button("Generate Answer"):
        if question:
            with st.spinner("Searching locally and Generating via Gemini..."):
                try:
                    st.session_state.rag.setup_qa_chain(temperature=temperature)
                    response = st.session_state.rag.query(question)
                    
                    st.markdown("#### Answer:")
                    st.info(response["answer"])
                    
                    with st.expander("View Retrieved Source Chunks & Similarity Scores"):
                        st.markdown(f"**Temperature Used:** {temperature}")
                        st.markdown(f"**Chunk Size Used:** {chunk_size}")
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown("**Source Fragments:**")
                            for i, doc in enumerate(response["source_documents"]):
                                st.text_area(f"Chunk {i+1}", value=doc.page_content, height=150, disabled=True)
                        
                        with col2:
                            st.markdown("**Similarity (L2):**")
                            st.write("*(Lower is better)*")
                            for item in response["similarity_scores"]:
                                st.metric(label="", value=f"{item['score']:.4f}")
                                st.caption(item['chunk_snippet'].replace('\n',' '))

                except Exception as e:
                    st.error(f"Error querying the system: {e}")
        else:
            st.warning("Please enter a question.")

