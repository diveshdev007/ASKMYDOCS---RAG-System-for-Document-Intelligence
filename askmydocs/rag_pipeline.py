import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from prompt_template import get_rag_prompt

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, api_key: str):
        """Initialize pipeline with Gemini API key."""
        os.environ["GOOGLE_API_KEY"] = api_key
        # Use a small, fast local embedding model from HuggingFace
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.qa_chain = None
        logger.info("RAG Pipeline initialized with Local Embeddings and Gemini.")

    def process_document(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        """
        Loads PDF, extracts text, chunks it, and creates a FAISS vectorstore.
        """
        logger.info(f"Loading document from {file_path}...")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise e
        
        logger.info(f"Extracted {len(documents)} pages. Splitting text with chunk_size={chunk_size}...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")
        
        logger.info("Generating local embeddings and building FAISS vector database...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info("Vector database built successfully.")
        
        return len(chunks)

    def setup_qa_chain(self, temperature: float = 0.3, top_k: int = 4):
        """
        Sets up the RetrievalQA chain with the specified Gemini model configuration.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please process a document first.")
        
        logger.info(f"Setting up QA Chain using Gemini with temperature={temperature} and top_k={top_k}...")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
        
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        
        prompt = get_rag_prompt()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("QA Chain is ready.")

    def query(self, question: str):
        """
        Queries the RAG pipeline.
        """
        if self.qa_chain is None:
            raise ValueError("QA Chain not initialized. Please call setup_qa_chain first.")
        
        logger.info(f"Processing query: {question}")
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=self.qa_chain.retriever.search_kwargs['k'])
        
        scores_info = [{"chunk_snippet": doc.page_content[:100] + "...", "score": score} for doc, score in docs_and_scores]
        
        res = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": res["result"],
            "source_documents": res["source_documents"],
            "similarity_scores": scores_info
        }
