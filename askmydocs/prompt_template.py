from langchain.prompts import PromptTemplate

# This prompt strictly forces the model to only use the retrieved context
RAG_PROMPT_TEMPLATE = """You are a helpful and knowledgeable assistant for answering questions based on the provided document context.

Follow these rules strictly:
1. You must answer the question using ONLY the provided context.
2. If the answer is not found in the context, respond with 'I don't know'.
3. Do not include outside knowledge or try to make up an answer.
4. Keep your answer concise and clear.

Context:
{context}

Question:
{question}

Helpful Answer:"""

def get_rag_prompt():
    """Returns the LangChain PromptTemplate for the RAG system."""
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
