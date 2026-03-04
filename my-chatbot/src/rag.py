"""RAG pipeline implementation."""
import os
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_documents(knowledge_dir: Path) -> List:
    """Load documents from knowledge base directory."""
    documents = []
    
    for file_path in knowledge_dir.glob("*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    return documents


def create_vectorstore(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Create FAISS vectorstore from documents."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore


def create_rag_chain(vectorstore, model_name: str = "gpt-3.5-turbo"):
    """Create RAG question-answering chain."""
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    prompt_template = """Use the following pieces of context to answer the question at the end.
If you dont know the answer, just say that you dont know, dont try to make up an answer.

Context: {context}

Question: {question}
Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain


def initialize_rag(knowledge_dir: Path = None, config: dict = None):
    """Initialize complete RAG pipeline."""
    if knowledge_dir is None:
        knowledge_dir = Path("data/knowledge_base")
    
    if config is None:
        config = {}
    
    chatbot_config = config.get("chatbot", {})
    chunk_size = chatbot_config.get("chunk_size", 1000)
    chunk_overlap = chatbot_config.get("chunk_overlap", 200)
    model_name = chatbot_config.get("llm_model", "gpt-3.5-turbo")
    
    print(f"Loading documents from {knowledge_dir}...")
    documents = load_documents(knowledge_dir)
    
    if not documents:
        print("Warning: No documents found in knowledge base")
        return None
    
    print(f"Loaded {len(documents)} documents")
    print("Creating vector store...")
    vectorstore = create_vectorstore(documents, chunk_size, chunk_overlap)
    
    print("Creating RAG chain...")
    chain = create_rag_chain(vectorstore, model_name)
    
    print("RAG pipeline initialized!")
    return chain
