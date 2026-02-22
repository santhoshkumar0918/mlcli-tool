"""Chatbot/RAG plugin for LangChain-based projects.

Generates projects for building chatbots with Retrieval-Augmented Generation (RAG)
using LangChain, OpenAI, and vector stores.
"""

from typing import Dict, List, Any
from mlcli.plugins.base import PluginBase


class ChatbotPlugin(PluginBase):
    """Plugin for chatbot/RAG projects using LangChain."""
    
    name = "chatbot"
    description = "LangChain-based chatbot with RAG (Retrieval-Augmented Generation)"
    dependencies = ["langchain", "openai", "faiss-cpu"]
    
    def get_directory_structure(self) -> List[str]:
        """Return minimal directory structure for chatbot projects."""
        return [
            "data/knowledge_base",
            "src",
        ]
    
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return essential boilerplate files for chatbot projects."""
        return {
            "README.md": self._get_readme(),
            ".gitignore": self._get_gitignore(),
            ".env.example": self._get_env_example(),
            "requirements.txt": self.get_requirements(),
            "src/app.py": self._get_app(),
            "src/rag.py": self._get_rag(),
            "data/knowledge_base/sample.txt": self._get_sample_data(),
        }
    
    def get_requirements(self) -> str:
        """Return requirements.txt content."""
        return """# LangChain ecosystem
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10

# Vector stores
faiss-cpu>=1.7.4

# LLM providers
openai>=1.0.0

# UI
gradio>=4.0.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
rich>=13.0.0
"""
    
    def get_config_template(self) -> Dict[str, Any]:
        """Return plugin-specific config."""
        return {
            "plugin": "chatbot",
            "chatbot": {
                "vector_store": "faiss",
                "embedding_model": "text-embedding-ada-002",
                "llm_model": "gpt-3.5-turbo",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        }
    
    def _get_readme(self) -> str:
        return """# Chatbot Project (RAG with LangChain)

Built with **ML Assistant CLI** chatbot plugin.

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Add documents to knowledge base
cp /path/to/your/docs/* data/knowledge_base/

# 3. Run the chatbot
python src/app.py

# 4. Open http://localhost:7860
```

## Project Structure

```
├── data/
│   └── knowledge_base/    # Your documents (.txt, .pdf, .md)
├── src/
│   ├── app.py             # Gradio UI
│   └── rag.py             # RAG pipeline
├── .env                   # API keys (create from .env.example)
├── .env.example           # Template for environment variables
└── mlcli.yaml             # Configuration
```

## Configuration

Edit `mlcli.yaml` to customize:
- LLM model (gpt-3.5-turbo, gpt-4, etc.)
- Embedding model
- Chunk size and overlap
- Vector store type

## Supported Document Types

- Plain text (.txt)
- Markdown (.md)
- PDF (requires pypdf)
- CSV files

## Next Steps

1. Add your API key to `.env`
2. Add documents to `data/knowledge_base/`
3. Run `python src/app.py`
"""
    
    def _get_gitignore(self) -> str:
        return """# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# ML artifacts
*.faiss
*.pkl
logs/

# LangChain
chroma_db/
"""
    
    def _get_env_example(self) -> str:
        return """# OpenAI API Key (required)
OPENAI_API_KEY=your_key_here

# Optional: Use Azure OpenAI
# AZURE_OPENAI_API_KEY=
# AZURE_OPENAI_ENDPOINT=
"""
    
    def _get_rag(self) -> str:
        return '''"""RAG pipeline implementation."""
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
'''

    def _get_app(self) -> str:
        return '''"""Gradio UI for RAG Chatbot."""
import os
from pathlib import Path

import gradio as gr
import yaml
from dotenv import load_dotenv

from src.rag import initialize_rag

load_dotenv()


def load_config():
    """Load mlcli.yaml configuration."""
    config_path = Path("mlcli.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    config = load_config()
    
    chain = initialize_rag(config=config)
    
    if chain is None:
        print("Error: Could not initialize RAG pipeline. Check your documents.")
        return
    
    def chat(message, history):
        """Chat function for Gradio."""
        result = chain({"query": message})
        
        response = result["result"]
        
        if result.get("source_documents"):
            sources = list(set(doc.metadata.get("source", "Unknown") 
                              for doc in result["source_documents"]))
            if sources:
                response += f"\\n\\n*Sources: {', '.join(sources)}*"
        
        return response
    
    demo = gr.ChatInterface(
        chat,
        title="RAG Chatbot",
        description="Ask questions about your documents",
        examples=[
            "What is this document about?",
            "Summarize the key points",
            "What are the main recommendations?",
        ]
    )
    
    demo.launch()


if __name__ == "__main__":
    main()
'''

    def _get_sample_data(self) -> str:
        return """This is sample knowledge base content.

Add your documents here for the RAG system to learn from.

Supported formats:
- .txt files (plain text)
- .pdf files (requires pypdf package)
- .md files (markdown)

The system will automatically:
1. Load all documents from this directory
2. Split them into chunks
3. Create embeddings
4. Build a vector index for fast retrieval

Example questions you can ask:
- "What is this document about?"
- "Summarize the key points"
- "What are the main recommendations?"
"""
