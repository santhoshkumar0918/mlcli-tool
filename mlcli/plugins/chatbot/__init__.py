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
    dependencies = ["langchain", "openai", "faiss-cpu", "chromadb"]
    
    def get_directory_structure(self) -> List[str]:
        """Return directory structure for chatbot projects."""
        return [
            "data/knowledge_base",
            "src/chains",
            "src/prompts",
            "src/vectorstore",
            "notebooks",
            "tests",
            "logs",
        ]
    
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return boilerplate files for chatbot projects."""
        return {
            "README.md": self._get_readme(),
            ".gitignore": self._get_gitignore(),
            ".env.example": self._get_env_example(),
            "requirements.txt": self.get_requirements(),
            "src/chains/rag_chain.py": self._get_rag_chain(),
            "src/prompts/templates.py": self._get_prompts(),
            "src/vectorstore/store.py": self._get_vectorstore(),
            "src/app.py": self._get_app(),
            "data/knowledge_base/sample.txt": self._get_sample_data(),
        }
    
    def get_requirements(self) -> str:
        """Return requirements.txt content."""
        return """# LangChain ecosystem
langchain>=0.1.0
langchain-openai>=0.0.5

# Vector stores  
faiss-cpu>=1.7.4

# LLM providers
openai>=1.0.0

# UI
gradio>=4.0.0

# Utilities
python-dotenv>=1.0.0
"""
    
    def get_config_template(self) -> Dict[str, Any]:
        """Return plugin-specific config."""
        return {
            "plugin": "chatbot",
            "chatbot": {
                "vector_store": "faiss",
                "embedding_model": "text-embedding-ada-002",
                "llm_model": "gpt-3.5-turbo",
            }
        }
    
    def _get_readme(self) -> str:
        return """# Chatbot Project (RAG with LangChain)

Built with MLCLI chatbot plugin.

## Quick Start

1. Set up environment: `cp .env.example .env` and add OPENAI_API_KEY
2. Add docs to `data/knowledge_base/`
3. Run: `python src/app.py`
4. Open http://localhost:7860
"""
    
    def _get_gitignore(self) -> str:
        return """__pycache__/
.env
.venv
*.faiss
logs/
"""

    def _get_env_example(self) -> str:
        return """OPENAI_API_KEY=your_key_here
"""

    def _get_rag_chain(self) -> str:
        return '''"""RAG chain implementation."""
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from src.vectorstore.store import get_vectorstore


def create_rag_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
'''

    def _get_prompts(self) -> str:
        return '''"""Prompt templates."""
QA_PROMPT = "Context: {context}\\nQuestion: {question}\\nAnswer:"
'''

    def _get_vectorstore(self) -> str:
        return '''"""Vector store setup."""
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from pathlib import Path


def get_vectorstore():
    """Load or create vector store."""
    embeddings = OpenAIEmbeddings()
    docs_dir = Path("data/knowledge_base")
    
    # Load documents
    documents = []
    for file_path in docs_dir.glob("*.txt"):
        loader = TextLoader(str(file_path))
        documents.extend(loader.load())
    
    # Create vector store
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
'''

    def _get_app(self) -> str:
        return '''"""Gradio UI for chatbot."""
import gradio as gr
from src.chains.rag_chain import create_rag_chain


chain = create_rag_chain()


def chat(message, history):
    result = chain({"query": message})
    return result["result"]


demo = gr.ChatInterface(chat, title="RAG Chatbot")

if __name__ == "__main__":
    demo.launch()
'''

    def _get_sample_data(self) -> str:
        return """This is sample knowledge base content.

Add your documents here for the RAG system to learn from.
You can add .txt files, and the system will index them automatically.
"""
