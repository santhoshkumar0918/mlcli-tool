"""Chatbot/RAG plugin for LangChain-based projects.

Generates projects for building chatbots with Retrieval-Augmented Generation (RAG)
using LangChain, OpenAI, and vector stores.
"""

from typing import Dict, List, Any
from mlcli.plugins.base import PluginBase


class ChatbotPlugin(PluginBase):
    """Plugin for chatbot/RAG projects using LangChain."""
    
    name = "chatbot"
    description = "LangChain-based chatbot with RAG + Intent Classification (mlcli workflow supported)"
    dependencies = ["langchain", "openai", "faiss-cpu"]
    
    def get_directory_structure(self) -> List[str]:
        """Return directory structure for chatbot projects."""
        return [
            "data/raw",
            "data/processed",
            "data/knowledge_base",
            "models",
            "reports/figures",
            "notebooks",
            "src",
        ]
    
    def get_boilerplate_files(self) -> Dict[str, str]:
        """Return boilerplate files for chatbot projects."""
        return {
            "README.md": self._get_readme(),
            ".gitignore": self._get_gitignore(),
            ".env.example": self._get_env_example(),
            "requirements.txt": self.get_requirements(),
            "src/app.py": self._get_app(),
            "src/rag.py": self._get_rag(),
            "src/intent_classifier.py": self._get_intent_classifier(),
            "data/knowledge_base/sample.txt": self._get_sample_data(),
            "data/raw/intents.csv": self._get_sample_intents_csv(),
            "generate_demo_data.py": self._get_data_generator(),
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
            "data": {
                "target_column": "intent",
                "test_size": 0.2,
                "random_state": 42,
            },
            "model": {
                "task": "classification",
                "algorithms": ["logistic_regression", "random_forest", "xgboost"],
                "cv_folds": 5,
            },
            "chatbot": {
                "vector_store": "faiss",
                "embedding_model": "text-embedding-ada-002",
                "llm_model": "gpt-3.5-turbo",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
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

    def _get_sample_intents_csv(self) -> str:
        return """text,word_count,has_question_mark,has_greeting,has_price_word,has_help_word,has_order_word,has_cancel_word,intent
hello there,2,0,1,0,0,0,0,greeting
hi how are you,4,0,1,0,0,0,0,greeting
good morning,2,0,1,0,0,0,0,greeting
hey,1,0,1,0,0,0,0,greeting
what is the price,4,1,0,1,0,0,0,pricing
how much does it cost,5,1,0,1,0,0,0,pricing
what are your charges,4,1,0,1,0,0,0,pricing
tell me the cost,4,0,0,1,0,0,0,pricing
i need help,3,0,0,0,1,0,0,support
can you help me,4,1,0,0,1,0,0,support
i have a problem,4,0,0,0,1,0,0,support
please assist me,3,0,0,0,1,0,0,support
track my order,3,0,0,0,0,1,0,order_status
where is my package,4,1,0,0,0,1,0,order_status
order status please,3,0,0,0,0,1,0,order_status
when will it arrive,4,1,0,0,0,1,0,order_status
cancel my order,3,0,0,0,0,0,1,cancel
i want to cancel,4,0,0,0,0,0,1,cancel
stop my subscription,3,0,0,0,0,0,1,cancel
refund please,2,0,0,0,0,0,1,cancel
bye,1,0,0,0,0,0,0,farewell
goodbye,1,0,0,0,0,0,0,farewell
see you later,3,0,0,0,0,0,0,farewell
thanks goodbye,2,0,0,0,0,0,0,farewell
offer available,2,0,0,1,0,0,0,pricing
how do i get started,5,1,0,0,1,0,0,support
need assistance,2,0,0,0,1,0,0,support
where is my order,4,1,0,0,0,1,0,order_status
cancel subscription,2,0,0,0,0,0,1,cancel
hello good morning,3,0,1,0,0,0,0,greeting
"""

    def _get_data_generator(self) -> str:
        return '''"""Generate a larger intent classification dataset for training.

Usage:
    python generate_demo_data.py

Output:
    data/raw/intents.csv  (300 rows)
"""
import random
import os

random.seed(42)
os.makedirs("data/raw", exist_ok=True)

greetings = ["hello","hi","hey","good morning","good evening","howdy","what\'s up","greetings"]
pricing = ["what is the price","how much does it cost","show me pricing","any offers","what are the charges"]
support = ["i need help","can you assist","i have an issue","something is broken","not working"]
order = ["track my order","where is my package","order status","delivery update","when will it arrive"]
cancel = ["cancel my order","i want a refund","stop subscription","cancel account","refund please"]
farewell = ["bye","goodbye","see you","take care","talk later","good night"]

label_map = {
    "greeting": greetings,
    "pricing": pricing,
    "support": support,
    "order_status": order,
    "cancel": cancel,
    "farewell": farewell,
}

rows = ["text,word_count,has_question_mark,has_greeting,has_price_word,has_help_word,has_order_word,has_cancel_word,intent"]
price_words = ["price","cost","charge","fee","offer"]
help_words = ["help","assist","issue","problem","broken"]
order_words = ["order","package","delivery","arrive","track"]
cancel_words = ["cancel","refund","stop","subscription"]
greet_words = ["hello","hi","hey","morning","evening","howdy"]

for intent, templates in label_map.items():
    for _ in range(50):
        text = random.choice(templates)
        wc = len(text.split())
        hq = 1 if "?" in text else 0
        hg = 1 if any(w in text for w in greet_words) else 0
        hp = 1 if any(w in text for w in price_words) else 0
        hh = 1 if any(w in text for w in help_words) else 0
        ho = 1 if any(w in text for w in order_words) else 0
        hc = 1 if any(w in text for w in cancel_words) else 0
        rows.append(f"{text},{wc},{hq},{hg},{hp},{hh},{ho},{hc},{intent}")

with open("data/raw/intents.csv", "w") as f:
    f.write("\n".join(rows))

print(f"Created data/raw/intents.csv ({len(rows)-1} rows)")
print("Target: intent (greeting, pricing, support, order_status, cancel, farewell)")
print()
print("Next steps:")
print("  mlcli preprocess -i data/raw/intents.csv -t intent -o data/processed")
print("  mlcli train -t data/processed/train.csv --test-data data/processed/test.csv --target intent -o models")
'''

    def _get_intent_classifier(self) -> str:
        return '''"""Use the trained mlcli model for intent classification in the chatbot."""
import joblib
from pathlib import Path


def load_intent_model(model_path: str = "models/best_model.pkl"):
    """Load the trained intent classifier."""
    path = Path(model_path)
    if not path.exists():
        print(f"[Warning] Intent model not found at {model_path}")
        print("Run: mlcli train -t data/processed/train.csv --target intent -o models")
        return None
    return joblib.load(path)


def load_pipeline(pipeline_path: str = "data/processed/preprocessing_pipeline.pkl"):
    """Load the preprocessing pipeline."""
    path = Path(pipeline_path)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_intent(text: str, model=None, pipeline=None) -> str:
    """Predict the intent of a user message.
    
    Args:
        text: Raw user message
        model: Loaded sklearn model (from mlcli train)
        pipeline: Loaded preprocessing pipeline (from mlcli preprocess)
    
    Returns:
        Predicted intent label (e.g. \'greeting\', \'pricing\', \'support\')
    """
    if model is None or pipeline is None:
        return "unknown"

    import pandas as pd
    price_words = ["price","cost","charge","fee","offer"]
    help_words  = ["help","assist","issue","problem","broken"]
    order_words = ["order","package","delivery","arrive","track"]
    cancel_words= ["cancel","refund","stop","subscription"]
    greet_words = ["hello","hi","hey","morning","evening","howdy"]

    words = text.lower().split()
    features = pd.DataFrame([{
        "word_count":         len(words),
        "has_question_mark":  1 if "?" in text else 0,
        "has_greeting":       1 if any(w in words for w in greet_words) else 0,
        "has_price_word":     1 if any(w in words for w in price_words) else 0,
        "has_help_word":      1 if any(w in words for w in help_words) else 0,
        "has_order_word":     1 if any(w in words for w in order_words) else 0,
        "has_cancel_word":    1 if any(w in words for w in cancel_words) else 0,
    }])

    preprocessor = pipeline.get("preprocessor")
    if preprocessor:
        features = preprocessor.transform(features)

    return model.predict(features)[0]


if __name__ == "__main__":
    model    = load_intent_model()
    pipeline = load_pipeline()

    test_msgs = [
        "hello there",
        "what is the price?",
        "i need help",
        "track my order",
        "cancel my subscription",
        "goodbye",
    ]
    for msg in test_msgs:
        intent = predict_intent(msg, model, pipeline)
        print(f"  {msg!r:40s} → {intent}")
'''
