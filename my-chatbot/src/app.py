"""Gradio UI for RAG Chatbot."""
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
                response += f"\n\n*Sources: {', '.join(sources)}*"
        
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
