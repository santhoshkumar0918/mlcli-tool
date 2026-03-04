# Chatbot Project (RAG with LangChain)

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
