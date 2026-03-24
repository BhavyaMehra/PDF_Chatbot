# PDF Chatbot

A conversational AI app that lets you upload any PDF and ask questions about it in natural language. Built with LangChain, FAISS, and Streamlit.

## How it works

1. Upload a PDF via the Streamlit interface
2. The document is chunked and embedded using HuggingFace sentence transformers
3. Your question is matched against the most relevant chunks using FAISS vector search
4. A DeepSeek LLM generates a response grounded strictly in the retrieved context

## Tech Stack

- LangChain for RAG pipeline and retrieval chain
- FAISS for vector storage and semantic search
- HuggingFace all-MiniLM-L6-v2 for embeddings
- DeepSeek R1 (via OpenAI-compatible API) as the LLM
- Streamlit for the frontend

## Setup

1. Clone the repo
2. Install dependencies

pip install -r requirements.txt

3. Create a .streamlit/secrets.toml file with your API credentials

[secrets]
openai_api_key = "your-key-here"
openai_api_base = "your-base-url-here"

4. Run the app

streamlit run chatbot_ui_modern.py

## Notes

- Only one PDF can be loaded at a time
- Responses are constrained to document context to reduce hallucination
- The data/ folder is gitignored and not included in the repo