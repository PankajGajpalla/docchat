# 📄 DocChat — Chat With Your Documents

An AI-powered application that lets you upload any PDF and ask questions about it in natural language. Built with RAG (Retrieval Augmented Generation).

## Features

- Upload any PDF and chat with it
- Answers grounded in your document (won't hallucinate)
- Conversation memory across questions
- Clean web interface built with Streamlit

## How it works

1. **Upload** — PDF is split into chunks
2. **Embed** — chunks converted to vectors using HuggingFace embeddings
3. **Store** — vectors saved in ChromaDB
4. **Retrieve** — your question finds the most relevant chunks by meaning
5. **Generate** — the LLM answers using only those chunks

## Tech Stack

- Python
- Streamlit (web interface)
- LangChain (orchestration)
- ChromaDB (vector database)
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Groq (LLM inference)

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Groq API key
4. Run: `streamlit run app.py`

## Demo

[Add a screenshot or screen recording here]