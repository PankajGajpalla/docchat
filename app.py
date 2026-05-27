import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.header("📄 DocChat")
    st.write("Upload a PDF and ask questions about it. Answers come straight from your document.")
    st.divider()
    st.subheader("How it works")
    st.write("1. Upload a PDF")
    st.write("2. Wait for processing")
    st.write("3. Ask anything about it")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Chat with your document")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if not uploaded_file:
    st.info("👆 Upload a PDF to get started.")
    st.stop()

# --- Process the PDF once ---
if "vectorstore" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing document..."):
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)

    st.success(f"Ready! Processed {len(chunks)} chunks. Ask me anything.")

# --- Initialize state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat input + RAG ---
user_question = st.chat_input("Ask about your document...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant = st.session_state.vectorstore.similarity_search(user_question, k=5)
            context = "\n\n".join(chunk.page_content for chunk in relevant)
            prompt = f"""Answer using ONLY the context below. If the answer isn't in the context, say "I don't have that information in the document."

Context:
{context}

Question: {user_question}

Answer:"""
            answer = st.session_state.llm.invoke(prompt).content
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})