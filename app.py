import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import pymupdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in your .env file!")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# Initialize Embedding Model & ChromaDB Persistent Client
@st.cache_resource
def load_rag_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="intellibot_kb")
    return model, collection

embedding_model, collection = load_rag_resources()

# Streamlit UI Configuration
st.set_page_config(page_title="IntelliBot - Context-Aware RAG Chatbot", layout="wide")
st.title("🤖 IntelliBot: Context-Aware RAG Chatbot")
st.markdown("Upload PDF/TXT documents via the sidebar and ask questions grounded strictly in your knowledge base with multi-turn conversation memory.")

# Sidebar for Document Ingestion & Clear Chat
with st.sidebar:
    st.header("📂 Knowledge Base Ingestion")
    
    try:
        count = collection.count()
        st.info(f"📊 Current Chunks in DB: **{count}**")
    except Exception:
        pass

    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process & Embed Documents") and uploaded_files:
        with st.spinner("Processing documents and generating embeddings..."):
            total_chunks = 0
            for uploaded_file in uploaded_files:
                file_text = ""
                try:
                    if uploaded_file.type == "application/pdf":
                        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                        for page in doc:
                            file_text += page.get_text()
                    else:
                        file_text = uploaded_file.read().decode("utf-8")
                    
                    chunk_size = 500
                    chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]
                    
                    if chunks:
                        embeddings = embedding_model.encode(chunks).tolist()
                        ids = [f"{uploaded_file.name}_{i}_{os.urandom(4).hex()}" for i in range(len(chunks))]
                        metadatas = [{"source": uploaded_file.name} for _ in range(len(chunks))]
                        
                        collection.add(
                            documents=chunks,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            ids=ids
                        )
                        total_chunks += len(chunks)
                except Exception as e:
                    st.error(f"Error parsing {uploaded_file.name}: {e}")
            
            st.success(f"Successfully embedded {total_chunks} chunks!")
            st.rerun()

    st.divider()
    if st.button("🗑 Clear Knowledge Base & Chat"):
        try:
            chromadb.PersistentClient(path="./chroma_db").delete_collection("intellibot_kb")
        except Exception:
            pass
        st.session_state.messages = []
        st.success("Knowledge base and chat cleared!")
        st.rerun()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Prior Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Helper function incorporating RAG retrieval + Chat History Memory
def generate_rag_response(prompt_text):
    try:
        # 1. Vector Search for relevant document chunks based on latest prompt
        query_vector = embedding_model.encode(prompt_text).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=3)
        
        context_texts = []
        sources = set()
        
        if results and results.get("documents") and len(results["documents"]) > 0:
            context_texts = results["documents"][0]
            if results.get("metadatas") and len(results["metadatas"]) > 0:
                for meta in results["metadatas"][0]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
        
        context = "\n\n".join(context_texts) if context_texts else "No relevant context found."
        source_str = ", ".join(sources) if sources else "Unknown source"

        # 2. Format recent conversation history (last 4 turns) for memory context
        chat_history_text = ""
        recent_messages = st.session_state.messages[-4:]  # Keep last few turns for context
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_history_text += f"{role}: {msg['content']}\n"

        # 3. Construct structured RAG prompt with memory & grounding constraints
        rag_prompt = f"""You are IntelliBot, a precise corporate assistant. 
Answer the user's question using ONLY the provided document context below and the conversation history. If the answer cannot be found in the context, state clearly that you do not know. Do not hallucinate.

Conversation History:
{chat_history_text}

Retrieved Document Context:
{context}

Current User Question: {prompt_text}
"""

        # 4. Generate response using Gemini API
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=rag_prompt
        )
        
        final_answer = f"{response.text}\n\n*📌 **Source(s):** {source_str}*"
        return final_answer
    except Exception as err:
        return f"⚠️ Network or API connection issue: {err}. Please try sending your message again."

# User Chat Input Handler
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating response..."):
            answer = generate_rag_response(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})