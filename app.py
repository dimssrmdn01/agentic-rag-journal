import os
import tempfile
import re
import shutil
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------------------------------
# IMPORT BARU UNTUK FITUR STREAMING
# ----------------------------------------------------------------------------
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

# Load rahasia dari file .env
load_dotenv()

if "db_path" not in st.session_state:
    st.session_state.db_path = tempfile.mkdtemp()
DB_PATH = st.session_state.db_path

# ----------------------------------------------------------------------------
# HANDLER STREAMING CUSTOM
# ----------------------------------------------------------------------------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, status_indicator):
        self.container = container
        self.status_indicator = status_indicator
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Hapus teks loading saat AI mulai mengetik kata pertama
        if self.status_indicator:
            self.status_indicator.empty()
            self.status_indicator = None
        
        self.text += token
        self.container.markdown(self.text + "▌") # Tambahkan kursor berkedip

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

        :root {
            --bg: #0D1117;
            --bg-elevated: #151B24;
            --panel: #1A2029;
            --border: #262E3A;
            --accent: #7C9CFF;
            --accent-soft: rgba(124,156,255,0.12);
            --amber: #E8A33D;
            --text: #E6E9EF;
            --text-muted: #8A93A3;
            --text-faint: #5C6470;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu, footer, header { visibility: hidden; }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1100px 480px at 15% -10%, rgba(124,156,255,0.10), transparent 60%),
                radial-gradient(900px 420px at 100% 0%, rgba(232,163,61,0.06), transparent 55%),
                var(--bg);
        }

        [data-testid="stHeader"] { background: transparent; }

        .block-container {
            padding-top: 2.2rem;
            max-width: 760px;
        }

        .app-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 4px;
        }

        .app-mark {
            width: 34px;
            height: 34px;
            border-radius: 9px;
            background: linear-gradient(135deg, var(--accent), #4C6FE0);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            color: #0D1117;
            flex-shrink: 0;
        }

        .app-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.01em;
            line-height: 1.2;
        }

        .app-subtitle {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin: 4px 0 1.4rem 46px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            color: var(--text-muted);
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 5px 13px;
            margin-bottom: 1.8rem;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #3DD68C;
            box-shadow: 0 0 0 3px rgba(61,214,140,0.15);
        }

        .status-dot.offline {
            background: #E36464;
            box-shadow: 0 0 0 3px rgba(227,100,100,0.15);
        }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background: transparent;
            border: none;
            padding: 0.35rem 0;
            gap: 12px;
        }

        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            width: 30px;
            height: 30px;
            border-radius: 8px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
        }

        [data-testid="stChatMessageAvatarUser"] {
            background: var(--panel) !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border);
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, var(--accent), #4C6FE0) !important;
            color: #0D1117 !important;
        }

        [data-testid="stChatMessageContent"] {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 16px;
            color: var(--text);
            font-size: 0.94rem;
            line-height: 1.6;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
            border-left: 2px solid var(--accent);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-elevated);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * { color: var(--text); }

        [data-testid="stSidebar"] .sidebar-heading {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-faint);
            margin: 1.6rem 0 0.6rem 0;
        }

        [data-testid="stSidebar"] .sidebar-note {
            font-size: 0.82rem;
            color: var(--text-muted) !important;
            line-height: 1.55;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] input,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 8px;
        }

        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: var(--accent) !important;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            border: 1px solid var(--border);
            background: var(--panel);
            color: var(--text);
        }

        .stButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* Chat input */
        [data-testid="stChatInput"] {
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--text) !important;
        }

        /* Source cards */
        .source-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 2px solid var(--amber);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 8px;
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.55;
        }

        .source-label {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.66rem;
            color: var(--amber);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 5px;
        }

        [data-testid="stExpander"] {
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        [data-testid="stExpander"] summary {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-muted);
        }

        /* Confidence pill */
        .confidence-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            padding: 4px 11px;
            border-radius: 999px;
            margin-bottom: 10px;
            border: 1px solid var(--border);
        }
        .confidence-pill.high { color: #3DD68C; background: rgba(61,214,140,0.08); }
        .confidence-pill.mid { color: var(--amber); background: rgba(232,163,61,0.08); }
        .confidence-pill.low { color: #E36464; background: rgba(227,100,100,0.08); }

        /* Citation badges inside answer text */
        .citation-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.66rem;
            font-weight: 600;
            min-width: 16px;
            height: 16px;
            padding: 0 4px;
            border-radius: 5px;
            background: var(--accent-soft);
            color: var(--accent);
            text-decoration: none !important;
            margin: 0 1px;
            transform: translateY(-3px);
            transition: background .15s ease, color .15s ease;
        }
        .citation-badge:hover { background: var(--accent); color: #0D1117; }

        /* Sources as native details/summary */
        details.source-block {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 2px 4px;
            margin-top: 6px;
        }
        details.source-block summary {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-muted);
            padding: 10px 8px;
            cursor: pointer;
            list-style: revert;
        }
        details.source-block[open] summary { color: var(--text); }
        details.source-block .source-card:target,
        details.source-block .source-card.flash {
            border-color: var(--accent);
            background: var(--accent-soft);
        }
        .source-card { scroll-margin-top: 90px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-heading">Konfigurasi</div>', unsafe_allow_html=True)

    default_key = os.getenv("GROQ_API_KEY", "")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Disimpan hanya untuk sesi ini, tidak ditulis ke file.",
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
    )

    top_k = st.slider("Jumlah dokumen diambil (k)", min_value=1, max_value=8, value=3)

    st.markdown('<div class="sidebar-heading">Tentang</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">Asisten riset yang menjawab pertanyaan berdasarkan '
        'dokumen di knowledge base kamu. Jawaban difilter lewat tahap relevance grading '
        'sebelum digenerate, supaya tidak mengarang di luar konteks.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Sesi</div>', unsafe_allow_html=True)
    if st.button("Bersihkan Riwayat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # UI Upload
    st.markdown('<div class="sidebar-heading">Upload Jurnal</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih PDF", type="pdf")

    # Eksekusi Upload
    if uploaded_file and st.button("Proses PDF", use_container_width=True):
        with st.spinner("Memproses dokumen..."):
            st.cache_resource.clear()
            
            st.session_state.db_path = tempfile.mkdtemp()
            DB_PATH = st.session_state.db_path
                    
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            
            embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            Chroma.from_documents(chunks, embedder, persist_directory=DB_PATH)
            
            os.remove(tmp_path)
            
            st.success("Jurnal berhasil dipelajari! Ingatan lama telah di-reset.")
            st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><div class="app-mark">R</div><div class="app-title">Agentic RAG</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Tanya jawab berbasis dokumen, dengan tahap penyaringan relevansi otomatis.</div>',
    unsafe_allow_html=True,
)

key_ready = bool(os.environ.get("GROQ_API_KEY"))
st.markdown(
    f'<div class="status-pill"><span class="status-dot{"" if key_ready else " offline"}"></span>'
    f'{"Terhubung ke Groq" if key_ready else "Masukkan API key di sidebar"}</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Agent setup
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_agent(model: str, db_path: str): 
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embedder) 
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # Mode streaming diaktifkan pada ChatGroq
    llm = ChatGroq(model=model, temperature=0, streaming=True)

    class GraphState(TypedDict):
        question: str
        chat_history: str  
        documents: List[str]
        generation: str
        total_retrieved: int
        relevant_count: int

    def retrieve(state):
        docs = retriever.invoke(state["question"])
        contents = [d.page_content for d in docs]
        return {"documents": contents, "total_retrieved": len(contents)}

    def grade(state):
        prompt = ChatPromptTemplate.from_template(
            "Is relevant? Answer 'yes' or 'no'.\nQuestion: {question}\nDoc: {doc}"
        )
        chain = prompt | llm | StrOutputParser()

        filtered = []
        for doc in state["documents"]:
            score = chain.invoke({"question": state["question"], "doc": doc}).strip().lower()
            if "yes" in score or "ya" in score:
                filtered.append(doc)
        return {"documents": filtered, "relevant_count": len(filtered)}

    # Tambahkan config 
    def generate(state: dict, config: RunnableConfig):
        if not state["documents"]:
            return {
                "generation": "Tidak ditemukan informasi relevan di dokumen untuk pertanyaan ini.",
                "documents": [],
            }

        numbered_context = "\n\n".join(
            f"[{i}] {doc}" for i, doc in enumerate(state["documents"], start=1)
        )
        prompt = ChatPromptTemplate.from_template(
            "Answer ONLY based on the numbered context below. "
            "Whenever you use information from a context chunk, cite it inline "
            "using its number in square brackets, e.g. [1] or [2][3]. "
            "Do not cite numbers that don't exist in the context.\n\n"
            "Chat History:\n{chat_history}\n\n"  
            "Context:\n{context}\n\nQuestion: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        
        return {
            "generation": chain.invoke({
                "question": state["question"], 
                "context": numbered_context,
                "chat_history": state.get("chat_history", "")  
            }, config=config),
            "documents": state["documents"],
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# ----------------------------------------------------------------------------
# Render helpers
# ----------------------------------------------------------------------------
def render_confidence_pill(relevant: int, total: int) -> str:
    if total == 0:
        return ""
    ratio = relevant / total
    tier = "high" if ratio >= 0.66 else ("mid" if ratio > 0 else "low")
    return (
        f'<div class="confidence-pill {tier}">'
        f'{relevant} dari {total} dokumen yang diambil dinilai relevan</div>'
    )

def linkify_citations(answer: str, anchor_prefix: str) -> str:
    def repl(match):
        n = match.group(1)
        return f'<a href="#{anchor_prefix}-{n}" class="citation-badge">{n}</a>'
    return re.sub(r"\[(\d+)\]", repl, answer)

def render_sources_block(sources: List[str], anchor_prefix: str) -> str:
    cards = "".join(
        f'<div class="source-card" id="{anchor_prefix}-{i}">'
        f'<div class="source-label">Dokumen {i}</div>{doc[:400]}...</div>'
        for i, doc in enumerate(sources, start=1)
    )
    return (
        f'<details class="source-block"><summary>Sumber ({len(sources)} dokumen)</summary>{cards}</details>'
    )

# ----------------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0

AVATARS = {"user": "👤", "assistant": "🤖"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "assistant" and msg.get("sources"):
            anchor_prefix = msg["id"]
            st.markdown(
                render_confidence_pill(msg.get("relevant_count", 0), msg.get("total_retrieved", 0)),
                unsafe_allow_html=True,
            )
            st.markdown(linkify_citations(msg["content"], anchor_prefix), unsafe_allow_html=True)
            st.markdown(render_sources_block(msg["sources"], anchor_prefix), unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------------
query = st.chat_input("Tanya sesuatu tentang dokumen kamu...")

if query:
    if not key_ready:
        st.error("Masukkan Groq API key di sidebar terlebih dahulu.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        
        status_indicator = st.empty()
        status_indicator.markdown("⏳ *Mencari dan menyaring dokumen relevan...*")
        
        pill_container = st.empty()
        stream_container = st.empty()
        sources_container = st.empty()
        
        app = init_agent(model_name, DB_PATH)
        
        # Siapkan handler streaming 
        stream_handler = StreamHandler(stream_container, status_indicator)
        
        # Kumpulkan histori chat sebelumnya 
        history_str = ""
        for m in st.session_state.messages[:-1][-4:]:
            role = "User" if m["role"] == "user" else "AI"
            history_str += f"{role}: {m['content']}\n"
        
        # Jalankan pipeline (Graph)
        result = app.invoke(
            {
                "question": query,
                "chat_history": history_str  
            },
            config={"callbacks": [stream_handler]}
        )
        
        # Bersihkan sisa indikator 
        status_indicator.empty()
        
        answer = result["generation"]
        sources = result.get("documents", [])
        total_retrieved = result.get("total_retrieved", len(sources))
        relevant_count = result.get("relevant_count", len(sources))

        st.session_state.msg_counter += 1
        anchor_prefix = f"msg{st.session_state.msg_counter}"

        # Timpa wadah streaming 
        if sources:
            pill_container.markdown(render_confidence_pill(relevant_count, total_retrieved), unsafe_allow_html=True)
        
        stream_container.markdown(linkify_citations(answer, anchor_prefix), unsafe_allow_html=True)
        
        if sources:
            sources_container.markdown(render_sources_block(sources, anchor_prefix), unsafe_allow_html=True)

    # Simpan ke memori sesi
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "id": anchor_prefix,
        "total_retrieved": total_retrieved,
        "relevant_count": relevant_count,
    })
