import os
import tempfile
import re
import base64
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
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

# ===========================================================
# 1. INIT & ENV
# ===========================================================
load_dotenv()

if "db_path" not in st.session_state:
    st.session_state.db_path = tempfile.mkdtemp()
DB_PATH = st.session_state.db_path

# ===========================================================
# 2. SVG ICON SYSTEM (CYBERPUNK THEME)
# ===========================================================
_ICON_PATHS = {
    "cpu": (
        '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>'
        '<rect x="9" y="9" width="6" height="6"/>'
        '<line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>'
        '<line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>'
        '<line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>'
        '<line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>', "stroke"
    ),
    "user-neon": (
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>', "stroke"
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
        'a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33'
        'l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 '
        '0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51'
        'V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 '
        '0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>', "stroke"
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>', "stroke"
    ),
    "refresh": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>', "stroke"
    ),
    "document": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>', "stroke"
    ),
    "link": (
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
        '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>', "stroke"
    )
}

def icon_svg(name: str, size: int = 18, color: str = "currentColor") -> str:
    body, kind = _ICON_PATHS[name]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )

def icon_data_uri(name: str, size: int = 64, color: str = "%2300F0FF") -> str:
    body, kind = _ICON_PATHS[name]
    color_raw = color.replace("%23", "#")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color_raw}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

def icon_label(name: str, text: str, size: int = 15, color: str = "#00F0FF") -> str:
    return f'<span class="icon-inline">{icon_svg(name, size, color)}<span>{text}</span></span>'

ASSISTANT_AVATAR = icon_data_uri("cpu", size=64, color="%2300F0FF")
USER_AVATAR = icon_data_uri("user-neon", size=64, color="%23FF007C")

# ===========================================================
# 3. STREAM HANDLER
# ===========================================================
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, status_indicator):
        self.container = container
        self.status_indicator = status_indicator
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.status_indicator:
            self.status_indicator.empty()
            self.status_indicator = None
        self.text += token
        self.container.markdown(self.text + "█") 

# ===========================================================
# 4. PAGE CONFIG & CYBERPUNK CSS
# ===========================================================
st.set_page_config(
    page_title="RAG CyberAgent",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap');

        :root {
            --bg-base: #050814;
            --bg-panel: rgba(10, 17, 40, 0.7);
            --border-glow: #00F0FF;
            --border-dim: #1A2B4C;
            --accent-cyan: #00F0FF;
            --accent-pink: #FF007C;
            --accent-blue: #0047FF;
            --text-main: #E0F7FA;
            --text-muted: #8A9CA8;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text-main); }
        h1, h2, h3, .cyber-title { font-family: 'Orbitron', sans-serif !important; }
        
        #MainMenu, footer { visibility: hidden; }
        .stDeployButton { display: none; }

        /* Cyberpunk Grid Background */
        .stApp {
            background-color: var(--bg-base);
            background-image: 
                linear-gradient(rgba(0, 240, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.03) 1px, transparent 1px);
            background-size: 30px 30px;
            background-position: center center;
        }

        .icon-inline { display: inline-flex; align-items: center; gap: 0.5rem; vertical-align: middle; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--accent-cyan); }
        .icon-inline svg { flex-shrink: 0; display: block; filter: drop-shadow(0 0 3px var(--accent-cyan)); }

        /* App Header */
        .cyber-header { display: flex; align-items: center; gap: 15px; margin-bottom: 5px; padding-bottom: 10px; border-bottom: 1px solid var(--border-dim); }
        .cyber-logo { width: 40px; height: 40px; border-radius: 4px; background: rgba(0, 240, 255, 0.1); border: 1px solid var(--accent-cyan); display: flex; align-items: center; justify-content: center; box-shadow: 0 0 10px rgba(0, 240, 255, 0.2); }
        .cyber-title { font-size: 1.8rem; font-weight: 700; color: #FFF; text-shadow: 0 0 10px var(--accent-cyan), 0 0 20px var(--accent-blue); letter-spacing: 2px; margin: 0; }
        .cyber-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--text-muted); margin: 8px 0 1.5rem 0; letter-spacing: 0.5px; }

        /* Status Indicators */
        .status-pill { display: inline-flex; align-items: center; gap: 8px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--accent-cyan); background: rgba(0, 240, 255, 0.05); border: 1px solid var(--accent-cyan); border-radius: 2px; padding: 4px 12px; margin-bottom: 1.5rem; box-shadow: inset 0 0 5px rgba(0,240,255,0.2); text-transform: uppercase; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-cyan); box-shadow: 0 0 8px var(--accent-cyan); animation: pulse 2s infinite; }
        .status-dot.offline { background: var(--accent-pink); box-shadow: 0 0 8px var(--accent-pink); }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

        /* Sidebar Styling */
        [data-testid="stSidebar"] { background: var(--bg-panel); border-right: 1px solid var(--border-glow); box-shadow: 5px 0 20px rgba(0, 240, 255, 0.05); }
        [data-testid="stSidebar"] * { color: var(--text-main); }
        [data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div { background: rgba(0,0,0,0.5) !important; border: 1px solid var(--border-dim) !important; color: var(--accent-cyan) !important; border-radius: 2px; font-family: 'IBM Plex Mono', monospace; }
        [data-testid="stTextInput"] input:focus { border-color: var(--accent-cyan) !important; box-shadow: 0 0 8px rgba(0,240,255,0.3) !important; }
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: var(--accent-cyan) !important; box-shadow: 0 0 10px var(--accent-cyan); }
        
        /* Buttons */
        .stButton > button { border-radius: 2px; font-family: 'Orbitron', sans-serif; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; border: 1px solid var(--accent-cyan); background: rgba(0, 240, 255, 0.05); color: var(--accent-cyan); transition: all 0.2s ease; }
        .stButton > button:hover { background: var(--accent-cyan); color: #000; box-shadow: 0 0 15px var(--accent-cyan); }

        /* File Uploader */
        [data-testid="stFileUploaderDropzone"] { background: rgba(0,0,0,0.4) !important; border: 1px dashed var(--accent-cyan) !important; border-radius: 2px !important; }

        /* Chat Bubbles */
        [data-testid="stChatMessage"] { background: transparent; border: none; padding: 0.5rem 0; gap: 15px; }
        [data-testid="stChatMessageContent"] { background: var(--bg-panel); border: 1px solid var(--border-dim); border-radius: 2px; padding: 14px 18px; color: var(--text-main); font-size: 0.95rem; line-height: 1.6; border-left: 2px solid var(--accent-cyan); }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { border-left: 2px solid var(--accent-pink); }

        /* Chat Input */
        [data-testid="stChatInput"] { background: var(--bg-panel); border: 1px solid var(--border-dim); border-radius: 2px; }
        [data-testid="stChatInput"] textarea { color: var(--accent-cyan) !important; font-family: 'IBM Plex Mono', monospace; }
        [data-testid="stChatInput"] textarea:focus { box-shadow: inset 0 0 10px rgba(0,240,255,0.1); }

        /* Citations & Sources */
        .confidence-pill { display: inline-flex; align-items: center; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; padding: 4px 10px; border-radius: 2px; margin-bottom: 12px; border: 1px solid var(--border-dim); text-transform: uppercase; letter-spacing: 0.5px; }
        .confidence-pill.high { color: var(--accent-cyan); background: rgba(0,240,255,0.05); border-color: var(--accent-cyan); }
        .confidence-pill.mid { color: #FFAA00; background: rgba(255,170,0,0.05); border-color: #FFAA00; }
        .confidence-pill.low { color: var(--accent-pink); background: rgba(255,0,124,0.05); border-color: var(--accent-pink); }
        
        .citation-badge { display: inline-flex; align-items: center; justify-content: center; font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 700; min-width: 18px; height: 18px; padding: 0 4px; border-radius: 2px; background: rgba(0,240,255,0.1); border: 1px solid var(--accent-cyan); color: var(--accent-cyan); text-decoration: none !important; margin: 0 2px; transform: translateY(-2px); transition: all .2s ease; }
        .citation-badge:hover { background: var(--accent-cyan); color: #000; box-shadow: 0 0 8px var(--accent-cyan); }

        details.source-block { border: 1px solid var(--border-dim); border-radius: 2px; padding: 4px; margin-top: 10px; background: rgba(0,0,0,0.3); }
        details.source-block summary { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--accent-cyan); padding: 8px; cursor: pointer; text-transform: uppercase; letter-spacing: 1px; }
        .source-card { background: var(--bg-base); border: 1px solid var(--border-dim); border-left: 2px solid var(--accent-cyan); border-radius: 2px; padding: 12px; margin-bottom: 8px; font-size: 0.82rem; color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; }
        .source-label { font-size: 0.65rem; color: var(--accent-cyan); text-transform: uppercase; margin-bottom: 6px; font-weight: 600; letter-spacing: 1px; }
        details.source-block .source-card:target { border-color: var(--accent-pink); box-shadow: inset 0 0 10px rgba(255,0,124,0.1); }
    </style>
    """, unsafe_allow_html=True
)

# ===========================================================
# 5. SIDEBAR
# ===========================================================
with st.sidebar:
    st.markdown(f'<div style="margin: 1.5rem 0 0.8rem 0;">{icon_label("settings", "Konfigurasi Sistem")}</div>', unsafe_allow_html=True)

    default_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input("API_KEY", value="", type="password", placeholder="gsk_...")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    model_name = st.selectbox("LLM_CORE", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    top_k = st.slider("K_RETRIEVAL", min_value=1, max_value=8, value=3)

    st.markdown('<hr style="border-color: #1A2B4C;">', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-bottom: 0.8rem;">{icon_label("info", "Data Ingestion")}</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file and st.button("INISIALISASI DATA", use_container_width=True):
        with st.spinner("MEMPROSES DOKUMEN..."):
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
            st.success("DATABASE TERBARUI!")
            st.rerun()

    st.markdown('<hr style="border-color: #1A2B4C;">', unsafe_allow_html=True)
    if st.button("RESET MEMORY CACHE", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ===========================================================
# 6. APP HEADER
# ===========================================================
st.markdown(
    f'<div class="cyber-header">'
    f'<div class="cyber-logo">{icon_svg("cpu", 24, "#00F0FF")}</div>'
    f'<h1 class="cyber-title">AGENTIC RAG</h1>'
    f'</div>', unsafe_allow_html=True
)
st.markdown('<div class="cyber-subtitle">NEURAL KNOWLEDGE RETRIEVAL & FILTERING SYS_</div>', unsafe_allow_html=True)

key_ready = bool(os.environ.get("GROQ_API_KEY"))
dot_class = "" if key_ready else " offline"
status_text = "SYS_ONLINE (GROQ LINKED)" if key_ready else "SYS_OFFLINE (AWAITING API KEY)"
st.markdown(
    f'<div class="status-pill"><span class="status-dot{dot_class}"></span>{status_text}</div>', 
    unsafe_allow_html=True
)

# ===========================================================
# 7. AGENT SETUP (LANGGRAPH)
# ===========================================================
@st.cache_resource(show_spinner=False)
def init_agent(model: str, db_path: str): 
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embedder) 
    retriever = db.as_retriever(search_kwargs={"k": top_k})
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
        prompt = ChatPromptTemplate.from_template("Is relevant? Answer 'yes' or 'no'.\nQuestion: {question}\nDoc: {doc}")
        chain = prompt | llm | StrOutputParser()
        filtered = []
        for doc in state["documents"]:
            score = chain.invoke({"question": state["question"], "doc": doc}).strip().lower()
            if "yes" in score or "ya" in score:
                filtered.append(doc)
        return {"documents": filtered, "relevant_count": len(filtered)}

    def generate(state: dict, config: RunnableConfig):
        if not state["documents"]:
            return {"generation": "SYS_ERR: Data relevan tidak ditemukan dalam knowledge base.", "documents": []}
        numbered_context = "\n\n".join(f"[{i}] {doc}" for i, doc in enumerate(state["documents"], start=1))
        prompt = ChatPromptTemplate.from_template(
            "Answer ONLY based on the numbered context below. Cite it inline using its number in square brackets, e.g. [1].\n\n"
            "Chat History:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        return {
            "generation": chain.invoke({"question": state["question"], "context": numbered_context, "chat_history": state.get("chat_history", "")}, config=config),
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

# ===========================================================
# 8. RENDER HELPERS
# ===========================================================
def render_confidence_pill(relevant: int, total: int) -> str:
    if total == 0: return ""
    ratio = relevant / total
    tier = "high" if ratio >= 0.66 else ("mid" if ratio > 0 else "low")
    return f'<div class="confidence-pill {tier}">RELEVANCE FILTER: {relevant}/{total} BLOCKS ACCEPTED</div>'

def linkify_citations(answer: str, anchor_prefix: str) -> str:
    def repl(match):
        n = match.group(1)
        return f'<a href="#{anchor_prefix}-{n}" class="citation-badge">{n}</a>'
    return re.sub(r"\[(\d+)\]", repl, answer)

def render_sources_block(sources: List[str], anchor_prefix: str) -> str:
    cards = "".join(
        f'<div class="source-card" id="{anchor_prefix}-{i}"><div class="source-label">{icon_svg("document", 12)} DATA_BLOCK_{i}</div>{doc[:400]}...</div>'
        for i, doc in enumerate(sources, start=1)
    )
    return f'<details class="source-block"><summary>{icon_svg("link", 14)} SOURCE_NODES ({len(sources)})</summary>{cards}</details>'

# ===========================================================
# 9. CHAT HISTORY
# ===========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0

AVATARS = {"user": USER_AVATAR, "assistant": ASSISTANT_AVATAR}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "assistant" and msg.get("sources"):
            anchor_prefix = msg["id"]
            st.markdown(render_confidence_pill(msg.get("relevant_count", 0), msg.get("total_retrieved", 0)), unsafe_allow_html=True)
            st.markdown(linkify_citations(msg["content"], anchor_prefix), unsafe_allow_html=True)
            st.markdown(render_sources_block(msg["sources"], anchor_prefix), unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# ===========================================================
# 10. CHAT INPUT
# ===========================================================
query = st.chat_input("TRANSMIT_QUERY...")

if query:
    if not key_ready:
        st.error("SYS_ERR: GROQ API KEY MISSING. INPUT REQUIRED IN CONSOLE.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        status_indicator = st.empty()
        status_indicator.markdown("`< SCANNING DATABASES... >`", unsafe_allow_html=True)
        
        pill_container = st.empty()
        stream_container = st.empty()
        sources_container = st.empty()
        
        app = init_agent(model_name, DB_PATH)
        stream_handler = StreamHandler(stream_container, status_indicator)
        
        history_str = "".join([f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}\n" for m in st.session_state.messages[:-1][-4:]])
        
        result = app.invoke(
            {"question": query, "chat_history": history_str},
            config={"callbacks": [stream_handler]}
        )
        
        status_indicator.empty()
        
        answer = result["generation"]
        sources = result.get("documents", [])
        total_retrieved = result.get("total_retrieved", len(sources))
        relevant_count = result.get("relevant_count", len(sources))

        st.session_state.msg_counter += 1
        anchor_prefix = f"msg{st.session_state.msg_counter}"

        if sources:
            pill_container.markdown(render_confidence_pill(relevant_count, total_retrieved), unsafe_allow_html=True)
        
        stream_container.markdown(linkify_citations(answer, anchor_prefix), unsafe_allow_html=True)
        
        if sources:
            sources_container.markdown(render_sources_block(sources, anchor_prefix), unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "id": anchor_prefix,
        "total_retrieved": total_retrieved,
        "relevant_count": relevant_count,
    })
