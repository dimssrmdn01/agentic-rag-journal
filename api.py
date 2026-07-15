import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Setup API
app = FastAPI(title="Agentic RAG API")

# Izinkan akses dari file HTML (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "./chroma_db"

# API Key Groq
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Format request dari HTML
class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Simpan PDF sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        # Proses PDF
        loader = PyPDFLoader(tmp_path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        
        # Embed & Simpan
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(chunks, embedder, persist_directory=DB_PATH)
        
        os.remove(tmp_path)
        return {"status": "success", "message": "Jurnal berhasil dipelajari!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        # Setup Agent
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=DB_PATH, embedding_function=embedder)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        class GraphState(TypedDict):
            question: str
            documents: List[str]
            generation: str

        def retrieve(state):
            docs = retriever.invoke(state["question"])
            return {"documents": [d.page_content for d in docs]}

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
            return {"documents": filtered}

        def generate(state):
            if not state["documents"]:
                return {"generation": "Tidak ditemukan informasi relevan di dokumen untuk pertanyaan ini."}
            prompt = ChatPromptTemplate.from_template(
                "Answer ONLY based on context.\n\nContext: {context}\n\nQuestion: {question}"
            )
            chain = prompt | llm | StrOutputParser()
            context_text = "\n\n".join(state["documents"])
            return {"generation": chain.invoke({"question": state["question"], "context": context_text})}

        # LangGraph Workflow
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade", grade)
        workflow.add_node("generate", generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        workflow.add_edge("grade", "generate")
        workflow.add_edge("generate", END)
        app_agent = workflow.compile()

        # Eksekusi
        result = app_agent.invoke({"question": request.question})
        
        return {
            "answer": result["generation"],
            "sources": result.get("documents", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))