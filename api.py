import os
import tempfile
from typing import List, TypedDict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
if not (groq_api_key := os.getenv("GROQ_API_KEY")):
    raise ValueError("GROQ_API_KEY is missing in .env")
os.environ["GROQ_API_KEY"] = groq_api_key

DB_PATH = "./chroma_db"
EMBEDDER = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

app = FastAPI(title="Agentic RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class GradeResult(BaseModel):
    is_relevant: str = Field(description="Jawab 'yes' jika relevan, jika tidak 'no'")

class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

def get_db():
    return Chroma(persist_directory=DB_PATH, embedding_function=EMBEDDER)

def retrieve(state: GraphState):
    retriever = get_db().as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

def grade(state: GraphState):
    if not state["documents"]:
        return {"documents": []}
        
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Is relevant? Answer 'yes' or 'no'.\nQuestion: {question}\nDoc: {doc}"
    )
    chain = prompt | llm.with_structured_output(GradeResult)
    
    inputs = [{"question": state["question"], "doc": doc} for doc in state["documents"]]
    scores = chain.batch(inputs)
    
    filtered = [doc for doc, score in zip(state["documents"], scores) if score.is_relevant.lower() == "yes"]
    return {"documents": filtered}

def generate(state: GraphState):
    if not state["documents"]:
        return {"generation": "Tidak ditemukan informasi relevan di dokumen."}
        
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY based on context.\n\nContext: {context}\n\nQuestion: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    return {"generation": chain.invoke({"question": state["question"], "context": "\n\n".join(state["documents"])})}

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)
app_agent = workflow.compile()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
            
        loader = PyPDFLoader(tmp_path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
        Chroma.from_documents(chunks, EMBEDDER, persist_directory=DB_PATH)
        os.remove(tmp_path)
        
        return {"status": "success", "message": "Jurnal berhasil dipelajari!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        result = app_agent.invoke({"question": request.question})
        docs = result.get("documents", [])
        
        return {
            "answer": result["generation"],
            "sources": [f"{doc[:150]}..." for doc in docs],
            "confidence": f"HIGH ({len(docs)} Verified Sources)" if docs else "LOW (Out of Context)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))