import os
from typing import TypedDict, List
from dotenv import load_dotenv  # <-- Import dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#load
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(" FATAL ERROR: GROQ_API_KEY tidak ditemukan! Pastikan file .env sudah diisi dengan benar.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

#Config
DB_PATH = "./chroma_db"

#DB
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedder)
retriever = db.as_retriever(search_kwargs={"k": 3})

#LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# State
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

#Nodes
def retrieve(state):
    print("-> RETRIEVE")
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

def grade(state):
    print("-> GRADE (Batch Processing)")
    prompt = ChatPromptTemplate.from_template(
        "Is this document relevant? Answer 'yes' or 'no' only.\nQuestion: {question}\nDocument: {doc}"
    )
    chain = prompt | llm | StrOutputParser()
    
    # UPGRADE: BATCH PROCESSING
    # Eksekusi semua dokumen secara paralel, bukan satu-satu lagi!
    if not state["documents"]:
        return {"documents": []}
        
    inputs = [{"question": state["question"], "doc": doc} for doc in state["documents"]]
    scores = chain.batch(inputs)  # <-- Ngebut di sini!
    
    filtered_docs = []
    for doc, score in zip(state["documents"], scores):
        if "yes" in score.strip().lower():
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs}

def generate(state):
    print("-> GENERATE")

    
    if not state["documents"]:
        return {"generation": "Out of context. No relevant documents found."}

  
    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY based on context.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    
    chain = prompt | llm | StrOutputParser()
    
    
    context_text = "\n\n".join(state["documents"])
    answer = chain.invoke({"question": state["question"], "context": context_text})
    
    return {"generation": answer}

#Build
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("generate", generate)

#Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)

#Compile
app = workflow.compile()

if __name__ == "__main__":
    print("\n🚀 Stress Test Lokal...")
    query = "What is the specification of Xiaomi 11T Pro?"
    result = app.invoke({"question": query})

    # Output
    print("\n===========================")
    print(" JAWABAN:")
    print("===========================")
    print(result["generation"])