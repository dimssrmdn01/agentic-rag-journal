import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Config
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
DB_PATH = "./chroma_db"

# DB
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedder)
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# State
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# Nodes
def retrieve(state):
    print("-> RETRIEVE")
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

def grade(state):
    print("-> GRADE")
    prompt = ChatPromptTemplate.from_template(
        "Is this document relevant? Answer 'yes' or 'no' only.\nQuestion: {question}\nDocument: {doc}"
    )
    chain = prompt | llm | StrOutputParser()
    
    filtered_docs = []
    for doc in state["documents"]:
        score = chain.invoke({"question": state["question"], "doc": doc}).strip().lower()
        if "yes" in score:
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs}

def generate(state):
    print("-> GENERATE")
    
    # Guard clause
    if not state["documents"]:
        return {"generation": "Out of context."}
        
    # Prompting
    prompt = ChatPromptTemplate.from_template(
        "Answer ONLY based on context.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    # Chain
    chain = prompt | llm | StrOutputParser()
    
    # Execute
    context_text = "\n\n".join(state["documents"])
    answer = chain.invoke({"question": state["question"], "context": context_text})
    
    return {"generation": answer}

# Build
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("generate", generate)

# Edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Test
print("\n Stress Test...")
query = "What is the specification of Xiaomi 11T Pro?"
result = app.invoke({"question": query})

# Output
print("\n===========================")
print(" JAWABAN:")
print("===========================")
print(result["generation"])