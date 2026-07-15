from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Config
DB_PATH = "./chroma_db"
QUERY = "What is Retrieval-Augmented Generation?"

# Load DB
print("Memuat database...")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embedder)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Search
print(f"\nMencari: '{QUERY}'\n")
results = retriever.invoke(QUERY)

# Print
for i, doc in enumerate(results):
    print(f"--- Hasil {i+1} (Hal {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content[:300] + "...\n")