from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Config
PDF_PATH = "jurnal.pdf"
DB_PATH = "./chroma_db"

# Load
print("Membaca PDF...")
docs = PyPDFLoader(PDF_PATH).load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
chunks = splitter.split_documents(docs)

# Embed
print("Memuat embedding...")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store
print("Menyimpan vektor...")
Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=DB_PATH)
print("Selesai! ")