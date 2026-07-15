# Agentic RAG Journal

Asisten tanya-jawab berbasis dokumen (PDF) yang dibangun dengan pendekatan **agentic RAG** — bukan sekadar retrieve-then-generate biasa, tapi menyisipkan tahap *relevance grading* di antaranya agar jawaban tidak mengarang di luar konteks dokumen.

Tersedia dalam dua antarmuka: dashboard **Streamlit** siap pakai, dan **API FastAPI + frontend HTML custom** untuk yang ingin arsitektur backend-frontend terpisah.

## Cara Kerja

```
PDF  ->  Ingest & Chunking  ->  Vector Store (ChromaDB)
                                        |
Pertanyaan  ->  Retrieve  ->  Relevance Grading  ->  Generate  ->  Jawaban
```

1. **Retrieve** — mengambil `k` potongan dokumen paling mirip secara semantik dengan pertanyaan, dari ChromaDB.
2. **Grade** — tiap potongan dokumen dinilai ulang oleh LLM: relevan atau tidak dengan pertanyaan. Yang tidak relevan dibuang sebelum sampai ke tahap generate.
3. **Generate** — jawaban disusun hanya dari potongan dokumen yang lolos penyaringan. Kalau tidak ada yang relevan, sistem terus terang bilang tidak menemukan informasinya alih-alih mengarang.

Alur ini diimplementasikan sebagai graph menggunakan **LangGraph**.

## Fitur

- Ingesti PDF ke vector database (ChromaDB) dengan chunking otomatis.
- Pipeline agentic RAG: retrieve → grade → generate, mengurangi risiko halusinasi.
- Dua pilihan antarmuka:
  - **Streamlit** (`app.py`) — chat UI lengkap dengan *confidence indicator* (menampilkan berapa dari dokumen yang diambil benar-benar dinilai relevan) dan **sitasi yang bisa diklik** (`[1]`, `[2]`, dst di jawaban langsung mengarah ke potongan dokumen sumbernya).
  - **FastAPI + HTML** (`api.py` + `index.html`) — arsitektur backend/frontend terpisah dengan endpoint `/upload` dan `/chat`, cocok untuk didemokan sebagai REST API atau diintegrasikan ke frontend lain.
- Script command-line mandiri (`ingest.py`, `retrieve.py`, `agent.py`) untuk menjalankan/menguji tiap tahap pipeline secara terpisah, tanpa perlu UI.

> Catatan status: confidence indicator dan clickable citation saat ini baru tersedia di versi Streamlit. Versi FastAPI/HTML masih di tahap sebelumnya dan belum menyertakan dua fitur itu.

## Struktur Proyek

```
agentic-rag-journal/
  app.py            # Antarmuka Streamlit (chat UI + confidence indicator + citation)
  api.py             # Backend FastAPI (endpoint /upload dan /chat)
  index.html         # Frontend custom untuk dipasangkan dengan api.py
  agent.py           # Script CLI untuk menguji pipeline retrieve-grade-generate
  ingest.py          # Script CLI untuk memasukkan PDF ke ChromaDB
  retrieve.py        # Script CLI untuk menguji tahap retrieval saja
  requirements.txt
  .env               # Berisi GROQ_API_KEY (jangan pernah di-commit)
  chroma_db/         # Vector store persisten (di-generate otomatis, di-gitignore)
```

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB |
| Backend API | FastAPI |
| UI | Streamlit, HTML/CSS/JS (Tailwind) |
| Loader/Splitter | `langchain-community`, `langchain-text-splitters`, `pypdf` |

## Instalasi

```bash
git clone <url-repo-kamu>
cd agentic-rag-journal

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Buat file `.env` di root proyek:

```
GROQ_API_KEY=your_groq_api_key_here
```

Dapatkan API key gratis di [console.groq.com](https://console.groq.com).

## Cara Menjalankan

### 1. Ingest dokumen ke vector store

Taruh PDF kamu sebagai `jurnal.pdf` di root proyek (atau ubah `PDF_PATH` di `ingest.py`), lalu:

```bash
python ingest.py
```

### 2a. Menjalankan dashboard Streamlit

```bash
streamlit run app.py
```

### 2b. Menjalankan API + frontend custom

```bash
uvicorn api:app --reload
```

Lalu buka `index.html` langsung di browser. Upload PDF dari sidebar, atau pastikan sudah menjalankan `ingest.py` sebelumnya.

### Uji cepat lewat CLI (opsional)

```bash
python retrieve.py   # uji tahap retrieval saja
python agent.py      # uji pipeline penuh retrieve -> grade -> generate
```

## Keterbatasan

- Sitasi `[n]` di jawaban bergantung pada kepatuhan model terhadap instruksi format — model kecil (`llama-3.1-8b-instant`) kadang melewatkan atau salah menomori sitasi.
- Saat ini dirancang untuk satu dokumen aktif dalam satu waktu; upload dokumen baru akan menambah ke vector store yang sama, bukan menggantikannya.
- Vector store disimpan lokal (`chroma_db/`), belum ada mekanisme multi-user/multi-session terpisah.

## Roadmap

- [ ] Menyamakan fitur confidence indicator & citation ke versi FastAPI/HTML.
- [ ] Streaming response (jawaban muncul kata per kata).
- [ ] Dukungan multi-dokumen dengan badge sumber per dokumen.
- [ ] Riwayat percakapan persisten (tersimpan ke database, bukan hanya session).
- [ ] Ringkasan otomatis saat dokumen selesai diunggah.

## Lisensi

MIT — bebas digunakan, dimodifikasi, dan didistribusikan.
