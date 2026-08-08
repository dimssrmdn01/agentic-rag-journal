<div align="center">
  <h1>🤖 Agentic RAG Journal</h1>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI"></a>
    <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-1C3C3C.svg?logo=langchain&logoColor=white" alt="LangChain"></a>
    <a href="https://groq.com/"><img src="https://img.shields.io/badge/Groq-F55036.svg?logo=groq&logoColor=white" alt="Groq"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </p>

  <p><em>Asisten tanya-jawab dokumen cerdas dengan alur <b>Self-Correction</b> menggunakan LangGraph.</em></p>
</div>

---

Bukan sekadar sistem *retrieve-then-generate* biasa. **Agentic RAG Journal** menyisipkan tahap *relevance grading* (penyaringan relevansi) berbasis **Pydantic Structured Output** dan **Batch Processing** secara otomatis agar jawaban AI selalu akurat, tahan banting, dan **tidak berhalusinasi** di luar konteks dokumen.

Tersedia dalam dua arsitektur pilihan: **Dashboard Streamlit** (All-in-One) dan **FastAPI + HTML Custom** (Decoupled Architecture dengan Cyberpunk UI).

## Cara Kerja Pipeline

Alih-alih langsung menjawab, agen AI akan melewati tahap penyaringan ketat:

📄 **PDF** ➔ 🧩 **Ingest & Chunking** ➔ 🗄️ **Vector Store (ChromaDB)**
<br>
💬 **Pertanyaan** ➔ 🔍 **Retrieve** ➔ ⚖️ **Relevance Grade (Pydantic + Batch)** ➔ 📝 **Generate** ➔ ✨ **Jawaban**

1. **Retrieve:** Mengambil `k` potongan dokumen yang paling mirip secara semantik dengan pertanyaan pengguna.
2. **Grade:** Setiap potongan dokumen dievaluasi secara paralel (`chain.batch()`) dan divalidasi mutlak menggunakan skema **Pydantic**. Teks yang tidak relevan akan langsung **dibuang** untuk mencegah halusinasi.
3. **Generate:** Menyusun jawaban akhir murni dari potongan dokumen yang lolos seleksi. Jika tidak lolos, sistem secara jujur menyatakan keluar dari konteks (*Out of Context*).

## Fitur Utama

- **Ingesti Dokumen Otomatis:** Ekstraksi dan pemecahan (*chunking*) teks PDF langsung ke *vector database* ChromaDB.
- **Anti-Halusinasi & Pydantic Validation:** Memaksa LLM memberikan respons terstruktur, mencegah *error parsing*, dan menyaring informasi di luar dokumen.
- **Dual Interface:**
  - **Streamlit (`app.py`):** Dashboard UI siap pakai lengkap dengan *Confidence Indicator* dan sitasi dokumen interaktif.
  - **FastAPI + HTML Custom (`api.py` + `index.html`):** Arsitektur *client-server* terpisah dengan *endpoint* REST API (`/upload` dan `/chat`), lengkap dengan *Confidence Score* (`SYS_EVAL`) dan pratinjau sumber dokumen.
- **CLI Tools:** Termasuk skrip mandiri (`ingest.py`, `retrieve.py`, `agent.py`) untuk eksperimen atau *testing* tiap komponen secara terpisah.

## Tech Stack

| Kategori | Teknologi |
| :--- | :--- |
| **Orchestration** | LangGraph, Pydantic |
| **LLM** | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Database** | ChromaDB |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend / UI** | Streamlit, HTML5, Tailwind CSS, Vanilla JS |
| **Document Processing**| `langchain-community`, `pypdf`, `python-dotenv` |

## Struktur Proyek

```text
agentic-rag-journal/
 ├── app.py             # Antarmuka Streamlit (chat UI + citation)
 ├── api.py             # Backend FastAPI (endpoint /upload dan /chat)
 ├── index.html         # Frontend custom (Tailwind + JS)
 ├── agent.py           # Script CLI untuk test pipeline penuh
 ├── ingest.py          # Script CLI untuk pipeline embedding ke ChromaDB
 ├── retrieve.py        # Script CLI untuk test akurasi pencarian
 ├── requirements.txt   # Daftar dependensi library
 ├── .env               # Kunci rahasia Groq API (Ignored by Git)
 └── chroma_db/         # Folder database vektor lokal
```

## Panduan Instalasi

### 1. Clone Repositori

```bash
git clone https://github.com/dimssrmdn01/agentic-rag-journal.git
cd agentic-rag-journal
```

### 2. Buat Virtual Environment (opsional tapi direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate    # Pengguna Windows: venv\Scripts\activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi API Key

Buat file bernama `.env` di dalam folder utama proyek dan masukkan API Key dari Groq Console:

```
GROQ_API_KEY=gsk_kunci_rahasia_kamu_di_sini
```

## Cara Menjalankan Aplikasi

Kamu dapat memilih salah satu dari dua antarmuka yang disediakan:

**Opsi A - Menggunakan Streamlit (All-in-One)**

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di `localhost:8501`.

**Opsi B - Menggunakan FastAPI & HTML Custom (Client-Server)**

Jalankan server backend terlebih dahulu:

```bash
uvicorn api:app --reload
```

Lalu buka file `index.html` melalui Live Server atau browser kesayanganmu.

**Opsi C - Mode CLI (Untuk Testing/Eksperimen)**

```bash
python retrieve.py   # Uji coba kecepatan dan akurasi retrieval
python agent.py      # Uji coba pipeline penuh di terminal
```

## Keterbatasan

- Unggahan dokumen baru akan ditambahkan (*append*) ke dalam vector database yang sama. Saat ini dioptimalkan untuk membedah satu topik/buku pada satu waktu.
- Data tersimpan secara lokal tanpa mekanisme manajemen sesi pengguna ganda (multi-user).

## Roadmap Pengembangan

- [ ] Implementasi Streaming Response (jawaban LLM mengalir kata per kata).
- [ ] Manajemen multi-dokumen (AI dapat membedakan dan merujuk dari buku yang berbeda).
- [ ] Integrasi database relasional untuk riwayat percakapan persisten.
- [ ] Fitur ringkasan eksekutif instan saat dokumen pertama kali diunggah.

## Lisensi & Atribusi

Proyek ini menggunakan Lisensi MIT. Kamu sangat dipersilakan untuk menggunakan, memodifikasi, dan mendistribusikannya secara bebas.

Dibuat dengan ☕ dan antusiasme terhadap Data Science oleh Dimas Arya Ramadhan.
