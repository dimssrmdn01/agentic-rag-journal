<div align="center">
  <h1>Agentic RAG Journal</h1>

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

<div align="center">

[Overview](#overview) • [Cara Kerja Pipeline](#cara-kerja-pipeline) • [Fitur Utama](#fitur-utama) • [Tech Stack](#tech-stack) • [Struktur Proyek](#struktur-proyek) • [Instalasi](#panduan-instalasi) • [Menjalankan Aplikasi](#cara-menjalankan-aplikasi) • [Keterbatasan](#keterbatasan) • [Roadmap](#roadmap-pengembangan) • [Lisensi](#lisensi--atribusi)

</div>

---

## Overview

Bukan sekadar sistem *retrieve-then-generate* biasa. **Agentic RAG Journal** menyisipkan tahap *relevance grading* (penyaringan relevansi) secara otomatis agar jawaban AI selalu akurat, tepat sasaran, dan **tidak berhalusinasi** di luar konteks dokumen.

Tersedia dalam dua arsitektur pilihan: **Dashboard Streamlit** (All-in-One) dan **FastAPI + HTML** (Decoupled Architecture).

---

## Cara Kerja Pipeline

Alih-alih langsung menjawab, agen AI akan melewati tahap penyaringan ketat sebelum menghasilkan jawaban akhir:

```text
PDF → Ingest & Chunking → Vector Store (ChromaDB)

Pertanyaan → Retrieve → Relevance Grade → Generate → Jawaban
```

| Tahap | Deskripsi |
|---|---|
| **1. Retrieve** | Mengambil `k` potongan dokumen yang paling mirip secara semantik dengan pertanyaan pengguna. |
| **2. Grade** | Setiap potongan dokumen dievaluasi ulang oleh LLM. Teks yang tidak relevan dengan pertanyaan langsung **dibuang** sebelum sampai ke tahap pembuatan jawaban. |
| **3. Generate** | Menyusun jawaban akhir murni dari potongan dokumen yang lolos seleksi. Jika tidak ada yang lolos, sistem akan secara jujur menyatakan bahwa informasi tidak ditemukan. |

---

## Fitur Utama

- **Ingesti Dokumen Otomatis** — Ekstraksi dan pemecahan (*chunking*) teks PDF langsung ke *vector database* ChromaDB.
- **Anti-Halusinasi** — Pipeline *Retrieve → Grade → Generate* meminimalisir jawaban ngawur.
- **Dual Interface:**
  - **Streamlit (`app.py`)** — Dashboard UI siap pakai lengkap dengan *Confidence Indicator* dan **Sitasi Dokumen Interaktif** (referensi `[1]`, `[2]` yang bisa diklik langsung ke sumber teks aslinya).
  - **FastAPI + HTML Custom (`api.py` + `index.html`)** — Arsitektur *client-server* terpisah dengan *endpoint* REST API (`/upload` dan `/chat`). Sangat fleksibel untuk modifikasi UI/UX (Tailwind CSS).
- **CLI Tools** — Termasuk skrip mandiri (`ingest.py`, `retrieve.py`, `agent.py`) untuk eksperimen atau *testing* tiap komponen secara terpisah.

> **Catatan:** fitur sitasi interaktif dan indikator kepercayaan saat ini terintegrasi penuh hanya di versi Streamlit.

---

## Tech Stack

| Kategori | Teknologi |
|---|---|
| **Orchestration** | LangGraph |
| **LLM** | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Database** | ChromaDB |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend / UI** | Streamlit, HTML5, Tailwind CSS, Vanilla JS |
| **Document Processing** | `langchain-community`, `pypdf` |

---

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
├── .env               # Kunci rahasia Groq API
└── chroma_db/         # Folder database vektor lokal
```

---

## Panduan Instalasi

### Prerequisites
- Python 3.9+
- Groq API key ([console.groq.com/keys](https://console.groq.com/keys))

### 1. Clone Repositori
```bash
git clone https://github.com/dimssrmdn01/agentic-rag-journal.git
cd agentic-rag-journal
```

### 2. Buat Virtual Environment (opsional, direkomendasikan)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi API Key
Buat file `.env` di folder utama proyek:
```
GROQ_API_KEY=gsk_kunci_rahasia_kamu_di_sini
```

---

## Cara Menjalankan Aplikasi

Pilih salah satu dari tiga mode berikut:

**Opsi A — Streamlit (All-in-One)**
```bash
streamlit run app.py
```
Aplikasi otomatis terbuka di `localhost:8501`. PDF bisa diunggah langsung lewat sidebar.

**Opsi B — FastAPI + HTML Custom (Client-Server)**
```bash
uvicorn api:app --reload
```
Lalu buka `index.html` langsung lewat File Explorer.

**Opsi C — Mode CLI (Testing/Eksperimen)**
```bash
python retrieve.py   # Uji coba kecepatan dan akurasi retrieval
python agent.py      # Uji coba pipeline penuh di terminal
```

---

## Keterbatasan

- Penomoran sitasi `[n]` sangat bergantung pada kepatuhan prompting dari model Llama-3; model yang lebih kecil terkadang melewatkan urutan sitasi.
- Unggahan dokumen baru akan ditambahkan (*append*) ke vector database yang sama — saat ini dioptimalkan untuk satu topik/buku dalam satu waktu.
- Data tersimpan lokal tanpa mekanisme manajemen sesi multi-user.

---

## Roadmap Pengembangan

- [ ] Sinkronisasi fitur confidence indicator & sitasi ke antarmuka FastAPI/HTML
- [ ] Implementasi streaming response (jawaban LLM mengalir kata per kata)
- [ ] Manajemen multi-dokumen (AI dapat membedakan dan merujuk dari buku yang berbeda)
- [ ] Integrasi database relasional untuk riwayat percakapan persisten
- [ ] Fitur ringkasan eksekutif instan saat dokumen pertama kali diunggah

---

## Lisensi & Atribusi

Proyek ini menggunakan Lisensi MIT. Kamu sangat dipersilakan untuk menggunakan, memodifikasi, dan mendistribusikannya secara bebas.

Dibuat oleh **Dimas Arya Ramadhan**.

<div align="center">
  <a href="https://github.com/dimssrmdn01"><img src="https://img.shields.io/badge/GitHub-1c100b?style=for-the-badge&logo=github&logoColor=white"></a>
</div>
