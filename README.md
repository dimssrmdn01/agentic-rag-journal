<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=36&duration=2500&pause=1000&color=7C9CFF&center=true&vCenter=true&width=600&lines=🤖+Agentic+RAG+Journal;Tanya-Jawab+Dokumen+Cerdas;Anti-Halusinasi+dengan+LangGraph" alt="Typing SVG" />
  </a>
  
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

Bukan sekadar sistem *retrieve-then-generate* biasa. **Agentic RAG Journal** menyisipkan tahap *relevance grading* (penyaringan relevansi) secara otomatis agar jawaban AI selalu akurat, tepat sasaran, dan **tidak berhalusinasi** di luar konteks dokumen.

Tersedia dalam dua arsitektur pilihan: **Dashboard Streamlit** (All-in-One) dan **FastAPI + HTML** (Decoupled Architecture).

##  Cara Kerja Pipeline

Alih-alih langsung menjawab, agen AI akan melewati tahap penyaringan ketat:

📄 **PDF** ➔ 🧩 **Ingest & Chunking** ➔ 🗄️ **Vector Store (ChromaDB)**
<br>
💬 **Pertanyaan** ➔ 🔍 **Retrieve** ➔ ⚖️ **Relevance Grade** ➔ 📝 **Generate** ➔ ✨ **Jawaban**

1. **🔍 Retrieve:** Mengambil `k` potongan dokumen yang paling mirip secara semantik dengan pertanyaan pengguna.
2. **⚖️ Grade:** Setiap potongan dokumen dievaluasi ulang oleh LLM. Jika ada teks yang tidak relevan dengan pertanyaan, teks tersebut langsung **dibuang** sebelum sampai ke tahap pembuatan jawaban.
3. **📝 Generate:** Menyusun jawaban akhir murni dari potongan dokumen yang lolos seleksi. Jika tidak ada yang lolos, sistem akan secara jujur menyatakan bahwa informasi tidak ditemukan.

##  Fitur Utama

- **Ingesti Dokumen Otomatis:** Ekstraksi dan pemecahan (*chunking*) teks PDF langsung ke *vector database* ChromaDB.
- **Anti-Halusinasi:** Pipeline *Retrieve ➔ Grade ➔ Generate* meminimalisir jawaban ngawur.
- **Dual Interface:**
  - 🖥️ **Streamlit (`app.py`):** Dashboard UI siap pakai lengkap dengan *Confidence Indicator* dan **Sitasi Dokumen Interaktif** (referensi `[1]`, `[2]` yang bisa diklik langsung ke sumber teks aslinya).
  - 🌐 **FastAPI + HTML Custom (`api.py` + `index.html`):** Arsitektur *client-server* terpisah dengan *endpoint* REST API (`/upload` dan `/chat`). Sangat fleksibel untuk modifikasi UI/UX (Tailwind CSS).
- **CLI Tools:** Termasuk skrip mandiri (`ingest.py`, `retrieve.py`, `agent.py`) untuk eksperimen atau *testing* tiap komponen secara terpisah.

*(Catatan: Fitur sitasi interaktif dan indikator kepercayaan saat ini terintegrasi penuh di versi Streamlit).*

##  Tech Stack

| Kategori | Teknologi |
| :--- | :--- |
| **Orchestration** | LangGraph |
| **LLM** | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Database** | ChromaDB |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend / UI** | Streamlit, HTML5, Tailwind CSS, Vanilla JS |
| **Document Processing**| `langchain-community`, `pypdf` |

##  Struktur Proyek

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

##  Panduan Instalasi

**1. Clone Repositori**

```bash
git clone https://github.com/dimssrmdn01/agentic-rag-journal.git
cd agentic-rag-journal
```

**2. Buat Virtual Environment** (opsional tapi direkomendasikan)

```bash
python -m venv venv
source venv/bin/activate   # Pengguna Windows: venv\Scripts\activate
```

**3. Install Dependensi**

```bash
pip install -r requirements.txt
```

**4. Konfigurasi API Key**

Buat file bernama `.env` di dalam folder utama proyek dan masukkan API Key dari Groq Console:

```
GROQ_API_KEY=gsk_kunci_rahasia_kamu_di_sini
```

##  Cara Menjalankan Aplikasi

Kamu dapat memilih salah satu dari dua antarmuka yang disediakan:

**Opsi A - Menggunakan Streamlit (All-in-One)**

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di `localhost:8501`. Kamu dapat mengunggah PDF langsung melalui sidebar.

**Opsi B - Menggunakan FastAPI & HTML Custom (Client-Server)**

Jalankan server backend terlebih dahulu:

```bash
uvicorn api:app --reload
```

Lalu buka file `index.html` secara langsung (klik dua kali) melalui File Explorer kamu.

**Opsi C - Mode CLI (Untuk Testing/Eksperimen)**

```bash
python retrieve.py   # Uji coba kecepatan dan akurasi retrieval
python agent.py      # Uji coba pipeline penuh di terminal
```

##  Keterbatasan

- Penomoran sitasi `[n]` sangat bergantung pada kepatuhan prompting dari model Llama-3. Terkadang model yang lebih kecil melewatkan urutan sitasi.
- Unggahan dokumen baru akan ditambahkan (*append*) ke dalam vector database yang sama. Saat ini dioptimalkan untuk membedah satu topik/buku pada satu waktu.
- Data tersimpan secara lokal tanpa mekanisme manajemen sesi (multi-user).

## Roadmap Pengembangan

- [ ] Sinkronisasi fitur confidence indicator & sitasi ke antarmuka FastAPI/HTML.
- [ ] Implementasi Streaming Response (jawaban LLM mengalir kata per kata).
- [ ] Manajemen multi-dokumen (AI dapat membedakan dan merujuk dari buku yang berbeda).
- [ ] Integrasi database relasional untuk riwayat percakapan persisten.
- [ ] Fitur ringkasan eksekutif instan saat dokumen pertama kali diunggah.

##  Lisensi & Atribusi

Proyek ini menggunakan Lisensi MIT. Kamu sangat dipersilakan untuk menggunakan, memodifikasi, dan mendistribusikannya secara bebas.

Dibuat dengan ☕ dan antusiasme terhadap Data Science oleh **Dimas Arya Ramadhan**.
