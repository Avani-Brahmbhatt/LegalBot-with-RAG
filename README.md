# 🧑‍⚖️ LegalBot with RAG

**LegalBot** is an intelligent legal assistant powered by Retrieval-Augmented Generation (RAG) and Meta's `llama3-70b-8192` model. It is designed to provide accurate, context-aware responses to questions based on Indian legal documents such as statutes, acts, and judgments.

---

## 🔍 Overview

LegalBot combines the power of semantic search with large language models to enable conversational access to Indian law. By embedding and indexing legal documents, it allows users to query complex legal topics and receive informed answers grounded in real legal text.

---

## 🧠 Key Features

- 🔍 Semantic search across a corpus of Indian legal texts using FAISS
- 🦙 Contextual understanding with LLaMA model
- 📚 Flexible and scalable knowledge base for legal documents
- 💬 Conversational interface with memory-aware responses
- 🧱 Modular design for easy extension and integration

---

## 🛠️ Tech Stack

- **LLM**: `llama3-70b-8192` via Groq API
- **Vector Store**: FAISS for embedding-based document retrieval
- **Framework**: LangChain for memory and retrieval chaining
- **Language**: Python
- **Data**: Indian legal documents (text format)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Avani-Brahmbhatt/LegalBot-with-RAG.git
cd LegalBot-with-RAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Legal Documents

Place your `.txt` or `.pdf` files containing Indian legal text into the `data/` directory.

### 4. Create Vector Store

```bash
python vector_db.py
```

### 5. Connect LLM to Vector DB

```bash
python llm_connection.py
```
### 6. Launch LegalBot

```bash
streamlit run app.py
```
---

## 💬 Sample Query

> **User:** What are the fundamental rights and duties of Indian citizens?

---
![Screenshot 2025-05-01 122411](https://github.com/user-attachments/assets/80a11211-3609-4083-bdc8-cb1205ffec03)

![image](https://github.com/user-attachments/assets/68772002-5448-477d-952c-31ebdef25ddd)
