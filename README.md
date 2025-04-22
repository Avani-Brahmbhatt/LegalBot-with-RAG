# 🧑‍⚖️ LegalBot with RAG

**LegalBot** is an intelligent legal assistant powered by Retrieval-Augmented Generation (RAG) and Meta's `Llama-3.1-8B-Instruct` model. It is designed to provide accurate, context-aware responses to questions based on Indian legal documents such as statutes, acts, and judgments.

---

## 🔍 Overview

LegalBot combines the power of semantic search with large language models to enable conversational access to Indian law. By embedding and indexing legal documents, it allows users to query complex legal topics and receive informed answers grounded in real legal text.

---

## 🧠 Key Features

- 🔍 Semantic search across a corpus of Indian legal texts using FAISS
- 🦙 Contextual understanding with LLaMA 3.1-8B Instruct model
- 📚 Flexible and scalable knowledge base for legal documents
- 💬 Conversational interface with memory-aware responses
- 🧱 Modular design for easy extension and integration

---

## 🛠️ Tech Stack

- **LLM**: `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace
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
> 
> **LegalBot:**"Fundamental rights are six rights guaranteed by the Constitution of India to every Indian citizen. They are:
           - Right to Equality (Article 14-18)
           - Right to Freedom (Article 19-22)
           - Right against Exploitation (Article 23-24)
           - Right to Freedom of Religion (Article 25-28)
           - Cultural and Educational Rights (Article 29-30)
           - Right to Constitutional Remedies (Article 32)

 Fundamental duties, on the other hand, are the duties of every Indian citizen towards the nation. They are listed in Article 51A of the Constitution and include duties such as abiding by the Constitution, respecting the National Flag and Anthem, upholding the sovereignty and integrity of India, defending the country, and promoting harmony and the spirit of common brotherhood."

---
