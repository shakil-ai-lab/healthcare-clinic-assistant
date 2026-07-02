# 🏥 Healthcare Clinic Assistant using RAG, LangGraph, and LangChain

## 📌 Project Overview

The Healthcare Clinic Assistant is an AI-powered chatbot developed using LangChain, LangGraph, FAISS, and Google Gemini. The assistant answers user questions by retrieving information only from the clinic's policy documents using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🎯 Objectives

- Build a Retrieval-Augmented Generation (RAG) application.
- Store clinic documents in a FAISS vector database.
- Retrieve relevant information based on user queries.
- Create a retriever tool for the AI agent.
- Build an AI agent capable of using tools.
- Implement a LangGraph workflow.
- Add conversation memory using LangGraph.

---

## 🛠 Technologies Used

- Python
- LangChain
- LangGraph
- HuggingFace DeepSeek Model
- FAISS
- HuggingFace Embeddings
- Sentence Transformers

---

## 📂 Project Structure

```text
Healthcare-Clinic-Assistant/

│── app.py
│── requirements.txt
│── README.md
│── .env
│
├── docs/
│     appointment_policy.txt
│     doctor_schedule.txt
│     lab_test_instructions.txt
│     medicine_refill_policy.txt
│     emergency_policy.txt
│
├── src/
│     document_loader.py
│     vector_store.py
│     retriever.py
│     tools.py
│     agent.py
│     graph.py
│
├── vectorstore/
│     index.faiss
│     index.pkl
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

...