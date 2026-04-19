# 📄 Conversational PDF AI (RAG + LangGraph)

An intelligent conversational AI system that allows users to upload PDFs and ask questions about their content.
The system dynamically decides whether to answer using a Large Language Model (LLM) directly or retrieve relevant document context using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📥 **PDF Upload & Processing**
  Extracts and splits PDF content into manageable chunks.

* 🧠 **RAG Pipeline**
  Uses embeddings + vector database (Chroma) to retrieve relevant context.

* 🔀 **Dynamic Routing (LangGraph)**
  Automatically decides:

  * **Simple queries → Direct LLM**
  * **Document-based queries → RAG**

* 💬 **Conversational Memory**
  Maintains chat history for context-aware follow-up questions.

* ⚡ **Efficient & Smart Responses**
  Avoids unnecessary retrieval for general queries → faster responses.

---

## 🏗️ Architecture

```
User Input
   ↓
LangGraph (Decision Node)
   ↓
 ┌───────────────┬───────────────┐
 │               │               │
Simple Query     RAG Query
(LLM Direct)     (Retriever + LLM)
 │               │
 └──────→ Final Answer ←─────────┘
                ↑
         Chat History Memory
```

---

## 🧠 How It Works

1. User uploads one or more PDFs
2. Documents are:

   * Loaded
   * Split into chunks
   * Embedded using OpenAI embeddings
3. Stored in a **Chroma vector database**
4. When a user asks a question:

   * LangGraph **classifies the query**
   * Routes it to:

     * Direct LLM (for general questions)
     * RAG pipeline (for document-based queries)
5. Chat history is included for better context understanding

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – UI
* **LangGraph** – Workflow orchestration
* **LangChain** – RAG components
* **OpenAI API** – LLM + embeddings
* **ChromaDB** – Vector database

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🔑 Usage

1. Enter your OpenAI API key
2. Upload one or more PDF files
3. Ask questions like:

   * *“Summarize the document”*
   * *“What is clustering?”*
   * *“Explain more”*

---

## 🎯 Key Design Decisions

### 🔹 Why LangGraph?

Used to create a **controlled workflow** instead of a linear pipeline, enabling:

* Conditional routing
* Modular design
* Better scalability

---

### 🔹 Why Not Always Use RAG?

* RAG is slower and unnecessary for general questions
* Dynamic routing improves:

  * Efficiency ⚡
  * Cost 💰
  * Response quality 🎯

---

### 🔹 How is Memory Handled?

Chat history is:

* Stored per session
* Passed into retrieval and generation steps
  → Enables **context-aware conversations**

---

## 📌 Example

**User:** What is clustering?
**Assistant:** (LLM response)

**User:** Explain more
**Assistant:** (Uses chat history for context)

---

## 🚧 Limitations

* Large PDFs may hit token limits (future improvement: map-reduce summarization)
* Retrieval depends on embedding quality
* Requires OpenAI API key

---

## 🔮 Future Improvements

* 📊 Source highlighting (PDF page references)
* 📉 Confidence scoring
* 🌐 Deployment (Streamlit Cloud / Render)
* 🔗 Flask API backend

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests!

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* OpenAI
* LangChain
* LangGraph
* ChromaDB

---

## 💡 Author

**Your Name**
GitHub: https://github.com/your-username

---
