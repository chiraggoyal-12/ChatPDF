# ChatPDF

## 📄 Conversational RAG with PDF Uploads (Chat History Enabled)

A simple **Conversational Retrieval-Augmented Generation (RAG)** app built using **Streamlit + LangChain + OpenAI + ChromaDB**.

Upload PDFs and ask questions — the app remembers your conversation and provides context-aware answers.

---

## 🚀 Features

- 📂 Upload multiple PDF files  
- 💬 Chat with document content  
- 🧠 Maintains chat history (context-aware answers)  
- 🔍 Retrieval-Augmented Generation (RAG)  
- ⚡ Fast vector search using ChromaDB  
- 🧩 Minimal and clean implementation  

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **LLM:** OpenAI (`gpt-4o-mini`)  
- **Embeddings:** OpenAI Embeddings  
- **Vector DB:** Chroma  
- **Framework:** LangChain  

---

## 📁 Project Structure

    .
    ├── app.py              # Main Streamlit app
    ├── .env                # API key (optional)
    ├── requirements.txt    # Dependencies
    └── README.md

---

## ⚙️ Installation

### 1. Clone the repository

    git clone https://github.com/your-username/conversational-rag-pdf.git
    cd conversational-rag-pdf

### 2. Create virtual environment

    python -m venv venv
    venv\Scripts\activate   # Windows
    # or
    source venv/bin/activate  # Mac/Linux

### 3. Install dependencies

    pip install -r requirements.txt

---

## 🔑 Setup API Key

    Use `.env`

    OPENAI_API_KEY=your_api_key_here

---

## ▶️ Run the App

    streamlit run app.py

---

## 🧠 How It Works

1. Upload PDFs  
2. PDFs are:
   - Loaded using `PyPDFLoader`
   - Split into chunks  
3. Chunks are:
   - Converted into embeddings  
   - Stored in **Chroma vector DB**  
4. User question:
   - Reformulated using chat history  
   - Retrieved relevant context  
   - Passed to LLM for answer generation  
5. Chat history is maintained per session  

---

## 🔄 RAG Pipeline

    User Query
       ↓
    History-Aware Retriever
       ↓
    Relevant Chunks (Chroma)
       ↓
    LLM (GPT-4o-mini)
       ↓
    Final Answer

---

## 📌 Key Components

- `create_history_aware_retriever` → makes questions context-aware  
- `RunnableWithMessageHistory` → maintains chat memory  
- `Chroma` → vector storage  
- `RecursiveCharacterTextSplitter` → chunking  
- `ChatPromptTemplate` → structured prompts  

---

## 🧪 Example Use Cases

- 📚 Study PDFs interactively  
- 📑 Research papers Q&A  
- 📊 Business document analysis  
- 🧾 Legal / financial document querying  

---

## ⚠️ Limitations

- No persistent storage (data resets on refresh)  
- Depends on OpenAI API (cost involved)  
- Basic UI (can be extended)  

---

## 🔮 Future Improvements

- Persistent vector DB  
- User authentication  
- Better UI (chat interface)  
- Streaming responses  
- Multi-user sessions  

---

## 🤝 Contributing

Feel free to fork, improve, and submit a PR 🚀

---

## 📜 License

MIT License

---

## 💡 Author

Built with ❤️ using LangChain & OpenAI
