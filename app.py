
import os
import tempfile
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph

load_dotenv()

class GraphState(TypedDict):
  question: str
  context: str
  answer: str
  route: str
  chat_history: list

def classify_question(state, llm):
  question = state["question"]
  history = state.get("chat_history", [])

  history_text = "\n".join([msg.content for msg in history])

  prompt = f"""
  You are a classifier.

  Conversation so far:
  {history_text}

  If the question can be answered WITHOUT documents → simple  
  If it requires uploaded PDF context → rag  

  Question: {question}

  Answer ONLY one word: simple or rag
  """

  result = llm.invoke(prompt).content.strip().lower()

  # safety fallback
  if "rag" in result:
    return {"route": "rag"}
  else:
    return {"route": "simple"}

def retrieve(state, retriever):

  question = state["question"]
  history = state.get("chat_history", [])

  history_text = " ".join([msg.content for msg in history])

  query = history_text + " " + question

  docs = retriever.get_relevant_documents(query)

  context = "\n".join([doc.page_content for doc in docs])
  return {"context": context}

def generate_answer(state, llm):
  
  question = state["question"]
  history = state.get("chat_history", [])

  history_text = "\n".join([msg.content for msg in history])

  if state["route"] == "simple":
    prompt = f"""
      Conversation:
      {history_text}

      Question:
      {question}
    """
    response = llm.invoke(prompt).content

  else:
    prompt = f"""
    You are a helpful assistant.

    Conversation:
    {history_text}

    Context:
    {state.get("context", "")}

    Question:
    {state["question"]}
"""
    response = llm.invoke(prompt).content

  return {"answer": response}

## Streamlit UI
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

## OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if api_key:
  os.environ["OPENAI_API_KEY"] = api_key

  ## LLM
  llm = ChatOpenAI(model="gpt-4o-mini")

  ## Session ID
  session_id = st.text_input("Session ID", value="default_session")

  ## Chat history store
  if 'store' not in st.session_state:
    st.session_state.store = {}

  ## File uploader
  uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
  )

  if uploaded_files:
    documents = []

    ## Use tempfile instead of temp.pdf
    for uploaded_file in uploaded_files:
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

      loader = PyPDFLoader(tmp_path)
      docs = loader.load()
      documents.extend(docs)

    ## Split documents
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=2000,
      chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    ## OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    ## Vector store
    vectorstore = Chroma.from_documents(
      documents=splits,
      embedding=embeddings,
      persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever()

    ## Contextual question reformulation
    contextualize_q_system_prompt = (
      "Given a chat history and the latest user question, "
      "reformulate it into a standalone question if needed. "
      "Do NOT answer it."
    )

    ## QA prompt
    system_prompt = (
      "You are an assistant for question-answering tasks. "
      "Use the retrieved context to answer the question. "
      "If you don't know, say you don't know. "
      "Use max 10 sentences.\n\n{context}"
    )

    ## Session history function
    def get_session_history(session: str) -> BaseChatMessageHistory:
      if session not in st.session_state.store:
          st.session_state.store[session] = ChatMessageHistory()
      return st.session_state.store[session]

    graph = StateGraph(GraphState)


    graph.add_node("classify", lambda state: classify_question(state, llm))
    graph.add_node("retrieve", lambda state: retrieve(state, retriever))
    graph.add_node("generate_answer", lambda state: generate_answer(state, llm))

    graph.set_entry_point("classify")

    def route_decision(state):
        return state["route"]

    graph.add_conditional_edges(
      "classify",
      route_decision,
      {
        "simple": "generate_answer",
        "rag": "retrieve"
      }
    )

    graph.add_edge("retrieve", "generate_answer")

    app = graph.compile()

    ## User input
    user_input = st.text_input("Your question:")

    if user_input:
      session_history = get_session_history(session_id)

      history_messages = session_history.messages

      response = app.invoke({
        "question": user_input,
        "chat_history": history_messages
      })

      session_history.add_user_message(user_input)
      session_history.add_ai_message(response["answer"])

      st.write("### Route Taken:")
      st.write(response["route"])

      st.write("### Assistant:")
      st.write(response["answer"])

else:
    st.warning("Please enter the OpenAI API Key")
