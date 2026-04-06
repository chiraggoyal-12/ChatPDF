## RAG Q&A Conversation with PDF including chat history

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

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
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

        ## Contextual question reformulation
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "reformulate it into a standalone question if needed. "
            "Do NOT answer it."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        ## QA prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the retrieved context to answer the question. "
            "If you don't know, say you don't know. "
            "Use max 10 sentences.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        ## Session history function
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        ## Conversational chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        ## User input
        user_input = st.text_input("Your question:")

        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("### Assistant:")
            st.write(response["answer"])

            st.write("### Chat History:")
            st.write(session_history.messages)

else:
    st.warning("Please enter the OpenAI API Key")