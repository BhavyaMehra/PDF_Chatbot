import streamlit as st
import os
from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from models import Models

# Initialize models
models = Models()
embeddings = models.embeddings
llm = models.model

# Vector store setup
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory='./db/chroma_langchain_db'
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant that answers questions based on the provided context.'),
    ('human', 'Use the user question {input} to answer the question. Use only the {context} to answer the question.')
])

retriever = vector_store.as_retriever()
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Ingest parameters
chunk_size = 500
chunk_overlap = 50
data_folder = "./data"

def ingest_file(file_path):
    if not file_path.lower().endswith('.pdf'):
        st.warning(f"Skipping non-PDF file: {file_path}")
        return

    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(loaded_documents)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

def ingest_and_rename(file_path):
    # Skip if already ingested (filename starts with _)
    filename = os.path.basename(file_path)
    if filename.startswith('_'):
        return

    ingest_file(file_path)

    dir_path = os.path.dirname(file_path)
    new_file_name = '_' + filename
    new_file_path = os.path.join(dir_path, new_file_name)

    if os.path.exists(new_file_path):
        os.remove(new_file_path)  # delete existing renamed file first
    os.rename(file_path, new_file_path)
    st.info(f"Renamed file {filename} to {new_file_name}")

# Streamlit UI & state
st.set_page_config(page_title="Talk to your PDF!", page_icon=":robot_face:", layout='wide')
st.title("Talk to your PDF! Ask away!")

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# File upload & ingestion (left panel)
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'], accept_multiple_files=False)
    if uploaded_file:
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, uploaded_file.name)
        if not os.path.exists(file_path) and not os.path.exists(os.path.join(data_folder, '_' + uploaded_file.name)):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            ingest_and_rename(file_path)
            st.success(f"Uploaded and ingested {uploaded_file.name}")
        else:
            st.warning(f"File {uploaded_file.name} already uploaded and ingested.")

# Main chat interface (right panel)
for chat in st.session_state.conversation:
    st.chat_message("user").write(chat['user'])
    st.chat_message("assistant").write(chat['assistant'])

user_question = st.chat_input("Ask a question about the PDF:")
if user_question:
    response = retrieval_chain.invoke({'input': user_question})
    answer = response['answer']
    st.session_state.conversation.append({'user': user_question, 'assistant': answer})
    st.chat_message("user").write(user_question)
    st.chat_message("assistant").write(answer)
