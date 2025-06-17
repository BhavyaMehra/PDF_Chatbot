import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from models import Models
import streamlit as st

# --- Modern UI/UX THEME ---
st.set_page_config(page_title="Talk to your PDF!", layout='centered')

# --- Backend logic ---
def get_models():
    models = Models()
    return models.embeddings, models.model

def get_vector_store(_embeddings, docs=None):
    # If docs is provided, create a new FAISS index, else load from session
    if docs is not None:
        return FAISS.from_documents(docs, _embeddings)
    elif 'vector_store' in st.session_state and st.session_state['vector_store'] is not None:
        return st.session_state['vector_store']
    else:
        return None

def get_retrieval_chain(_vector_store, _llm):
    retriever = _vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant that answers questions based on the provided context.'),
        ('human', 'Use the user question {input} to answer the question. Use only the {context} to answer the question.')
    ])
    combine_docs_chain = create_stuff_documents_chain(_llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

chunk_size = 500
chunk_overlap = 50
data_folder = "./data"

# --- Helper functions ---
def clear_all_data():
    # Remove all files in data folder
    if os.path.exists(data_folder):
        for f in os.listdir(data_folder):
            f_path = os.path.join(data_folder, f)
            if os.path.isfile(f_path):
                os.remove(f_path)
    st.session_state.pop('pdf_name', None)
    st.session_state.conversation = []
    st.session_state.pdf_ready = False
    st.session_state['vector_store'] = None
    st.cache_resource.clear()
    return True

def ingest_pdf(file_path, embeddings):
    loader = PyPDFLoader(file_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(loaded_documents)
    # Create a new FAISS index from documents
    vector_store = get_vector_store(embeddings, docs=documents)
    st.session_state['vector_store'] = vector_store

# --- Streamlit UI ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'pdf_ready' not in st.session_state:
    st.session_state.pdf_ready = False
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

with st.sidebar:
    st.header("Upload your PDF")
    if st.session_state.get('pdf_ready', False):
        st.success(f"Current PDF: {st.session_state['pdf_name']}")
        if st.button("Upload New PDF", type="primary"):
            clear_all_data()
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], accept_multiple_files=False)
        if uploaded_file:
            clear_all_data()
            os.makedirs(data_folder, exist_ok=True)
            file_path = os.path.join(data_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            import time
            time.sleep(0.2)  # Ensure file is written to disk (Windows fix)
            if not os.path.exists(file_path):
                st.error(f"File {file_path} was not saved correctly.")
            else:
                embeddings, llm = get_models()
                ingest_pdf(file_path, embeddings)
                st.session_state['pdf_name'] = uploaded_file.name
                st.session_state.pdf_ready = True
                # Set a flag if this is not the first upload
                if st.session_state.get('has_uploaded_before', False):
                    st.session_state['just_uploaded_new_file'] = True
                st.session_state['has_uploaded_before'] = True
            st.rerun()

st.title("PDF Chatbot 🗨️")
st.markdown("""
**Upload a PDF and chat with it!**

Ask questions and get instant answers based only on your document.\
 Have fun exploring your PDFs!
""")

# Show friendly message only after the second (or more) upload
if st.session_state.get('just_uploaded_new_file', False):
    st.info("New PDF uploaded! Start chatting below.")
    st.session_state['just_uploaded_new_file'] = False

if st.session_state.get('pdf_ready', False) and st.session_state['vector_store'] is not None:
    embeddings, llm = get_models()
    vector_store = st.session_state['vector_store']
    retrieval_chain = get_retrieval_chain(vector_store, llm)
    chat_placeholder = st.container()
    with chat_placeholder:
        for chat in st.session_state.conversation:
            st.chat_message("user").write(chat['user'])
            st.chat_message("assistant").write(chat['assistant'])
        user_question = st.chat_input("Ask a question about the PDF:")
        if user_question:
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({'input': user_question})
                answer = response['answer']
            st.session_state.conversation.append({'user': user_question, 'assistant': answer})
            st.rerun()
else:
    st.info("Please upload a PDF to start chatting.")
