import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
import tempfile
import os

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["OPENAI_API_BASE"] = st.secrets["openai_api_base"]


st.title("Talk to your PDF! Ask away!")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def process_pdf(file):
    """
    Processes the uploaded PDF file and saves it temporarily.
    Creates a temporary file to store the uploaded PDF and returns the path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name
    return tmp_path

def load_and_embed_pdf(path):
    """
    Loads the PDF content and embeds it into a vector store.
    - Uses PyPDFLoader to load the PDF.
    - Splits the content into smaller chunks using RecursiveCharacterTextSplitter.
    - Embeds the chunks using HuggingFaceEmbeddings.
    - Creates a FAISS vector store from the embedded chunks and returns it.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = FAISS.from_documents(docs, embeddings)
    return db

def create_qa_chain(db):
    """
    Creates a QA chain for querying the vector store.
    - Converts the vector store into a retriever.
    - Initializes the ChatOpenAI model with specified parameters.
    - Creates a RetrievalQA chain using the retriever and the ChatOpenAI model.
    - Returns the QA chain.
    """
    retriever = db.as_retriever()
    llm = ChatOpenAI(
        model_name="deepseek/deepseek-r1:free",
        temperature=0,
        streaming=True  # optional, remove if not using streaming
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        memory=st.session_state.memory
    )
    return qa

# Initialize session state variables

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

if 'memory' not in st.session_state:
    st.session_state.memory = memory


# Function to display the chat history in the Streamlit app.
def display_chat_history():
    """
    Dynamically displays the chat history in the Streamlit app.
    - Appends only new messages to the chat interface.
    - Handles long conversations efficiently by limiting the number of displayed messages.
    """
    if 'chat_placeholder' not in st.session_state:
        st.session_state.chat_placeholder = st.empty()  # Create a persistent placeholder

    st.session_state.chat_placeholder.empty()  # Clear the placeholder before displaying new messages

    # Limit the number of messages displayed to avoid performance issues with long conversations
    max_messages = 50  # Display only the last 50 messages
    messages_to_display = st.session_state.memory.chat_memory.messages[-max_messages:]

    # Use the placeholder to display chat messages
    with st.session_state.chat_placeholder.container():
        for message in messages_to_display:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(f"**User:** {message.content}")
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(f"**Assistant:** {message.content}")

# User uploads a PDF file

uploaded_file = st.file_uploader("Upload a PDF file!", type="pdf")

# Error handling for PDF processing
if uploaded_file and not st.session_state.pdf_processed:
    try:
        # Process PDF file
        tmp_path = process_pdf(uploaded_file)

        # Create vector store and QA chain for PDF
        db = load_and_embed_pdf(tmp_path)
        qa = create_qa_chain(db)

        # Store in session_state to persist across reruns
        st.session_state.qa_chain = qa
        st.session_state.pdf_processed = True
        st.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {str(e)}")

# Show chat input and interaction only for PDFs
if st.session_state.pdf_processed and st.session_state.qa_chain is not None:
    # Display the entire chat history dynamically
    display_chat_history()

    # Allow user to ask a new question
    question = st.chat_input("Ask a question about the PDF:")

    if question:
        try:
            # Add user message to memory
            st.session_state.memory.chat_memory.add_user_message(question)
            with st.chat_message("user"):
                st.markdown(f"**User:** {question}")

            with st.spinner("Thinking..."):
                # Include chat history in the context for the QA chain
                context = '\n'.join([msg.content for msg in st.session_state.memory.chat_memory.messages])
                combined_query = f'{context}\n\n{question}'  # Prepend context to the question
                answer = st.session_state.qa_chain.invoke({'query': combined_query})
                model_answer = answer['result']

            # Add assistant message to memory
            st.session_state.memory.chat_memory.add_ai_message(model_answer)
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {model_answer}")
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
else:
    st.write("Please upload a PDF file to start.")


