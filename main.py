import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from models import Models  # Your existing models file

app = FastAPI()

# Add CORS middleware to allow your React app to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
models = Models()
embeddings = models.embeddings
llm = models.model

# Initialize vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory='./db/chroma_langchain_db'
)

# Create retrieval chain
retriever = vector_store.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant that answers questions based on the provided context.'),
    ('human', 'Use the user question {input} to answer the question. Use only the {context} to answer the question.')
])
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Request models
class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    answer: str
    context: str = ""

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str = ""

# API endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save the uploaded file
    data_folder = "./data"
    os.makedirs(data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, file.filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process the PDF
    try:
        loader = PyPDFLoader(file_path)
        loaded_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents = text_splitter.split_documents(loaded_documents)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        
        return UploadResponse(
            success=True,
            message="PDF uploaded and processed successfully",
            filename=file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = retrieval_chain.invoke({'input': request.input})
        return ChatResponse(
            answer=response['answer'],
            context=response.get('context', '')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/clear")
async def clear_chat():
    # Clear the vector store if needed
    return {"message": "Chat cleared"}

@app.get("/")
async def root():
    return {"message": "PDF Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
