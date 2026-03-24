# models.py
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
os.environ["OPENAI_API_BASE"] = st.secrets["openai_api_base"]

class Models:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

        self.model = ChatOpenAI(model_name='nvidia/nemotron-nano-12b-v2-vl:free', 
                                temperature=0)
