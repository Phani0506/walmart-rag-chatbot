# vector_store_utils.py

import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from data_loader import load_and_prepare_data

# Define the path for the persistent vector store
CHROMA_DB_PATH = "./chroma_db"

def create_or_load_vector_store():
    """
    Creates a new ChromaDB vector store if it doesn't exist,
    otherwise loads the existing one from disk.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if os.path.exists(CHROMA_DB_PATH):
        # If the database already exists, just load it
        print(f"Loading existing vector store from {CHROMA_DB_PATH}...")
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print("✓ Vector store loaded.")
    else:
        # If it doesn't exist, create it
        print("Existing vector store not found. Creating a new one...")
        
        # 1. Load and prepare the data
        documents = load_and_prepare_data()
        
        # 2. Create the vector store with the documents
        print("Creating embeddings and vector store... (This might take a few minutes)")
        vector_store = Chroma.from_texts(
            texts=documents, 
            embedding=embeddings, 
            persist_directory=CHROMA_DB_PATH
        )
        print("✓ New vector store created and saved.")
        
    return vector_store