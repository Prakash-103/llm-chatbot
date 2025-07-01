from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_db(documents):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = FAISS.from_documents(documents, embeddings)
    return db
