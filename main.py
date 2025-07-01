from app.loader import load_documents
from app.embedder import create_vector_db
from app.model import load_llm
from app.rag_chain import create_rag_chain
from app.interface import launch_gradio_interface

def main():
    docs = load_documents("your_pdfs")  # Put PDFs in this folder
    vector_db = create_vector_db(docs)
    llm = load_llm()
    rag_chain = create_rag_chain(llm, vector_db)
    launch_gradio_interface(rag_chain)

if __name__ == "__main__":
    main()
