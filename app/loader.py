from langchain.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(pdf_folder):
    all_docs = []
    for pdf_path in Path(pdf_folder).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs
