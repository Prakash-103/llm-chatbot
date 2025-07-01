from langchain.chains import RetrievalQA

def create_rag_chain(llm, vector_db):
    retriever = vector_db.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain
