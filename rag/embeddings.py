from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vectorstore(chunks, persist_directory):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb