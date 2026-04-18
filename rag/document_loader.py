from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(data_path: str):
    documents = []
    for pdf in Path(data_path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents