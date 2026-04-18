from rag.document_loader import load_documents
from rag.chunking import chunk_documents
from rag.embeddings import create_vectorstore
from rag.retriever import get_retriever
from rag.generator import load_llm

def run_rag(query, data_path, persist_dir, use_mmr=True):
    docs = load_documents(data_path)
    chunks = chunk_documents(docs)
    vectordb = create_vectorstore(chunks, persist_dir)
    retriever = get_retriever(vectordb, use_mmr=use_mmr)
    llm = load_llm()

    context_docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in context_docs])
    
    prompt = f"""
    Answer ONLY using the context below.
    If the answer is not present, say \"I don't know\".
    Context:
    {context}
    
    Question:
    {query}
    """

    return llm.invoke(prompt)

