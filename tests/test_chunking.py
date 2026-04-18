from rag.chunking import chunk_documents

def test_chunking_non_empty(sample_docs):
    chunks = chunk_documents(sample_docs)
    assert len(chunks) > 0