def get_retriever(vectordb, k=4, use_mmr=True):
    if use_mmr:
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "lambda_mult": 0.5}
        )
    return vectordb.as_retriever(search_kwargs={"k": k})