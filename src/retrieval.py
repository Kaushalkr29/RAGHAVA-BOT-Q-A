class retrieve:
    def __init__(self,question,vector_db):
        self.question=question
        self.vector_db=vector_db

    def top_k(self,TOP_K=3):
        retriever=self.vector_db.as_retriever(search_kwargs={"k":TOP_K})
        retrieved_doc=retriever.invoke(self.question)
        context="\n".join(doc.page_content for doc in retrieved_doc)
        return context, self.question
    
    def mmr_retrieval(self,question,vector_db,top_k):
        retriver=vector_db.max_marginal_relevance_search(query=question,
                                                         k=top_k,
                                                         fetch_k=20,
                                                         lambda_mult=0.2)
        retrieve_doc=retriver.invoke(question)
        context="\n".join(doc.page_content for doc in retrieve_doc)
        return context, question