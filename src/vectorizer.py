from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

class Vectorizer:

    def __init__(self,chunks=None):
        self.chunks=chunks
    
    def dense_vector(self):
        load_dotenv()
        GOOGLE_API_KEY=os.getenv("GEMINI_API_KEY")

        embedding_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if Path("Dense_db").exists():
            print("Loading existing Dense Vector DB")
            vector_db=Chroma(persist_directory="Dense_db",
                             embedding_function=embedding_model)
        else:
            print("Creating new Dense Vector DB")
            vector_db=Chroma.from_documents(persist_directory="Dense_db",
                                            embedding=embedding_model,documents=self.chunks)
            print("New Dense vector_db created")
            
        return vector_db
    
    def sparse_vector(self, k=4):

        # path=Path("Sparse_db")

        # if path.exists():
        #     print("Loading existing Sparse (TF-IDF) DB")
        #     tfidf_retriever = TFIDFRetriever.load_local(
        #         folder_path=str(path),
        #         allow_dangerous_deserialization=True
        #     )
        # else:
        #     print("Creating new Sparse (TF-IDF) DB")
        #     tfidf_retriever = TFIDFRetriever.from_documents(self.chunks)
        #     tfidf_retriever.k = k

        #     path.mkdir(parents=True, exist_ok=True)
        #     tfidf_retriever.save_local(folder_path=str(path))
        #     print("New sparse TF-IDF DB created")

        # return tfidf_retriever
    
        path = "Sparse_db"

        if Path(path).exists():
            print("Loading Sparse (TF-IDF) DB")
            return TFIDFRetriever.load_local(path, allow_dangerous_deserialization=True)
        else:
            print("Creating Sparse (TF-IDF) DB")
            retriever = TFIDFRetriever.from_documents(self.chunks)
            retriever.k = k
            retriever.save_local(path)
        return retriever
    
    def graph_vector(self, threshold=0.5):

        GRAPH_PATH = "Graph_db/graph.pkl"
        VECTOR_DB_PATH = "Graph_db/chroma"

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        os.makedirs("Graph_db", exist_ok=True)

        # Load existing databases
        if os.path.exists(GRAPH_PATH):

            vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )

            with open(GRAPH_PATH, "rb") as file:
                graph = pickle.load(file)

            print("Graph vector store loaded")

            return vector_store, graph

        # Create vector database
        vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        # Create graph
        graph = nx.Graph()

        texts = [chunk.page_content for chunk in self.chunks]
        vectors = embeddings.embed_documents(texts)

        # Add nodes
        for i, chunk in enumerate(self.chunks):
            graph.add_node(
                i,
                text=chunk.page_content,
                metadata=chunk.metadata
            )

        # Add similarity edges
        for i in range(len(self.chunks)):
            for j in range(i + 1, len(self.chunks)):

                score = cosine_similarity(
                    [vectors[i]],
                    [vectors[j]]
                )[0][0]

                if score >= threshold:
                    graph.add_edge(
                        i,
                        j,
                        weight=float(score)
                    )

        # Save graph locally
        with open(GRAPH_PATH, "wb") as file:
            pickle.dump(graph, file)

        print("Graph vector store created")

        return vector_store, graph