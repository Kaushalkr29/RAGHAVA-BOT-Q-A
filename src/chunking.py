from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

class Chunking:

    def __init__(self,documents):
        self.documents=documents
    
    def recursive_overlap(self):
        splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
        chunks=splitter.split_documents(self.documents)
        print("Recursive Overlap Chunking Complete")
        return chunks
    
    def semantic_chunk(self):
        splitter=SemanticChunker(embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        chunks=splitter.split_documents(self.documents)
        print("Semantic Chunking Complete")
        return chunks
    
    def fixed_chunking(self):
        """
        Splits documents into fixed-size chunks without overlap.
        """
        splitter = CharacterTextSplitter(separator="\n",
            chunk_size=1000
        )

        chunks = splitter.split_documents(self.documents)
        print("Fixed Chunking Complete")
        return chunks
    
    def paragraph_chunking(self):
        """
        Splits documents based on blank lines (paragraphs).
        """
        splitter=CharacterTextSplitter(separator="\n\n",
                                       chunk_size=4000,
                                       chunk_overlap=100)
        paragraph_chunks=splitter.split_documents(self.documents)
        print("Paragraph Chunking Complete")
        return paragraph_chunks
    
    def overlap_chunking(self):
        splitter=CharacterTextSplitter(separator="\n",
                                       chunk_size=2000,
                                       chunk_overlap=100)
        chunks=splitter.split_documents(self.documents)
        print("Overlap Chunking Complete")
        return chunks
    