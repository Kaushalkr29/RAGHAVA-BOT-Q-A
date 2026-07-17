from src.fileloader import FileLoader
from src.chunking import Chunking
from src.vectorizer import Vectorizer

d=FileLoader()
documents=d.doc()
c=Chunking(documents)

c.fixed_chunking()
c.overlap_chunking()
c.paragraph_chunking()
c.recursive_overlap()
chunks=c.semantic_chunk()

v=Vectorizer(chunks)
v.dense_vector()
v.sparse_vector()
v.graph_vector()

from generation.ollama_generation import generate_response

gr=generate_response("What is ML?")
print(gr)