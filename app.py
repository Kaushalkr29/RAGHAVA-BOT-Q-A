from flask import Flask, render_template, request, jsonify

from generation.gemini_generation import Generation
from src.fileloader import FileLoader
from src.chunking import Chunking
from src.vectorizer import Vectorizer
from src.retrieval import retrieve

# -----------------------------------------
# Build Knowledge Base (Runs Once)
# -----------------------------------------

loader = FileLoader()
documents = loader.doc()

chunker = Chunking(documents)

# Choose ONE chunking method
chunks = chunker.semantic_chunk()

vectorizer = Vectorizer(chunks)
vector_db = vectorizer.dense_vector()

retriever = retrieve()

# -----------------------------------------

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    query = data.get("query", "").strip()

    if not query:
        return jsonify({"response": "Please enter a question."})

    # Retrieve relevant chunks
    retrieved, question = retriever.top_k(query, vector_db)

    # Generate answer
    generator = Generation(question, retrieved)

    # Call the method
    response = generator.ask_question()

    return jsonify({
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True,port=8080)