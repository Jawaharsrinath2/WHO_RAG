import faiss
import google.generativeai as genai
import numpy as np
import pickle
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Google AI API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file!")
    print("Please create a .env file with:")
    print("GOOGLE_API_KEY=your_api_key_here")
    exit(1)
genai.configure(api_key=api_key)


def get_question_embedding(question):
    """Convert a question to embedding using Google's text-embedding-004 model"""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=question,
            task_type="retrieval_query",
        )
        return np.array(response['embedding']).reshape(1, -1).astype(np.float32)
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None


def ask_question(question, top_k=3):
    """
    Retrieve top_k most relevant chunks for the question.

    Returns a dict with question and ranked relevant_chunks [{chunk_text, metadata, distance, similarity_score}].
    """
    if not os.path.exists("faiss_index.bin") or not os.path.exists("chunck.pkl"):
        print("Vector Database not found. Please run pdf_vector.py first to create the database.")
        return None

    try:
        # Load the FAISS index
        index = faiss.read_index("faiss_index.bin")

        # Load the chunks and metadata
        with open("chunck.pkl", "rb") as f:
            data = pickle.load(f)
            chunks = data['chunck']
            metadata = data['metadata']
            total_pages = data['total_page']

        # Convert question to embedding
        question_embedding = get_question_embedding(question)
        if question_embedding is None:
            return None

        # Search for similar chunks
        distances, indices = index.search(question_embedding, top_k)

        # Prepare results
        results = {
            'question': question,
            'relevant_chunks': [],
            'total_chunks_found': len(indices[0]),
        }

        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if 0 <= idx < len(chunks):
                chunk_data = {
                    'rank': i + 1,
                    'similarity_score': float(1 / (1 + distance)),
                    'distance': float(distance),
                    'chunk_text': chunks[idx],
                    'metadata': metadata[idx] if idx < len(metadata) else {},
                }
                results['relevant_chunks'].append(chunk_data)

        return results

    except Exception as e:
        print(f"Error searching vector database: {e}")
        return None


def generate_final_answer(question: str, context: str) -> str:
    """Generate a concise final answer using Gemini given the question and a context block."""
    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        prompt = (
            "You are a helpful assistant. Answer strictly using ONLY the context. "
            "Be concise (2-5 sentences). If the answer isn't present, say: 'I can't find this in the document.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"Error generating answer: {e}"


def build_context_from_results(results, include_k: int = 3) -> str:
    """Concatenate the top-k chunks with page hints to form the context block."""
    lines = []
    for chunk in results['relevant_chunks'][:include_k]:
        page = chunk['metadata'].get('estimated_page', 'Unknown') if chunk['metadata'] else 'Unknown'
        lines.append(f"[Chunk {chunk['rank']} | Score {chunk['similarity_score']:.3f} | Page {page}]\n{chunk['chunk_text']}")
    return "\n\n".join(lines)


def answer_question(question: str, top_k: int = 3, similarity_threshold: float = 0.55):
    """Return results, the most relevant chunk, and a concise final answer built from top-k chunks."""
    results = ask_question(question, top_k=top_k)
    if not results or not results['relevant_chunks']:
        return None, None, "No relevant information found."

    # If best similarity is very low, bail early
    best_sim = results['relevant_chunks'][0]['similarity_score']
    if best_sim < similarity_threshold:
        return results, None, "I can't find this in the document."

    # Build multi-chunk context
    context_block = build_context_from_results(results, include_k=top_k)

    final_answer = generate_final_answer(question, context_block)
    top_chunk_text = results['relevant_chunks'][0]['chunk_text']
    return results, top_chunk_text, final_answer


def interactive_qa():
    """Interactive Q&A that prints like the provided example UI."""
    while True:
        question = input("?  Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue

        print("\n🔎  Searching and generating answer...")
        results, top_chunk, final_answer = answer_question(question, top_k=3)
        if results is None:
            print("Vector search failed or database missing.\n")
            continue

        print(f"🔍  Found {len(results['relevant_chunks'])} relevant chunks:")
        for c in results['relevant_chunks']:
            page = c['metadata'].get('estimated_page', 'Unknown') if c['metadata'] else 'Unknown'
            print(f"  Chunk {c['rank']}: Score {c['similarity_score']:.3f} (≈Page {page})")

        print(f"\n🤖  Answer: {final_answer}\n")


if __name__ == "__main__":
    # Start interactive session styled like the example
    interactive_qa()
        