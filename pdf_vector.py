import faiss
import google.generativeai as genai
import numpy as np
import pypdf
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

def pdf_to_vector(pdf_path):
    print(f"Pdf Loading : {pdf_path}")
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        total_pages = len(reader.pages)
        text = []

        for page_num, page in enumerate(reader.pages):
            print(f"Processing page {page_num + 1} of {total_pages}")
            page_text = page.extract_text()
            text.append({"page_number": page_num + 1, "text": page_text})

        text = ''.join([p["text"] for p in text])

    print(f"Total pages : {total_pages}")
    print(f"Total text length : {len(text)} characters")
    print(f"average text length : {len(text) // total_pages} characters per page")

    chunks=[]
    chunk_metadata=[]

    for i in range(0, len(text), 500):
        chunck_text = text[i:i+500]
        chunks.append(chunck_text)

        estimated_page = min((i // (len(text) // total_pages)) + 1, total_pages)
        chunk_metadata.append({
        'start_pos': i,
        "estimated_page": estimated_page,

     })

    print(f"Created {len(chunks)} chunks")
    print(f"getting embeddings from google")
    embeddings = []
    for i,chunck in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}")
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=chunck,
            task_type="retrieval_document"
        )
        embeddings.append(response['embedding'])

    print(f"creating faiss index")
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Use actual embedding dimension
    index.add(embeddings.astype(np.float32))

    print(f"saving index and metadata")
    with open("chunck.pkl", "wb") as f:
        pickle.dump({'chunck' : chunks, 'metadata' : chunk_metadata, 'total_page' : total_pages}, f)
    
    # Save the FAISS index
    faiss.write_index(index, "faiss_index.bin")
    
    print("done")
    return embeddings, chunks

if __name__ == "__main__":
    pdf_file = "WHO.pdf"
    embeddings, chunks = pdf_to_vector(pdf_file)
    print(f"set up Sucessfull")

