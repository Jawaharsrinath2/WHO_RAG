# WHO_RAG
A Retrieval-Augmented Generation (RAG) assistant built on WHO World Health Statistics 2025.   
It uses FAISS for vector search, Google Gemini for answer generation, and Streamlit for an interactive UI.
It enables users to ask natural language questions and get concise, evidence-based answers directly from WHO guidelines.

⚡ Powered by:

FAISS → Vector similarity search

Google Gemini → Context-aware answer generation

Streamlit → Interactive user interface

🚀 Features

Ask WHO-related health queries in plain English

Retrieves relevant chunks with page references & similarity scores

Generates concise answers with citations



<img width="1536" height="1024" alt="ChatGPT Image Sep 7, 2025, 01_01_37 PM" src="https://github.com/user-attachments/assets/91362134-f32b-47a2-9292-338dfd98db35" />

📂 Repository Structure
.
├── app.py              # Streamlit UI
├── pdf_vector.py       # Convert WHO PDF → Embeddings + FAISS index
├── question_vector.py  # Query handler + answer generator
├── faiss_index.bin     # Pre-built FAISS index
├── chunck.pkl          # Chunks + metadata
├── requirements.txt    # Dependencies
└── WHO.pdf             # WHO World Health Statistics 2025 (optional)

⚡ Quick Start
🔹 Run Locally
git clone https://huggingface.co/spaces/srinath2003/who-rag-assistant
cd who-rag-assistant
pip install -r requirements.txt
streamlit run app.py

🔹 Environment Variables

Create a .env file with your Google API key:
GOOGLE_API_KEY=your_api_key_here

👨‍💻 Author

Jawaharsrinath M N
AI & Data Enthusiast | Building AI-powered solutions with impact
🔗https://www.linkedin.com/in/jawahar-srinath
🔗https://github.com/Jawaharsrinath2

