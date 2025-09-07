import streamlit as st
from questio_vector import answer_question

st.set_page_config(page_title="🩺 WHO RAG Assistant", layout="wide")

st.title("🌍 WHO RAG-Powered Medical Research Assistant")
st.markdown(
    """
    Ask me anything about the **World Health Organization (WHO) guidelines & statistics 2025**.  
    This assistant uses **RAG (Retrieval-Augmented Generation)** with **FAISS + Google Gemini**.  
    """
)

# Question box
question = st.text_input("❓ Type your question here")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        st.markdown("⏳ Searching the WHO knowledge base...")
        results, top_chunk, final_answer = answer_question(question)

        if results:
            st.success("✅ Answer generated")

            st.markdown("### 🤖 Answer")
            st.write(final_answer)

            # Show relevant chunks for transparency
            with st.expander("📑 Relevant Context (for transparency)"):
                for c in results['relevant_chunks']:
                    page = c['metadata'].get('estimated_page', '?')
                    st.markdown(
                        f"**Page {page} | Score {c['similarity_score']:.3f}**\n\n"
                        f"```{c['chunk_text'][:400]}...```"
                    )
        else:
            st.error("⚠️ Sorry, I couldn't find anything relevant in WHO data.")
