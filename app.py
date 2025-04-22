import streamlit as st
import pickle
import faiss
import openai
import numpy as np

# — Setup —
st.set_page_config(page_title="Smart Immunization Assistant")
st.title("💉 Smart Immunization Assistant")

openai.api_key = st.secrets["OPENAI_API_KEY"]


if not openai.api_key:
    st.error("OpenAI API key not found. Please set it as an environment variable or in Streamlit secrets.")
    st.stop()

@st.cache_resource
def load_resources():
    try:
        idx = faiss.read_index("data/faiss_index.faiss")
        with open("data/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        return idx, chunks
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

index, chunks = load_resources()
if index is None or chunks is None:
    st.error("Failed to load necessary resources. Please check your data files.")
    st.stop()

def retrieve_context(q, k=4):
    try:
        emb_response = openai.embeddings.create(input=q, model="text-embedding-ada-002")
        emb = emb_response.data[0].embedding
        D, I = index.search(np.array([emb], dtype="float32"), k)
        
        # Make sure indices are valid
        valid_indices = [i for i in I[0] if 0 <= i < len(chunks)]
        if not valid_indices:
            return "No relevant information found."
        
        return "\n\n---\n\n".join(chunks[i] for i in valid_indices)
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "An error occurred while retrieving information."

def rag_answer(q):
    try:
        ctx = retrieve_context(q)
        
        prompt = (
          "You are a friendly pediatric vaccine assistant. Use the context below to answer.\n\n"
          f"{ctx}\n\nQ: {q}\nA:"
        )
        
        res = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate vaccine information based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200, 
            temperature=0.7
        )
        
        return res.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer at this time."

# — UI —
q = st.text_input("Ask your vaccine question (e.g. “My baby is 6 weeks old—what vaccines?”):")

if st.button("Get Answer") and q:
    with st.spinner("Looking up your answer…"):
        answer = rag_answer(q)
    st.markdown(answer)