import pickle, faiss, openai, numpy as np
from pdf_processing import load_and_chunk
import os
import streamlit as st


openai.api_key = st.secrets["OPENAI_API_KEY"]


def build_faiss(pdf_path="Immunization (1).pdf"):
    chunks = load_and_chunk(pdf_path)
    embs = []
    for c in chunks:
        resp = openai.embeddings.create(input=c, model="text-embedding-ada-002")
        embs.append(resp.data[0].embedding)
    arr = np.array(embs, dtype="float32")
    idx = faiss.IndexFlatL2(arr.shape[1])
    idx.add(arr)
    pickle.dump(chunks, open("data/chunks.pkl", "wb"))
    faiss.write_index(idx, "data/faiss_index.faiss")
    print("✅ Built FAISS index with", len(chunks), "chunks")

if __name__ == "__main__":
    build_faiss()
