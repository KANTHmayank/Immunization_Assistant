# Smart Immunization Assistant

A conversational AI assistant that helps parents and caregivers understand and track childhood vaccinations.  
Built with Streamlit and OpenAI’s GPT‑3.5 Turbo using a Retrieval‑Augmented Generation (RAG) approach over an official immunization PDF.

---

## 🔍 Features

- **Free‑text Q&A**  
  Ask any vaccine‑related question (e.g. “What is the MMR vaccine for?”) and get a concise, accurate answer sourced from the official guideline PDF.

- **Age‑based schedule lookup**  
  Questions like “My baby is 6 weeks old—what vaccines are due?” trigger retrieval of the relevant schedule passages and a synthesized recommendation.

- **PDF knowledge base**  
  All text is automatically chunked, embedded, and indexed—no manual table parsing required.

- **Easy deployment**  
  Live demo hosted on Streamlit Cloud; updates auto‑redeployed via GitHub.

---

## 📁 Repository Structure

Immunization_Assistant/ ├── app.py # Streamlit UI and RAG query logic ├── build_index.py # One‑time script to chunk PDF and build FAISS index ├── pdf_processing.py # PDF loading & text chunking utility ├── requirements.txt # Python dependencies └── data/ ├── faiss_index.faiss # FAISS index file (generated) ├── chunks.pkl # Pickled list of text chunks (generated)


## ⚙️ Installation & Local Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/assgn.git
   cd assgn



2. Create a virtual environment (optional)
   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Set your OpenAI API key
   export OPENAI_API_KEY="your-openai-key"

5. Build the FAISS Index (One‑Time)
   python build_index.py

6. Run Locally
   streamlit run app.py



APP SCREENSHOTS

![Screenshot (147)](https://github.com/user-attachments/assets/72b5318c-0fc5-4a3d-9510-d38c08289ec5)
![Screenshot (148)](https://github.com/user-attachments/assets/38aa88b8-e7f5-490d-a5b9-dfc23a24b150)
![Screenshot (149)](https://github.com/user-attachments/assets/f8ab11fc-b645-4752-8359-4f686892b3eb)



Built by Mayank Kanth

