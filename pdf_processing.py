import pdfplumber
from tiktoken import get_encoding

def load_and_chunk(pdf_path, chunk_size=500):
    enc = get_encoding("cl100k_base")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text += (p.extract_text() or "") + "\n"
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(enc.decode(tokens[i : i + chunk_size]))
    return chunks
