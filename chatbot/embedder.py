from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

def save_faiss_index(index, filepath):
    faiss.write_index(index, f"{filepath}.index")

def load_faiss_index(filepath):
    return faiss.read_index(f"{filepath}.index")

def save_metadata(metadata, filepath):
    with open(f"{filepath}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_metadata(filepath):
    with open(f"{filepath}_meta.pkl", "rb") as f:
        return pickle.load(f)

def create_and_save_index(text_chunks, save_path="vector_store/doc_index"):
    embedder = Embedder()
    embeddings = embedder.embed_texts(text_chunks)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    save_faiss_index(index, save_path)
    save_metadata(text_chunks, save_path)

    print(f"Saved FAISS index and metadata to `{save_path}`")

if __name__ == "__main__":
    from document_loader import load_and_split

    file_path = "os1a-slides.pdf"  # or example.pdf
    chunks = load_and_split(file_path)
    create_and_save_index(chunks)
