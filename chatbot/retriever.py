from embedder import Embedder, load_faiss_index, load_metadata
import numpy as np

class Retriever:
    def __init__(self, index_path="vector_store/doc_index"):
        self.index = load_faiss_index(index_path)
        self.metadata = load_metadata(index_path)
        self.embedder = Embedder()

    def retrieve(self, query, top_k=3):
        query_vec = self.embedder.embed_texts([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        results = [self.metadata[i] for i in I[0]]
        return results

# Example usage
if __name__ == "__main__":
    retriever = Retriever()
    query = "What is the main topic of the document?"
    results = retriever.retrieve(query)

    print("Top matching chunks:\n")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk[:200]}...\n")
