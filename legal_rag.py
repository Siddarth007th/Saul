from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

# load legal text
with open("laws.txt", "r") as f:
    laws = f.readlines()

# convert laws to embeddings
embeddings = model.encode(laws)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search_law(query):

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, 2)

    results = [laws[i] for i in indices[0]]

    return "\n".join(results)
