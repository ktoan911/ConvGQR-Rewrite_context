import json

import chromadb
from chromadb.utils import embedding_functions
from rewrite_local import Rewrite

rewrite = Rewrite()

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)
client = chromadb.PersistentClient(path="./chroma_data")

collection = client.get_or_create_collection(
    "embeddings_rag",
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"},
)
# with open(
#     r"/home/toannk/Desktop/Code/ConvGQR-Rewrite_context/embeddings.json", "r"
# ) as f:
#     data = json.load(f)


# collection.add(
#     ids=[str(item["id"]) for item in data],
#     documents=[item["text"] for item in data],
#     embeddings=[item["embedding"] for item in data],
# )

# # # Truy vấn

# results = collection.query(
#     query_texts=["hello", "hi"], n_results=10, include=["documents", "distances"]
# )
# print(results["ids"])
with open("test_data.json", "r") as f:
    data = json.load(f)
len_data = len(data)
len5 = 0
len10 = 0
batch = 128
for i in range(0, len_data, batch):
    print(f"Processing {i} to {i + batch}")
    items = data[i : i + batch]
    ids = [str(item["id"]) for item in items]
    rewritten_questions = rewrite.rewrite_queries(
        [item["context"] for item in items], [item["question"] for item in items]
    )
    results = collection.query(
        query_texts=rewritten_questions,
        n_results=10,
        include=["documents", "distances"],
    )
    for k in range(len(results["ids"])):
        p = results["ids"][k]
        top10 = p[:10]
        top5 = p[:5]
        if str(ids[k]) in top10:
            len10 += 1
        if str(ids[k]) in top5:
            len5 += 1
print(f"Number of items in top 10: {float(len10) / len_data}")
print(f"Number of items in top 5: {float(len5) / len_data}")
