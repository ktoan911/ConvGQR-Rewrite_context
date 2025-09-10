import json

from sentence_transformers import SentenceTransformer

# Dùng GPU
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")

with open("test_text.json", "r") as f:
    texts = json.load(f)


def get_embeddings_to_json(s, output_file="embeddings.json", batch_size=256):
    results = []
    for i in range(0, len(s), batch_size):
        batch = [item["text"] for item in s[i : i + batch_size]]
        ids = [item["id"] for item in s[i : i + batch_size]]

        # model.encode sẽ tự chạy trên GPU vì đã set device="cuda"
        embeddings = model.encode(
            batch, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size
        )

        for text, emb, id in zip(batch, embeddings, ids):
            results.append(
                {
                    "text": text,
                    "embedding": emb.tolist(),
                    "id": id,
                }
            )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(results)} embeddings vào {output_file}")


get_embeddings_to_json(texts, output_file="embeddings.json", batch_size=64)
