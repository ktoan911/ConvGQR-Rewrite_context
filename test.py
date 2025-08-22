# Cài đặt
# pip install transformers torch
# pip install neuralcoref --no-binary neuralcoref

from transformers import pipeline

# Tải model vào pipeline
coref_resolver = pipeline("coreference-resolution", model="coref-spanbert-large-en")

# Sử dụng
text = "My sister has a dog. She loves him."
clusters = coref_resolver(text)

print(clusters)
# Kết quả sẽ nhóm "My sister" và "She", "a dog" và "him"
# [
#   [('My sister', (0, 10)), ('She', (23, 26))],
#   [('a dog', (17, 22)), ('him', (33, 36))]
# ]