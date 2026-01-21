import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from step2_embeddings import get_embedding, orig_embeddings, orig_ids

copy_dir = "data/copies"
neg_dir = "data/negatives"

similarities = []
labels = []

# Copies = duplicates (label 1)
for file in os.listdir(copy_dir):
    emb = get_embedding(os.path.join(copy_dir, file))
    sims = cosine_similarity([emb], orig_embeddings)[0]
    similarities.append(np.max(sims))
    labels.append(1)

# Negatives = non-duplicates (label 0)
for file in os.listdir(neg_dir):
    emb = get_embedding(os.path.join(neg_dir, file))
    sims = cosine_similarity([emb], orig_embeddings)[0]
    similarities.append(np.max(sims))
    labels.append(0)

print("Collected similarity scores:", len(similarities))

from sklearn.metrics import f1_score, classification_report

best_f1 = 0
best_t = 0

for t in np.arange(0.7, 0.98, 0.02):
    preds = [1 if s > t else 0 for s in similarities]
    f1 = f1_score(labels, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest Threshold:", best_t)
print("Best F1 Score:", best_f1)

# Show final detailed metrics
final_preds = [1 if s > best_t else 0 for s in similarities]
print("\nClassification Report:")
print(classification_report(labels, final_preds))

