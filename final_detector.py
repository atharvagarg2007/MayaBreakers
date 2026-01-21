import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from step2_embeddings import get_embedding, orig_embeddings, orig_ids

THRESHOLD = 0.8   # Your learned optimal threshold

def is_duplicate(image_path):
    emb = get_embedding(image_path)
    sims = cosine_similarity([emb], orig_embeddings)[0]
    best_sim = np.max(sims)
    best_idx = np.argmax(sims)
    matched_original = orig_ids[best_idx]

    if best_sim > THRESHOLD:
        return True, best_sim, matched_original
    else:
        return False, best_sim, None
print(is_duplicate(r"C:\Users\athar\Desktop\Atharva Garg 22.04.2023\All folders\Atharva Garg\Classes\MNNIT\Semister\4th sem\Turin playground\project\data\negatives\random7.jpg"))
