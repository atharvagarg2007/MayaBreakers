import faiss
import json
import pickle
import imagehash
import numpy as np
from PIL import Image

from step2_embeddings import get_embedding

# LOAD OFFLINE ARTIFACTS
FAISS_INDEX_PATH = "offline/faiss.index"
IMAGE_IDS_PATH = "offline/image_ids.json"
PHASH_DB_PATH = "offline/phash_db.pkl"

# Load FAISS vector database
index = faiss.read_index(FAISS_INDEX_PATH)

# Load image IDs (mapping index -> image name)
with open(IMAGE_IDS_PATH, "r") as f:
    image_ids = json.load(f)

# Load pHash database
with open(PHASH_DB_PATH, "rb") as f:
    phash_db = pickle.load(f)


# DUPLICATE DETECTION FUNCTION
def is_duplicate(image_path, phash_threshold=10, clip_threshold=0.8):
    """
    Returns:
    (is_duplicate: bool, matched_image: str or None, similarity: float)
    """

    # pHash filtering 
    query_img = Image.open(image_path).convert("RGB")
    query_phash = imagehash.phash(query_img)

    candidates_exist = False
    for _, stored_phash in phash_db.items():
        if (query_phash - stored_phash) < phash_threshold:
            candidates_exist = True
            break

    # Fast reject if no pHash match
    if not candidates_exist:
        return False, None, 0.0

    # CLIP + FAISS
    query_emb = get_embedding(image_path)
    query_emb = query_emb.reshape(1, -1).astype("float32")

    # Search nearest neighbor
    D, I = index.search(query_emb, k=1)

    best_similarity = float(D[0][0])
    best_match = image_ids[I[0][0]]

    if best_similarity > clip_threshold:
        return True, best_match, best_similarity
    else:
        return False, None, best_similarity



# TEST
if __name__ == "__main__":
    test_image = "data/originals/sample.jpg" 
    result = is_duplicate(test_image)
    print(result)
