import os
import sys
import json
import pickle

import faiss
import imagehash
import numpy as np
from PIL import Image

# FORCE PROJECT ROOT TO PATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from step2_embeddings import get_embedding

# LOAD OFFLINE ARTIFACTS
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "offline", "faiss.index")
IMAGE_IDS_PATH   = os.path.join(PROJECT_ROOT, "offline", "image_ids.json")
PHASH_DB_PATH    = os.path.join(PROJECT_ROOT, "offline", "phash_db.pkl")

index = faiss.read_index(FAISS_INDEX_PATH)

with open(IMAGE_IDS_PATH, "r") as f:
    image_ids = json.load(f)

with open(PHASH_DB_PATH, "rb") as f:
    phash_db = pickle.load(f)

# DUPLICATE DETECTION
def is_duplicate(image_path, phash_threshold=10, clip_threshold=0.8):
    """
    Returns:
    (is_duplicate: bool, matched_image: str or None, similarity: float)
    """

    query_img = Image.open(image_path).convert("RGB")
    query_phash = imagehash.phash(query_img)

    # pHash pre-filter
    if not any(
        (query_phash - stored_phash) < phash_threshold
        for stored_phash in phash_db.values()
    ):
        return False, None, 0.0

    # CLIP + FAISS
    query_emb = get_embedding(image_path).reshape(1, -1).astype("float32")
    D, I = index.search(query_emb, k=1)

    if I[0][0] == -1:
        return False, None, 0.0

    similarity = float(D[0][0])
    match = image_ids[I[0][0]]

    return similarity > clip_threshold, match, similarity

# EXAMPLE TESTS
if __name__ == "__main__":
    # Test a duplicate
    copy_img = os.path.join(PROJECT_ROOT, "data", "copies", os.listdir(os.path.join(PROJECT_ROOT, "data", "copies"))[0])
    print("COPY:", is_duplicate(copy_img))

    # Test a negative
    neg_img = os.path.join(PROJECT_ROOT, "data", "negatives", os.listdir(os.path.join(PROJECT_ROOT, "data", "negatives"))[0])
    print("NEGATIVE:", is_duplicate(neg_img))
