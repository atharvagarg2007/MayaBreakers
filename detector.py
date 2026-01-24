import os
import sys
import json
import pickle

import faiss
import imagehash
import numpy as np
from PIL import Image

from step2_embeddings import get_embedding


# ----------------------------------------------------------------------
# Resolve project paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

OFFLINE_DIR = os.path.join(PROJECT_ROOT, "offline")

FAISS_INDEX_PATH = os.path.join(OFFLINE_DIR, "faiss.index")
IMAGE_IDS_PATH = os.path.join(OFFLINE_DIR, "image_ids.json")
PHASH_DB_PATH = os.path.join(OFFLINE_DIR, "phash_db.pkl")


# ----------------------------------------------------------------------
# Load offline artifacts (fail fast if missing)
# ----------------------------------------------------------------------
for path in (FAISS_INDEX_PATH, IMAGE_IDS_PATH, PHASH_DB_PATH):
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Required artifact not found: {path}\n"
            "Run `python build_index.py` first."
        )

index = faiss.read_index(FAISS_INDEX_PATH)

with open(IMAGE_IDS_PATH, "r") as f:
    image_ids = json.load(f)

with open(PHASH_DB_PATH, "rb") as f:
    phash_db = pickle.load(f)


# ----------------------------------------------------------------------
# Duplicate detection
# ----------------------------------------------------------------------
def is_duplicate(image_path, phash_threshold=10, clip_threshold=0.8):
    """
    Args:
        image_path (str): path to query image
        phash_threshold (int): max Hamming distance for pHash filter
        clip_threshold (float): cosine similarity threshold

    Returns:
        (is_duplicate: bool, matched_image: str | None, similarity: float)
    """

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --------------------------------------------------------------
    # pHash pre-filter
    # --------------------------------------------------------------
    query_img = Image.open(image_path).convert("RGB")
    query_phash = imagehash.phash(query_img)

    if not any(
        (query_phash - stored_phash) < phash_threshold
        for stored_phash in phash_db.values()
    ):
        return False, None, 0.0

    # --------------------------------------------------------------
    # CLIP + FAISS
    # --------------------------------------------------------------
    query_emb = get_embedding(image_path)
    query_emb = query_emb.reshape(1, -1).astype("float32")

    D, I = index.search(query_emb, k=1)

    if I[0][0] == -1:
        return False, None, 0.0

    similarity = float(D[0][0])
    match = image_ids[I[0][0]]

    return similarity > clip_threshold, match, similarity


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    COPIES_DIR = os.path.join(PROJECT_ROOT, "data", "copies")
    NEGATIVES_DIR = os.path.join(PROJECT_ROOT, "data", "negatives")

    if os.path.isdir(COPIES_DIR) and os.listdir(COPIES_DIR):
        test_copy = os.path.join(COPIES_DIR, os.listdir(COPIES_DIR)[0])
        print("COPY:", is_duplicate(test_copy))

    if os.path.isdir(NEGATIVES_DIR) and os.listdir(NEGATIVES_DIR):
        test_neg = os.path.join(NEGATIVES_DIR, os.listdir(NEGATIVES_DIR)[0])
        print("NEGATIVE:", is_duplicate(test_neg))
