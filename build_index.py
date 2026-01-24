import os
import json
import pickle

import faiss
import imagehash
import numpy as np
from PIL import Image

from step2_embeddings import get_embedding


def build_index():
    # ------------------------------------------------------------------
    # Resolve project paths (robust on Windows / Linux / anywhere)
    # ------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = BASE_DIR

    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "originals")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "offline")

    FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")
    IMAGE_IDS_PATH = os.path.join(OUTPUT_DIR, "image_ids.json")
    PHASH_DB_PATH = os.path.join(OUTPUT_DIR, "phash_db.pkl")

    EMBEDDING_DIM = 512

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"DATA_DIR not found: {DATA_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Build embeddings + pHash DB
    # ------------------------------------------------------------------
    embeddings = []
    image_ids = []
    phash_db = {}

    print("Starting offline preprocessing...")
    print("Data dir:", DATA_DIR)
    print("Output dir:", OUTPUT_DIR)

    for img_name in sorted(os.listdir(DATA_DIR)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(DATA_DIR, img_name)
        print(f"Processing: {img_name}")

        # pHash
        img = Image.open(img_path).convert("RGB")
        phash_db[img_name] = imagehash.phash(img)

        # CLIP embedding
        emb = get_embedding(img_path)
        embeddings.append(emb)
        image_ids.append(img_name)

    if not embeddings:
        raise RuntimeError("No valid images found in data/originals")

    embeddings = np.asarray(embeddings, dtype="float32")

    print("Total images indexed:", len(image_ids))
    print("Embedding shape:", embeddings.shape)

    # ------------------------------------------------------------------
    # Build FAISS index (cosine similarity via inner product)
    # ------------------------------------------------------------------
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(IMAGE_IDS_PATH, "w") as f:
        json.dump(image_ids, f)

    with open(PHASH_DB_PATH, "wb") as f:
        pickle.dump(phash_db, f)

    print("\nOffline preprocessing complete")
    print("Saved files:")
    print(" -", FAISS_INDEX_PATH)
    print(" -", IMAGE_IDS_PATH)
    print(" -", PHASH_DB_PATH)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    build_index()
