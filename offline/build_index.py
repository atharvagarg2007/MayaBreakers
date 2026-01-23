import os
import json
import pickle

import faiss
import imagehash
import numpy as np
from PIL import Image

from step2_embeddings import get_embedding

DATA_DIR = "data/originals"  
OUTPUT_DIR = "offline"

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")
IMAGE_IDS_PATH = os.path.join(OUTPUT_DIR, "image_ids.json")
PHASH_DB_PATH = os.path.join(OUTPUT_DIR, "phash_db.pkl")

EMBEDDING_DIM = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

embeddings = []
image_ids = []
phash_db = {}

print("🔄 Starting offline preprocessing...")

for img_name in os.listdir(DATA_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(DATA_DIR, img_name)
    print(f"Processing: {img_name}")

    # 1️⃣ pHash
    img = Image.open(img_path).convert("RGB")
    phash_db[img_name] = imagehash.phash(img)

    # 2️⃣ CLIP embedding
    emb = get_embedding(img_path)   # (512,)
    embeddings.append(emb)
    image_ids.append(img_name)

embeddings = np.array(embeddings).astype("float32")

print("Total images indexed:", len(image_ids))
print("Embedding shape:", embeddings.shape)


index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(embeddings)


faiss.write_index(index, FAISS_INDEX_PATH)

with open(IMAGE_IDS_PATH, "w") as f:
    json.dump(image_ids, f)

with open(PHASH_DB_PATH, "wb") as f:
    pickle.dump(phash_db, f)

print("✅ Offline preprocessing complete")
print("Saved files:")
print(" -", FAISS_INDEX_PATH)
print(" -", IMAGE_IDS_PATH)
print(" -", PHASH_DB_PATH)
