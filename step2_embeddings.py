import os
import clip
import torch
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_embedding(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# Load originals
orig_dir = "data/originals"
orig_embeddings = []
orig_ids = []

for file in os.listdir(orig_dir):
    path = os.path.join(orig_dir, file)
    orig_embeddings.append(get_embedding(path))
    orig_ids.append(file.split(".")[0])

orig_embeddings = np.array(orig_embeddings)

print("Original embeddings shape:", orig_embeddings.shape)
