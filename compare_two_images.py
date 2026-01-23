import os
import imagehash
import numpy as np
from PIL import Image

from step2_embeddings import get_embedding

# Checking Similarities

def cosine_similarity(a, b):
    return float(np.dot(a, b))


def similarity_to_percentage(similarity, threshold=0.85, max_sim=0.95):
    
    if similarity <= threshold:
        return 0.0

    similarity = min(similarity, max_sim)
    percent = (similarity - threshold) / (max_sim - threshold)
    return round(percent * 100, 2)


# Duplicate checking function

def are_duplicates(img_path_1, img_path_2, phash_threshold=25, clip_threshold=0.85, ):

    if not os.path.isfile(img_path_1):
        raise FileNotFoundError(f"Image not found: {img_path_1}")

    if not os.path.isfile(img_path_2):
        raise FileNotFoundError(f"Image not found: {img_path_2}")

    # pHash similarity

    img1 = Image.open(img_path_1).convert("RGB")
    img2 = Image.open(img_path_2).convert("RGB")

    phash1 = imagehash.phash(img1)
    phash2 = imagehash.phash(img2)

    phash_distance = phash1 - phash2

    if phash_distance > phash_threshold:
        return False, 0.0, 0.0, phash_distance

    # CLIP similarity

    emb1 = get_embedding(img_path_1)
    emb2 = get_embedding(img_path_2)

    clip_similarity = cosine_similarity(emb1, emb2)
    match_percentage = similarity_to_percentage(clip_similarity, threshold=clip_threshold)

    is_duplicate = clip_similarity >= clip_threshold

    return is_duplicate, match_percentage, clip_similarity, phash_distance


# Testing

if __name__ == "__main__":
    print("Image Duplicate Checker\n")

    img1 = input("Enter path to first image: ").strip()
    img2 = input("Enter path to second image: ").strip()

    is_dup, percent, similarity, phash_dist = are_duplicates(img1, img2)

    print("\nRESULT")
    print("-------")
    print("Duplicate:", is_dup)
    print(f"Match Percentage: {percent}%")
    print(f"CLIP Similarity: {similarity:.4f}")
    print(f"pHash Distance: {phash_dist}")
