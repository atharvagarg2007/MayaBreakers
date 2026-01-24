# Near-Duplicate Image Detection (NDID)
### Turin’s Playground – AI-Based Project

## 📌 Problem Statement
Modern digital platforms such as **social media, e-commerce, content hosting, and news aggregation systems** handle **millions of image uploads every day**. A large fraction of these uploads are **duplicate or near-duplicate images**—the same image uploaded multiple times or slightly modified versions of an existing one.

Traditional systems struggle to detect such duplicates when images are:
- Resized
- Cropped
- Compressed
- Color-adjusted
- Watermarked or lightly edited

This project proposes an **AI-driven Near-Duplicate Image Detection (NDID) system** that is robust to these transformations.

---

## 🎯 Project Goals

### 🔹 Storage Optimization
- Eliminate redundant image storage
- Save large-scale cloud storage costs

### 🔹 Spam & Content Integrity
- Prevent repost bots from flooding platforms
- Protect original creators from copyright infringement

### 🔹 Search & Feed Relevance
- Avoid repeated thumbnails for the same content
- Improve user experience in feeds and search results

---

## 🧠 Proposed Technical Architecture

### 1. Image Embedding Generation
- Convert images into vector embeddings using **pretrained vision models**
- Supported models:
  - CNN-based models (e.g., ResNet – baseline)
  - Vision-language models (e.g., CLIP Image Encoder – preferred)

### 2. Embedding Storage
- Store embeddings in a searchable vector space
- Initial approach:
  - In-memory vector storage
- Extendable to:
  - FAISS, Milvus, Pinecone

### 3. Similarity Comparison
- Compare incoming image embeddings with existing ones using:
  - Cosine Similarity
  - Euclidean Distance

### 4. Duplicate Classification
- Images are classified as:
  - Duplicate
  - Near-Duplicate
  - Unique
- Decision is based on a configurable similarity threshold

---

## 🔁 System Pipeline

Image Upload
↓
Preprocessing
↓
Embedding Generation (ResNet / CLIP)
↓
Vector Storage
↓
Similarity Search
↓
Threshold-Based Classification
---

## 🛠️ Tech Stack
- **Language:** Python
- **Models:** ResNet, CLIP
- **Similarity Metrics:** Cosine Similarity, Euclidean Distance
- **Storage:** In-memory vectors (initial phase)

---

## ✨ Key Features
- Robust to common image transformations
- Scalable and modular architecture
- Model-agnostic embedding pipeline
- Designed for real-world, large-scale platforms
- Made a UI for this project
---

## 📊 Evaluation Metrics
- Precision and Recall for duplicate detection
- False positive and false negative rates
- Embedding similarity distribution analysis
---
## 👥 Contributors
- [Atharva Garg (Team Leader)](https://github.com/atharvagarg2007)
- [Anirudh Gowri Sankaran](https://github.com/anirudh5-creator)
- [Joel Jose Manoj](https://github.com/Joel5799)
- [Justin Biju](https://github.com/J-ustinJ)

---

This project is developed as part of the **Turin’s Playground** competition.  
