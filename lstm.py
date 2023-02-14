import numpy as np

# Load embeddings, which is a dictionary (ytid -> 512-dim feature vector)
try:
    embeddings = np.load("embeddings.npy", allow_pickle=True).item()
    print(f"Loaded {len(embeddings.keys())} feature vectors.")
except:
    print("Error loading embeddings")