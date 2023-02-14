import numpy as np
import pandas as pd
from datasets import load_dataset

# Load embeddings, which is a dictionary (ytid -> 512-dim feature vector)
try:
    embeddings = np.load("embeddings.npy", allow_pickle=True).item()
    ids = sorted(list(embeddings.keys()))
    print(f"Loaded {len(ids)} feature vectors.")
except:
    print("Error loading embeddings")

# Load MusicCaps captions
try:
  ds = load_dataset('google/MusicCaps', split='train')
  df = pd.DataFrame({'ytid': ds['ytid'],
                     'caption': ds['caption'],
                     'is_eval': ds['is_audioset_eval']})
  # discard entries without embedding
  df_captions = df[df.ytid.isin(ids)]
  print(f"Loaded {df.shape[0]} captions, using {df_captions.shape[0]} ")
except:
    print("Error loading captions")


# TODO: create captions dataset vocabulary

# TODO: tokenize captions

# TODO: define LSTM model with 512-dimensional input

# TODO: train LSTM model