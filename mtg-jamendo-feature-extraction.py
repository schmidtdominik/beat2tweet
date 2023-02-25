# Run these two commands before:
#!git clone https://github.com/minzwon/sota-music-tagging-models.git
#!pip install librosa==0.10

import os
import tqdm, torch
import sys
sys.path.insert(1, './sota-music-tagging-models/training')

import model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

import subprocess
from pathlib import Path

import torch.nn as nn
import tarfile
import librosa

# Jamendo tags in order
JAMENDO_TAGS = np.array(['genre---alternative','genre---ambient','genre---atmospheric','genre---chillout','genre---classical','genre---dance','genre---downtempo','genre---easylistening','genre---electronic','genre---experimental','genre---folk','genre---funk','genre---hiphop','genre---house','genre---indie','genre---instrumentalpop','genre---jazz','genre---lounge','genre---metal','genre---newage','genre---orchestral','genre---pop','genre---popfolk','genre---poprock','genre---reggae','genre---rock','genre---soundtrack','genre---techno','genre---trance','genre---triphop','genre---world','instrument---acousticguitar','instrument---bass','instrument---computer','instrument---drummachine','instrument---drums','instrument---electricguitar','instrument---electricpiano','instrument---guitar','instrument---keyboard','instrument---piano','instrument---strings','instrument---synthesizer','instrument---violin','instrument---voice','mood/theme---emotional','mood/theme---energetic','mood/theme---film','mood/theme---happy','mood/theme---relaxing'])

FOLDER_PATH = lambda i: f"{i:02d}/"
METADATA_PATH = "raw_30s_cleantags.tsv"

pretrain_dataset = "jamendo" # mtat, msd or jamendo
SR = 16000 # sampling rate
T = 10 # length of extracted sequences
N = 5 # sequences of T seconds to sample from each song

FIXED_METADATA_PATH = METADATA_PATH.split('.tsv')[0]+"-fixed.tsv"

device = "cuda" if torch.cuda.is_available() else "cpu"
hcnn = model.HarmonicCNN().to(device)
state_dict = torch.load(f'sota-music-tagging-models/models/{pretrain_dataset}/hcnn/best_model.pth', map_location=device)
hcnn.load_state_dict(state_dict)
hcnn.eval();

replacements = {"\tgenre": ", genre", "\tmood": ", mood", "\tinstrument": ", instrument"}

def get_top_tags(outputs, k=5, threshold=.4):
  indices = np.where(outputs>threshold)[0]
  try:
      sorted_indices = indices[torch.argsort(-outputs[indices])][:k]
      return [JAMENDO_TAGS[i] for i in sorted_indices]
  except:
      return []

with open(METADATA_PATH, 'r') as f_in, open(FIXED_METADATA_PATH, 'w') as f_out:
    for i, line in enumerate(f_in):
        if i == 0:
          f_out.write(line) 
          continue
        try:
            first_tab_index = line.index('\tgenre')
        except:
          try:
            first_tab_index = line.index('\tinstrument')
          except: first_tab_index = line.index('\tmood')
        
        first_part = line[:first_tab_index+3]
        second_part = line[first_tab_index+3:]
        for old, new in replacements.items():
            second_part = second_part.replace(old, new)
        f_out.write(first_part + second_part)

metadata = pd.read_csv(FIXED_METADATA_PATH, sep="\t")

for tar in tqdm.tqdm(range(100)):

    print(f"Starting file {tar:02d}")

    # Load tags from metadata
    tar_metadata = metadata[metadata.PATH.str[:2]==f"{tar:02d}"]
    metadata_tags = {x.PATH.split("/")[1].split(".")[0]: x.TAGS.split(",") for i, x in tar_metadata.iterrows()}

    folder = FOLDER_PATH(tar)
    try:
        mp3_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp3')]
    except:
        print(f"Error finding {f}.")

    # Define a hook function to store the activations in variable embedding
    def hook(module, input, output):
        global activations
        activations.append(output.detach().numpy()[0])

    dense1 = hcnn.dense1
    handle = dense1.register_forward_hook(hook)

    activations = []
    embedding = {}
    tags = {}

    samples_len = T * SR

    # Load each MP3 file as a NumPy array
    for mp3_file in tqdm.tqdm(mp3_files):

        try:
            array, sr = librosa.load(mp3_file, sr=SR)

            audio_id = mp3_file.split("/")[-1][:-8]
            audio_len = array.shape[0]
            song_tags = [tag.strip() for tag in metadata_tags[audio_id]]
            
            # extract N samples of T seconds evenly spread across the audio
            sample_starts = ((audio_len-samples_len)*np.linspace(0.05, 0.95, N)).astype(int)
            
            # for each extracted sample from the audio file
            for i_start, s in enumerate(sample_starts):

              sample = torch.tensor(array[s:s+samples_len])
              batched_sample = torch.stack([sample[:80000], sample[-80000:]]).to(device)

              output = hcnn(batched_sample)
              sample_tags = get_top_tags(output[0]+output[1], k=5, threshold=.3)

              # sample id is "audio_id"_"start_timestep" (with SR=16000)
              embedding[f"{audio_id}_{s}"] = np.concatenate(activations)
              tags[f"{audio_id}_{s}"] = list(set(song_tags+sample_tags))

              activations.clear()
        
        except Exception as e:  
            print(f"Error with {mp3_file}, {e}")
            continue

    handle.remove()
    
    # store tags and embeddings 
    with open(f'tags_{tar:02d}.json', 'w') as f:
        json.dump(tags, f)
    np.save(f'embeddings_{tar:02d}.npy', embedding)