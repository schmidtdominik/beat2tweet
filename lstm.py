import numpy as np
import string
import os
import torch
import torchvision
import pandas as pd
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import warnings
# UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
# or sourceTensor.clone().detach().requires_grad_(True) rather than torch.tensor(sourceTensor)
warnings.simplefilter("ignore", category=UserWarning)

# TODO: refactor

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
    df = pd.DataFrame({'ytid': ds['ytid'], 'caption': ds['caption'], 'is_eval': ds['is_audioset_eval']})
    # discard entries without embedding
    df_captions = df[df.ytid.isin(ids)]
    print(f"Loaded {df.shape[0]} captions, using {df_captions.shape[0]} ")
except:
    print("Error loading captions")


def cleaning_text(caption):
    table = str.maketrans('', '', string.punctuation)
    caption.replace("-", " ")
    # converts to lower case and remove punctuation
    desc = [word.lower().translate(table) for word in caption.split()]
    # remove hanging 's and a and remove tokens with numbers
    desc = [word for word in desc if (len(word) > 1 and word.isalpha())]
    # convert back to string
    caption = ' '.join(desc)

    return caption

def preprocess_captions(captions):
    for audio_file, caption in captions.items():
        caption = cleaning_text(caption)
        captions[audio_file] = caption

    return captions

captions = dict(zip(df_captions.ytid, df_captions.caption))
captions = preprocess_captions(captions)

# define a vocabulary
def text_vocabulary(descriptions):
    captions = list(descriptions.values())
    vocab = set(['<start>', '<end>'])
    for caption in captions:
        for token in caption.strip().split():
            vocab.add(token)
    return vocab

# force <pad> to have idx 0 (convention)
vocab = ['<pad>'] + list(text_vocabulary(captions))

word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

def tokenize(caption):
    caption = cleaning_text(caption)
    token_list = []
    # Add <start> to the beginning and <end> to the end of each caption
    caption_list = ["<start>"] + caption.split() + ["<end>"]
    token_list = [word_to_idx[word] for word in caption_list]
    return token_list

df_captions.loc[:,'tokenized_caption'] = df_captions['caption'].apply(tokenize)
train_df = df_captions[~df_captions.is_eval]
eval_df = df_captions[df_captions.is_eval]

# Define the audio captioning dataset
class AudioCaptionDataset(Dataset):
    def __init__(self, captions, embeddings):
        self.captions = captions
        self.embeddings = embeddings

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        ytid = caption['ytid']
        embedding = self.embeddings[ytid]
        tokenized_caption = torch.LongTensor(caption['tokenized_caption'])
        return {"embedding": embedding, "tokenized_caption": tokenized_caption}

# Define the collate function for the audio captioning dataset
def collate_fn_try(batch):
    embeddings = []
    captions = []
    for b in batch:
        embeddings.append(torch.from_numpy(b['embedding']))
        captions.append(torch.tensor(b['tokenized_caption']))
    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    padded_captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=word_to_idx["<pad>"])  # Use the <pad> index for padding
    return padded_embeddings, padded_captions


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(input.device)
        else:
            if num_layers == 1: h0 = hidden.unsqueeze(0)
            else: h0 = torch.cat((hidden.unsqueeze(0),
                            torch.zeros(num_layers-1, input.size(0), hidden_size).to(device)), dim=0)
        hidden = (h0, c0)

        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)

        return output, hidden


# Instantiate the model
vocab_size = len(vocab)
input_size = vocab_size
hidden_size = 512
output_size = vocab_size
num_layers = 1
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# Instantiate the dataset
train_dataset = AudioCaptionDataset(train_df.to_dict('records'),
                {id: embeddings[id] for id in train_df.ytid.unique()})
eval_dataset = AudioCaptionDataset(eval_df.to_dict('records'),
                {id: embeddings[id] for id in eval_df.ytid.unique()})

# Train the model
lr = 1e-4
batch_size = 64
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_try)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_try)

def generate_caption(i, max_caption_length=64, show_true_caption=True, show_ytid=True):

  ytid = df_captions.iloc[i]["ytid"]
  true_caption = df_captions.iloc[i]["caption"]
  embedding = torch.from_numpy(embeddings[ytid])

  x = embedding.unsqueeze(0).to(device, dtype=torch.float)
  model.eval()
  # breaks if starting sequence is only one token (?)
  caption = torch.tensor([word_to_idx[word] for word in ['<pad>', '<start>']]).unsqueeze(0).to(device)

  # Generate the caption word by word
  with torch.no_grad():
      while caption[0][-1] != word_to_idx['<end>'] and len(caption[0]) < max_caption_length:
          logits, hidden = model(caption[:, :-1], x)
          predicted_word_index = logits.argmax(-1)[:, -1].item()
          predicted_word = idx_to_word[predicted_word_index]
          caption = torch.cat([caption, torch.tensor([[predicted_word_index]], dtype=torch.long).to(device)], dim=1)

  predicted_caption = ' '.join([idx_to_word[word_idx] for word_idx in caption[0].tolist()][2:-1])

  if show_ytid: print(f"https://www.youtube.com/watch?v={ytid}")
  if show_true_caption: print(f"True caption: {true_caption}\n")
  print(f"Predicted caption: {predicted_caption}")

print("Start training...")
# Train the model
for epoch in range(num_epochs):
    model.train()  # set model to train mode
    for i, (x, captions) in enumerate(train_dataloader):
        x = x.to(device, dtype=torch.float)
        captions = captions.to(device, dtype=torch.long)
        optimizer.zero_grad()
        logits, hidden = model(captions[:, :-1], x)
        # print(logits.shape)
        loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])
        loss = loss_fn(logits.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

    model.eval()  # set model to eval mode
    eval_loss = 0
    with torch.no_grad():
        for x, captions in eval_dataloader:
            x = x.to(device, dtype=torch.float)
            captions = captions.to(device, dtype=torch.long)
            logits, hidden = model(captions[:, :-1], x)
            loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])
            loss = loss_fn(logits.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            eval_loss += loss.item() * x.size(0)  # accumulate loss for entire eval dataset
        eval_loss /= len(eval_dataloader.dataset)  # compute average eval loss

    print(f"Epoch {epoch}, train loss {loss.item():.4f}, eval loss {eval_loss:.4f}")
    generate_caption(0, show_ytid=(epoch == 0), show_true_caption=(epoch == 0))
    generate_caption(1, show_ytid=(epoch == 0), show_true_caption=(epoch == 0))
    print("\n")