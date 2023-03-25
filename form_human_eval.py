import streamlit as st
import pandas as pd
import json
import random
import numpy as np
import csv
import re
import string


def preprocessing_remove_unk(text_input):
    # remove punctuations
    desc = re.sub(r'[^\w\s]',' ',text_input)
    table = str.maketrans('','',string.punctuation)

    # turn uppercase letters into lowercase ones
    desc = text_input.lower()

    # split into words
    desc = desc.split(' ')
    
    try: 
        # remove <unk> tokens
        desc.remove("<unk>")
    except ValueError:
        desc = desc
    try: 
        # remove <unk> tokens
        desc.remove("unk")
    except ValueError:
        desc = desc
        
    # remove the punctuations
    text_no_punctuation = [word.translate(table) for word in desc]

    # join the caption words
    caption = ' '.join(text_no_punctuation)
    
    return caption

st.set_page_config(layout="wide")

st.title("Beat2Tweet Evaluation Form")

st.text("Please listen to the below audio recording and rank the captions (1-3: from the most relevant to the least relevant for the audio file).")

with open('outputs/preds_gpt2_enc_noaug.json') as f:
    data_preds_gpt2_enc_noaug= json.load(f)

with open('outputs/preds_gpt2_enc_chataug.json') as f:
    data_preds_gpt2_enc_chataug= json.load(f)

with open('outputs/preds_lstm_attn_noaug.json') as f:
    data_preds_lstm_attn_noaug = json.load(f)

gtp2_keys = list(data_preds_gpt2_enc_noaug.keys())
lstm_keys = list(data_preds_lstm_attn_noaug.keys())

# Define a function to load the data
def load_data():

    captions ={}
    
    # Load the captions
    random_idx = random.randint(0,len(data_preds_gpt2_enc_noaug[gtp2_keys[0]]))
    print(random_idx)
    yt_idx = data_preds_gpt2_enc_noaug[gtp2_keys[2]][random_idx]
    print(yt_idx)
    preds_gpt2_enc_noaug = data_preds_gpt2_enc_noaug[gtp2_keys[1]][random_idx]
    preds_gpt2_enc_chataug = data_preds_gpt2_enc_chataug[gtp2_keys[1]][random_idx]

    preds_lstm_attn_noaug = data_preds_lstm_attn_noaug[lstm_keys[1]][random_idx]
    lstm_yt_dx = data_preds_lstm_attn_noaug[lstm_keys[2]][random_idx]
    print(lstm_yt_dx)

    captions["gpt2_enc_noaug"] = preprocessing_remove_unk(preds_gpt2_enc_noaug)
    captions["gpt2_enc_chataug"] = preprocessing_remove_unk(preds_gpt2_enc_chataug)
    captions["lstm_attn_noaug"] = preprocessing_remove_unk(preds_lstm_attn_noaug)

    # Load the audio file

    audio_file = "music_data/music_data/" + yt_idx + ".wav"

    models_list = list(captions.keys())
    random.shuffle(models_list)
    print(models_list)

    return  audio_file, yt_idx, captions, models_list


# Define a function to create the form
def create_form(audio_file, yt_idx, captions, models_list):
    with st.form("my_form"):
        # Create an empty DataFrame to hold the rankings
        rankings_df = pd.DataFrame(columns=["Model", "Ranking", "Song_ID"])
        
        # Display the audio file
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        
        # Loop through each caption and display a slider for ranking it

        for model in models_list:
            st.subheader(str(captions[model]))
            ranking = st.slider("Rank this caption", 1, 3, key=model)
            rankings_df = rankings_df.append({
                "Model": model,
                "Ranking": ranking,
                "Song_ID": yt_idx
            }, ignore_index=True)
        
        # Submit the form
        submitted = st.form_submit_button("Submit")
        if submitted==True:
            if (len(set(rankings_df["Ranking"])) != 3):
                st.write("Try again, assign only one caption per rating!")
            else:
                st.write("Thank you for your submission!")
                st.write(rankings_df)
                # save data to csv
                rankings_df.to_csv('/Users/corinacaraconcea/streamlit_nlp/rankings.csv', mode = 'a' , header=False,index=False)

# Load the data
audio_file, yt_idx, captions_df, models_list = load_data()

# Create the form
create_form(audio_file, yt_idx, captions_df, models_list)

