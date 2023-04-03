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
    # remove the punctuations
    text_no_punctuation = [word.translate(table) for word in desc]
    
    unk_list = ['unk']
    text_no_punctuation= [word for word in text_no_punctuation if word not in unk_list]
        
    # join the caption words
    caption = ' '.join(text_no_punctuation)
    
    return caption

st.set_page_config(layout="wide")

st.title("Beat2Tweet Evaluation Form")

st.text("Please listen to the below audio recording and rank the captions (1-2: from the most relevant to the least relevant for the audio file).")

with open('outputs/preds_gpt2_enc_summarized.json') as f:
    data_preds_gpt2 = json.load(f)

with open('outputs/preds_lstm_attn_summaries.json') as f:
    data_preds_lstm = json.load(f)

gtp2_keys = list(data_preds_gpt2.keys())
lstm_keys = list(data_preds_lstm.keys())

# Define a function to load the data
def load_data():

    captions ={}
    
    # Load the captions
    random_idx = random.randint(0,len(data_preds_gpt2[gtp2_keys[0]]))
    print(random_idx)
    yt_idx = data_preds_gpt2[gtp2_keys[2]][random_idx]
    print(yt_idx)
    preds_gpt2= data_preds_gpt2[gtp2_keys[1]][random_idx]

    lstm_yt_idx = list(data_preds_lstm[lstm_keys[2]]).index(yt_idx)

    preds_lstm = data_preds_lstm[lstm_keys[1]][lstm_yt_idx]
    lstm_yt_dx = data_preds_lstm[lstm_keys[2]][lstm_yt_idx]
    print(lstm_yt_dx)

    captions["gpt2"] = preprocessing_remove_unk(preds_gpt2)
    captions["lstm"] = preprocessing_remove_unk(preds_lstm)

    # Load the audio file

    audio_file = "music_data/music_data/" + yt_idx + ".wav"

    models_list = list(captions.keys())
    random.shuffle(models_list)
    print(models_list)

    return  audio_file, captions,models_list


# Define a function to create the form
def create_form(audio_file,captions,models_list):
    with st.form("my_form"):
        # Create an empty DataFrame to hold the rankings
        rankings_df = pd.DataFrame(columns=["Model", "Ranking"])
        
        # Display the audio file
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        
        # Loop through each caption and display a slider for ranking it

        for model in models_list:
            st.subheader(str(captions[model]))
            ranking = st.slider("Rank this caption", 1, 2, key=model)
            rankings_df = rankings_df.append({
                "Model": model,
                "Ranking": ranking
            }, ignore_index=True)
        
        # Submit the form
        submitted = st.form_submit_button("Submit")
        if submitted==True:
            if (len(set(rankings_df["Ranking"])) != 2):
                st.write("Try again, assign only one caption per rating!")
            else:
                st.write("Thank you for your submission!")
                st.write(rankings_df)
                # save data to csv
                rankings_df.to_csv('/Users/corinacaraconcea/streamlit_nlp/rankings.csv', mode = 'a' , header=False,index=False)

# Load the data
audio_file,captions_df,models_list = load_data()

# Create the form
create_form(audio_file, captions_df,models_list)

