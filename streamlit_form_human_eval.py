import streamlit as st
import pandas as pd
import json
import random
import numpy as np
import os
import csv

def cleaning_text(caption):
    table = str.maketrans('','',string.punctuation)
    caption.replace("-"," ")
    # split the sentences into words
    desc = caption.split()
    #converts to lower case
    desc = [word.lower() for word in desc]
    #remove punctuation from each token
    desc = [word.translate(table) for word in desc]
    #remove hanging 's and a 
    desc = [word for word in desc if(len(word)>1)]
    #remove tokens with numbers in them
    desc = [word for word in desc if(word.isalpha())]
    #convert back to string
    caption = ' '.join(desc)

    return caption

st.title("Beat2Tweet Evaluation Form")

st.text("Please listen to the below audio recording and rank the captions from 1(worst suited) to 5 (best suited).")

with open('/Users/corinacaraconcea/streamlit_nlp/outputs/preds_lstm_noattn_chataug.json') as f:
    data_preds_gpt2_notag_chataug = json.load(f)

with open('/Users/corinacaraconcea/streamlit_nlp/outputs/preds_lstm_noattn_noaug.json') as f:
    data_preds_gpt2_notag_noaug = json.load(f)

with open('/Users/corinacaraconcea/streamlit_nlp/outputs/preds_lstm_attn_noaug.json') as f:
    data_preds_lstm_attn_no_tag_no_aug = json.load(f)

with open('/Users/corinacaraconcea/streamlit_nlp/outputs/preds_lstm_noattn_noaug.json') as f:
    data_preds_lstm_no_attn_no_tag_no_aug = json.load(f)

# with open('outputs/preds_notag_noaug.json') as f:
#     data_preds_notag_noaug = json.load(f)

# Define a function to load the data
def load_data():
    
    # Load the captions
    random_idx = random.randint(0,len(data_preds_lstm_no_attn_no_tag_no_aug))
    yt_idx = data_preds_lstm_no_attn_no_tag_no_aug['track_ids'][random_idx]
    lstm_no_attn_no_tag_no_aug_pred = data_preds_lstm_no_attn_no_tag_no_aug['eval_pred_captions'][random_idx]
    lstm_attn_no_tag_no_aug_pred = data_preds_lstm_attn_no_tag_no_aug['eval_pred_captions'][random_idx]
    gpt2_notag_chataug_pred = data_preds_gpt2_notag_chataug['eval_pred_captions'][random_idx]
    gpt2_notag_noaug_pred = data_preds_gpt2_notag_noaug['eval_pred_captions'][random_idx]
    captions = [lstm_no_attn_no_tag_no_aug_pred,lstm_attn_no_tag_no_aug_pred,gpt2_notag_chataug_pred,gpt2_notag_noaug_pred]

    # Load the audio file
    print(yt_idx)
    audio_file = "/Users/corinacaraconcea/streamlit_nlp/music_data/music_data/" + yt_idx + ".wav"

    return  audio_file, captions


# Define a function to create the form
def create_form(audio_file,captions):
    with st.form("my_form"):
        # Create an empty DataFrame to hold the rankings
        rankings_df = pd.DataFrame(columns=["Model", "Ranking"])
        
        # Display the audio file
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        
        # Loop through each caption and display a slider for ranking it
        for i,caption in enumerate(captions):
            st.subheader(caption)
            ranking = st.slider("Rank this caption", 1, 5, key=i)
            rankings_df = rankings_df.append({
                "Model": i,
                "Ranking": ranking
            }, ignore_index=True)
        
        # Submit the form
        submitted = st.form_submit_button("Submit")
        if submitted==True:
            st.write("Thank you for your submission!")
            st.write(rankings_df)
            # save data to csv
            rankings_df = rankings_df[['Ranking']]
            rankings_df.T.to_csv('/Users/corinacaraconcea/streamlit_nlp/rankings.csv',mode = 'a' , header=False)

# Load the data
audio_file,captions_df = load_data()

# Create the form
create_form(audio_file, captions_df)
