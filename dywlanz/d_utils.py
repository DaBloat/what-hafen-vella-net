import streamlit as st
import whisper
import torch
import torch.nn.functional as F
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import emoji
import re

def censor_profanity(text):
    bad_words = ['damn', 'shit', 'fuck', 'bitch', 'asshole']  # Add more
    for word in bad_words:
        pattern = re.compile(rf'\b{re.escape(word)}\b', flags=re.IGNORECASE)
        text = pattern.sub(lambda m: '*' * len(m.group()), text)
    return text

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_sentiment_model():
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    model = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=3)
    model.load_state_dict(torch.load("fine_tuned_model_14.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

def predict_sentiment_text(transcribed_text, tokenizer, sentiment_model):
    label_names = [f'Negative {emoji.emojize('\N{unamused face}')}', f'Neutral {emoji.emojize('\N{neutral face}')}', f'Positive {emoji.emojize('\N{smiling face with smiling eyes}')}']
    encoding = tokenizer(transcribed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = sentiment_model(**encoding)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        
    return label_names[pred], f'{confidence:.2f}'
