from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, Audio
import pandas as pd
import streamlit as st
import emoji

@st.cache_resource
def load_tonal_model():
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2ForSequenceClassification.from_pretrained('dabloat-wav2rec2-emotion-aug')
    return processor, model

def predict_voice_sentiment(audio, model, processor):
    
    def preprocess_function(batch):
        audio = batch["path"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
        batch["input_values"] = inputs.input_values[0]
        batch["labels"] = batch["sentiment"]
        return batch
    
    class_names = [f"Negative {emoji.emojize('\N{unamused face}')}", f"Neutral {emoji.emojize('\N{neutral face}')}", f"Positive  {emoji.emojize('\N{smiling face with smiling eyes}')}"]
    dummy_sentiment = 0
    data = pd.DataFrame({'path':[audio], 'sentiment': [dummy_sentiment]})
    data = Dataset.from_pandas(data)
    data = data.cast_column('path', Audio(sampling_rate=16_000))
    data = data.map(preprocess_function, remove_columns=["path", "sentiment"])
    data.set_format(type="torch", columns=["input_values", "labels"])
    training_args = TrainingArguments(
                            output_dir="./wav2vec2-emotion",
                            per_device_eval_batch_size=32,
                            dataloader_drop_last=False,
                            do_train=False,
                            do_eval=True,
                            report_to="none"
                        )
    predictor = Trainer(
                model=model,
                args=training_args
                )
    sentiment = predictor.predict(data)
    logits = torch.tensor(sentiment.predictions)
    probs = F.softmax(logits, dim=-1)
    return class_names[int(np.argmax(logits))], f'{probs.numpy().tolist()[0][int(np.argmax(logits))]:.2f}'
