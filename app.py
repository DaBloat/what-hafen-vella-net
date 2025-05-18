import streamlit as st
from dywlanz import d_utils
from dabloat import db_utils


whisper_model = d_utils.load_whisper_model()

text_tokenizer, sentiment_model = d_utils.load_sentiment_model()
voice_processor, voice_model = db_utils.load_tonal_model()

st.title("Audio Sentiment Analysis")
st.write("Upload an audio file or enter text manually.")
audio_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
        
    st.audio("temp_audio.wav")
    st.info('Transcribing Audio')
    result = whisper_model.transcribe("temp_audio.wav")
    transcribed_text = result["text"]
    st.success("Transcription complete.")
    st.markdown(f"**Transcribed Text:** {transcribed_text}")
    
    if st.button("Analyze Sentiment"):
        sentiment, confidence = d_utils.predict_sentiment_text(transcribed_text, text_tokenizer, sentiment_model)
        voice_sentiment, voice_confidence = db_utils.predict_voice_sentiment('temp_audio.wav', voice_model, voice_processor)
        st.markdown(f"**Predicted Text Sentiment:** {sentiment}")
        st.markdown(f"**Text Sentiment Confidence:** {confidence}")
        st.markdown(f"**Predicted Voice Sentiment:** {voice_sentiment}")
        st.markdown(f"**Voice Sentiment Confidence:** {voice_confidence}")