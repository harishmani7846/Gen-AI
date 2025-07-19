import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = tf.keras.models.load_model("news_sentiment.h5")

# Setup tokenizer (must match training tokenizer)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(["This is a sample sentence for tokenizer"])  # placeholder fit

# Preprocessing function
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)
    return padded

# Prediction function
def predict_sentiment(text):
    processed = preprocess_text(text)
    prediction = model.predict(processed)[0][0]
    sentiment = "Positive ✅" if prediction > 0.5 else "Negative ❌"
    return f"📰 Headline: {text}\n🔍 Sentiment: {sentiment} ({prediction:.2f})"

# Gradio UI
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type a news headline here..."),
    outputs="text",
    title="🗞️ News Headline Sentiment Analyzer",
    description="Type a news headline below to see if it's Positive or Negative."
)

iface.launch()
