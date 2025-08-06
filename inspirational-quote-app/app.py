import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ✅ Load model
model = load_model("quote_model.h5")

# ✅ Tokenizer setup
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
training_quotes = [
    "Success is not final, failure is not fatal.",
    "Believe in yourself and all that you are.",
    "Don’t watch the clock; do what it does.",
    "It always seems impossible until it’s done.",
    "The only way to do great work is to love what you do."
]
tokenizer.fit_on_texts(training_quotes)

# ✅ Predict mood function
def predict_mood(user_quote):
    seq = tokenizer.texts_to_sequences([user_quote])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    prediction = model.predict(padded)[0]

    moods = ['Positive', 'Neutral', 'Motivational']
    mood = moods[np.argmax(prediction)]

    return f"📝 Quote: {user_quote}", f"🧠 Predicted Mood: {mood}"

# ✅ Gradio UI
gr.Interface(
    fn=predict_mood,
    inputs=gr.Textbox(placeholder="Type your favorite quote here..."),
    outputs=["text", "text"],
    title="💡 Quote Mood Analyzer",
    description="Type a quote and get its mood using an LSTM model.",
).launch()
