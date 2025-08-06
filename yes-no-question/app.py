import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load model
model = tf.keras.models.load_model("yesno_model.h5")

# Setup tokenizer (should match the one used in training)
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(["yes no maybe question test sample"])  # dummy fit

# Preprocess input
def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20)
    return padded

# Predict function
def answer_yes_no(question):
    processed = preprocess(question)
    pred = model.predict(processed)[0][0]
    answer = "Yes ✅" if pred > 0.5 else "No ❌"
    return f"Question: {question}\nAnswer: {answer} ({pred:.2f})"

# Gradio UI
iface = gr.Interface(
    fn=answer_yes_no,
    inputs=gr.Textbox(lines=2, placeholder="Ask a Yes/No question..."),
    outputs="text",
    title="🤖 Yes/No Question Answering AI",
    description="Ask any yes/no question, and get an intelligent answer!"
)

iface.launch()
