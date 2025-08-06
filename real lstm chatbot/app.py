import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

# ✅ Training data
conversations = [
    ("hi", "hello"),
    ("hello", "hi"),
    ("how are you", "i am fine"),
    ("what is your name", "i am a bot"),
    ("who is the president", "joe biden"),
    ("where is he from", "usa"),
    ("bye", "see you"),
    ("what is ai", "artificial intelligence"),
    ("who made you", "humans"),
    ("how old are you", "i am new")
]

# ✅ Preprocessing
inputs, answers = zip(*conversations)
all_texts = list(inputs + answers)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
vocab_size = len(tokenizer.word_index) + 1

max_len = max(len(x.split()) for x in inputs)
X = pad_sequences(tokenizer.texts_to_sequences(inputs), maxlen=max_len)

# ✅ Predict only first word of output
y = [seq[0] if len(seq) > 0 else 0 for seq in tokenizer.texts_to_sequences(answers)]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# ✅ Build model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    LSTM(64),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=300, verbose=0)

# ✅ Save model
model.save("real.h5")

def chatbot(message, history=[]):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    word = tokenizer.index_word[np.argmax(pred)] if np.argmax(pred) in tokenizer.index_word else "[unknown]"
    history.append((message, word))
    return "\n".join([f"You: {msg}\nBot: {res}" for msg, res in history]), history

# ✅ Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🧐 Real-Time Chatbot with LSTM Memory")
    chatbot_state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(placeholder="Enter your message", label="")
        out = gr.Textbox(label="Chat")
    txt.submit(chatbot, [txt, chatbot_state], [out, chatbot_state])

app.launch()
