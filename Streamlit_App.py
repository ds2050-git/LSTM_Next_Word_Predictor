import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_generation_lstm_rnn.h5', compile=False)

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

st.title("Next Word Prediction Using LSTM RNN")

input_text = st.text_input("Enter a sequence of words:", "To be or not to")
if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter a valid word sequence.")
    else:
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f'Next word: **{next_word}**')
