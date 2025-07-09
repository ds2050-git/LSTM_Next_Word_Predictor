
# LSTM Next Word Predictor

## Overview
A next-word prediction tool trained on Shakespeare's Hamlet using a Long Short-Term Memory (LSTM) neural network implemented with TensorFlow and Keras. The model learns contextual patterns and predicts the most probable next word based on a given phrase. Includes a Streamlit web app for interactive prediction.

## Dependencies
- Python 3.8+
- Required packages (listed in `requirements.txt`):
  ```
  tensorflow==2.17.0
  pandas 
  numpy
  scikit-learn
  tensorboard
  matplotlib
  streamlit
  ipykernel
  scikeras
  nltk
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ds2050-git/LSTM_Next_Word_Predictor.git
   cd LSTM_Next_Word_Predictor
   ```

2. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

- **Train the Model**:  
  Run `app.py` to train the LSTM model and generate:
  - `next_word_generation_lstm_rnn.h5` (saved model)
  - `tokenizer.pkl` (tokenizer for text preprocessing)

  ```bash
  python app.py
  ```

- **Test Predictions**:  
  Run `Prediction.py` to test next-word prediction using the saved model and tokenizer.
  ```bash
  python Prediction.py
  ```

- **Launch Streamlit App**:  
  Start the interactive web app for real-time next word prediction.
  ```bash
  streamlit run Streamlit_App.py
  ```

  Open your browser and enter a phrase like:
  ```
  To be or not to
  ```
  to get the predicted next word.

