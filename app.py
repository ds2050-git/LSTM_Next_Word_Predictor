import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,LSTM,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping



# Pre-processing
nltk.download('gutenberg')
from nltk.corpus import gutenberg
data=gutenberg.raw('shakespeare-hamlet.txt')

with open('Dataset.txt','w') as file:
    file.write(data)

with open('Dataset.txt','r') as file:
    text=file.read().lower()

t=Tokenizer()
t.fit_on_texts([text])
total_words=len(t.word_index)+1

input_sequences=[]
for line in text.split('\n'):
    token_list=t.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)


max_sequence_len=max([len(x) for x in input_sequences])

input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))


x=input_sequences[:,:-1]
y=input_sequences[:,-1]
y=tf.keras.utils.to_categorical(y,num_classes=total_words)


# Training
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

early_stopping=EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)

model=Sequential()
feature_dimensions=100
neurons=150
model.add(Embedding(total_words,feature_dimensions)) 
model.add(LSTM(neurons,return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation="softmax")) # Output Layer

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),verbose=1)



model.save("next_word_generation_lstm_rnn.h5")

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(t,f)
