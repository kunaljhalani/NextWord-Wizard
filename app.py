import pandas as pd
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model_rnn.h5')
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Load your dataset here or preprocess it as needed
df = pd.read_csv('tmdb_5000_movies.csv')
df = df['original_title']
movie_name = df.to_list()
tokenizer.fit_on_texts(movie_name)

vocab_array = np.array(list(tokenizer.word_index.keys()))

@app.route('/')
def home():
    return render_template('index.html', prediction="")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    input_number = request.form['input_number']

    no = int(input_number)
    # Make predictions
    text = input_text
    for i in range(no):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        text += " " + prediction

    return render_template('index.html', prediction=text)

if __name__ == '__main__':
    app.run(debug=True)
