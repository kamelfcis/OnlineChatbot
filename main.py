from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
from flask_cors import CORS
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import load_model

class CustomSGD(SGD):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, **kwargs):
        # Call the parent class constructor
        super(CustomSGD, self).__init__(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov, **kwargs)

    def get_updates(self, loss, params):
        # Here you can implement any custom behavior for the optimizer.
        # For now, we'll use the standard SGD behavior.
        return super(CustomSGD, self).get_updates(loss, params)

    def get_config(self):
        # To save your custom optimizer's parameters, you need to implement get_config.
        config = super(CustomSGD, self).get_config()
        config.update({
            'custom_param': self.custom_param,  # Replace with any custom parameters
        })
        return config

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
CORS(app)
with open('questions.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5', custom_objects={'Custom>SGD': CustomSGD})

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({"response": res})

