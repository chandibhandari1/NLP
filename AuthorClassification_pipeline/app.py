import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, flash, jsonify, render_template, make_response
from io import StringIO


DEBUG = True
app = Flask(__name__)

model = pickle.load(open('model.pickle', 'rb'))
tfidf = pickle.load(open('tfidf.pickle', 'rb'))
print('model info: ', model)
labels = ['EAP','HPL', 'MWS']



@app.route('/note', methods=['GET'])
def index():
    return 'Hello, World'


@app.route('/predict_author', methods=['POST'])
def predict_author():
    print("Prediction Started")
    req = request.get_json()
    texts = req['text']
    feature =tfidf.transform([texts]).toarray()
    prediction = model.predict(feature)
    print(prediction)
    print('Predicted Author:',labels[prediction[0]])
    return jsonify({"Given Text": texts, "Predicted Author": labels[prediction[0]]})


if __name__ == '__main__':
    print(("* Loading model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(threaded=True, debug=True)
