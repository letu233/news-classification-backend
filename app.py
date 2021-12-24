from flask import Flask, render_template, request, jsonify
from newspaper import Article
from utils import denoise_text, tokenize, lemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import joblib
import pickle
from flask_cors import CORS, cross_origin
import json
import pandas as pd
import os

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST', 'GET'])
@cross_origin()
def index():
    if request.method == 'GET':
        url = request.values.get('link')
        #print(url)
        #url = data.get("link")
        text = Article(url, language='en')
        text.download()
        text.parse()
        print(text.text)
        text = denoise_text(text.text)
        text = tokenize(text)
        text = lemmatizer(text)
        text = [text]

        vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))
        X_test = vectorizer.transform(text)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_matrix = X_test.todense()
        feature_index = tfidf_matrix[0, :].nonzero()[1]
        tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
        s = dict(tfidf_scores)
        sortedDict = sorted(s.items(), key=lambda x: x[1], reverse=True)[:10]
        listKey = []
        for i in range(len(sortedDict)):
            listKey.append(sortedDict[i][0])
        jsonListKey = json.dumps(listKey)

        model = load("xgboost_model.joblib.dat")
        predict = model.predict_proba(X_test)
        pro_fake = format(predict[0][0]*100, '.2f')
        pro_real = format(predict[0][1]*100, '.2f')
        return jsonify({"fake": pro_fake, "real": pro_real, "url": url, "key": jsonListKey, "answer": '1'})
    else:
        return jsonify({"result": "ONLY WORKS FOR GET REQUEST"})

@app.route('/source', methods=['POST', 'GET'])
@cross_origin()
def predict_source():
    if request.method == 'GET':
        choice = request.values.get('choice')
        print(choice)
        filename = chosen(choice)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(dir_path, "result")
        directory = os.path.join(folder, filename)
        df = pd.read_pickle(directory)
        df['fake'] = df['fake'].apply(lambda x: x * 100)
        df['real'] = df['real'].apply(lambda x: x * 100)
        df['fake'] = df['fake'].round(2)
        df['real'] = df['real'].round(2)
        json = df.to_json(orient = 'records')
        return json


def chosen(i):
    switcher = {
        '1': 'predict_abcnews.pkl',
        '2': 'predict_cnbc.pkl',
        '3': 'predict_foxnews.pkl',
        '4': 'predict_guardian.pkl',
        '5': 'predict_reuters.pkl',
    }
    return switcher.get(i, "Invalid")

if __name__ == "__main__":
    app.run(debug=True)