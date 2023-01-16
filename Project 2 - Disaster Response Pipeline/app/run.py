import json
import plotly
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
import numpy as np
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator,TransformerMixin



app = Flask(__name__)
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """
    Tokenizes input text
    
    Args:
    text: text data as str
    Returns:
    text: tokenized text
    """
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    This function will rxtract the part of verb and will create new feature for ML classifier
    """

    def starting_verb(self, text):
        """
        tokenize by sentences. It will tokenize each sentence into words and tag part of speech 
        and return true if the first word is an appropriate verb or RT for retweet
        INPUT : self and message
        OUTPUT : true and false based on approprite verb 
        
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False
        
    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        """
        returns self instance  which is needed for piprline
        """
        return self

    def transform(self, X):
        """
        applying starting_verb function to all values in X
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_categories_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(100*gen_count/gen_count.sum(), 2)
    gen = list(gen_count.index)
    cat_num = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum()
    cat_num = cat_num.sort_values(ascending=False)
    cat = list(cat_num.index)

    colors = ['yellow', 'green', 'red']

    # create visuals
    graphs = [
        {
            "data": [
                {
                    "type": "pie",
                    "uid": "f4de1f",
                    "hole": 0.4,
                    "name": "Genre",
                    "pull": 0,
                    "domain": {
                        "x": gen_per,
                        "y": gen
                    },
                    "marker": {
                        "colors": [
                            "#7fc97f",
                            "#beaed4",
                            "#fdc086"
                        ]
                    },
                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": gen,
                    "values": gen_count
                }
            ],
            "layout": {
                "title": "Count and Percent of Messages by Genre"
            }
        },
        {
            "data": [
                {
                    "type": "bar",
                    "x": cat,
                    "y": cat_num,
                    "marker": {
                        "color": 'grey'}
                }
            ],
            "layout": {
                "title": "Count of Messages by Category",
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
