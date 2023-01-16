import warnings
import argparse
import pickle
import string
import sys

import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
from joblib import parallel_backend
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier

nltk.download(['punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger'])

warnings.filterwarnings('always')

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """
    Loads a pandas DataFrame from a sqlite database

    Args:
    database_filepath: path of the sqlite database
    Returns:
    X: features (data frame)
    Y: target categories (data frame)
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_categories_messages", engine)

    # As per the analysis, child_alone doesn't have any information, so lets drop it
    df = df.drop(['child_alone'], axis=1)

    X = df.message.values
    Y = df.iloc[:, 4:]

    return X, Y


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


def build_model():
    """
    Creates a pipeline for model training including a GridSearchCV object.

    Returns:
    cv: GridSearchCV object
    """
    # Define individual models
    rf_clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    sv_clf = MultiOutputClassifier(LinearSVC(random_state=42))
    mlp_clf = MultiOutputClassifier(MLPClassifier(random_state=42))

    # Create the final pipeline

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('classifier', rf_clf)
    ])

   # Initiaze the hyperparameters for each dictionary
    param1 = {}
    param1['features__text_pipeline__vect__ngram_range'] = ((1, 1), (1, 2))
    param1['classifier__estimator__n_estimators'] = [50, 100, 250]
    param1['classifier__estimator__max_depth'] = [5, 8, 10]
    param1['classifier'] = [rf_clf]

    param2 = {}
    param2['features__text_pipeline__vect__ngram_range'] = ((1, 1), (1, 2))
    param2['classifier__estimator__C'] = [0.1, 1, 10]
    param2['classifier__estimator__loss'] = ['squared_hinge', 'hinge']
    param2['classifier__estimator__max_iter'] = [30000]
    param2['classifier'] = [sv_clf]

    param3 = {}
    param3['features__text_pipeline__vect__ngram_range'] = ((1, 1), (1, 2))
    param3['classifier__estimator__hidden_layer_sizes'] = [
        (50, 50, 50), (100, 100, 100)]
    param3['classifier__estimator__solver'] = ['adam', 'sgd']
    param3['classifier__estimator__max_iter'] = [30000]
    param3['classifier'] = [mlp_clf]

    params = [param1, param2, param3]

    cv = GridSearchCV(pipeline, params, cv=3, n_jobs=-
                      1, scoring='f1_micro', verbose=2)
    return cv


def evaluate_model(model, X, Y):
    """
    Prints the results of the GridSearchCV function. Predicts a test set and prints a classification report.

    Arguments:
    model: trained sci-kit learn estimator
    X: feature data frame for test set evaluation
    Y: target data frame for test set evaluation
    """

    print("##### Cross-validation results on validation set #####")
    print("Best score: {}".format(model.best_score_))
    best_params = model.best_estimator_.get_params()
    print("Best parameters set: {}".format(best_params))

    print("##### Scoring on test set #####")
    preds = model.predict(X)
    report = classification_report(Y, preds, target_names=list(Y.columns))
    print(f"Test set classification report: {report}")


def save_model(model, model_filepath):
    """
    Saves model as a .pkl file.

    Arguments:
    model: trained sci-kit learn estimator to save
    model_filepath: destination for model save
    """

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=45)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
