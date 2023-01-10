import argparse
import pickle
import string
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("maxent_ne_chunker")
nltk.download("wordnet")
import warnings
warnings.filterwarnings('always')


rm = set(stopwords.words("english"))


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
    df = df.drop(['child_alone'],axis=1)
	
    X = df[['message']]
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
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    text = word_tokenize(text)
    text = list(set(text) - rm)
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text


text_transformer = Pipeline(
    [("vecttext", CountVectorizer(tokenizer=tokenize)), ("tfidf", TfidfTransformer())]
)

preprocessor = ColumnTransformer(
    [("text", text_transformer, "message")], remainder="passthrough"
)


def build_model():
    """
    Creates a pipeline for model training including a GridSearchCV object.
    
    Returns:
    cv: GridSearchCV object
    """
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("clf", RandomForestClassifier())]
    )

    parameters = [
        {
            "clf": [RandomForestClassifier()],
            "clf__n_estimators": [5, 50, 100, 250],
            "clf__max_depth": [5, 8, 10],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
            "clf__random_state": [42],
        },
        {
            "clf": [MultiOutputClassifier(LinearSVC(max_iter=10000))],
            "clf__estimator__C": [1.0, 10.0, 100.0, 1000.0],
            "clf__estimator__max_iter": [5000],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
            "clf__estimator__random_state": [42],
        },
        {
            "clf": [MultiOutputClassifier(MultinomialNB())],
            "preprocessor__text__vecttext__max_df": [0.5, 0.75, 1.0],
            "preprocessor__text__vecttext__ngram_range": [(1, 1), (1, 2)],
        },
    ]

    cv = GridSearchCV(
        pipeline,
        parameters,
        cv=2,
        scoring='f1_micro',
        n_jobs=-1,
    )

    return cv


def evaluate_model(model, X, Y):
    """
    Prints the results of the GridSearchCV function. Predicts a test set and prints a classification report.
    
    Arguments:
    model: trained sci-kit learn estimator
    X: feature data frame for test set evaluation
    Y: target data frame for test set evaluation
    """

    df = pd.DataFrame.from_dict(model.cv_results_)
    print("##### Cross-validation results on validation set #####")
    print("Best score:{}".format(model.best_score_))
    print("Best parameters set:{}".format(model.best_estimator_.get_params()["clf"]))
    print("mean_test_f1_micro: {}".format(df["mean_test_f1_micro"]))
    print("##### Scoring on test set #####")
    preds = model.predict(X)
    print(
        "Test set classification report: {}".format(
            classification_report(Y, preds, target_names=list(Y.columns))
        )
    )


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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=45)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
