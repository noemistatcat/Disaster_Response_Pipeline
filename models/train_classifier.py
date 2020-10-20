# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import re
import joblib
import nltk
import sys
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Loads the database from the ETL pipeline and outputs pandas dataframes containing features and targets
    :param database_filepath: file path of the database from ETL pipeline
    :return: X, Y: ETL pipeline database (features and targets)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from DisasterResponse', con=engine)
    X = df.iloc[:, 1]
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Applies pre-processing steps to the text data
    :param text: Text data for pre-processing: tokenization, normalization, stemming, lemmatization
    :return: lemmatized: Preprocessed text data
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Tokenize into words
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    words = [x for x in tokens if x not in stopwords.words("english")]
    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return lemmatized


def build_model():
    """
    Builds the model using Pipeline and GridSearchCV
    :return: GridSearchCV object
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),
                         ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Tests the model's performance using sklearn's classification report
    :param model: sklearn fitted model
    :param X_test: the X test set
    :param Y_test: the Y test classifications
    :param category_names: the category names
    :return: classification report: reports the model's precision, recall, f1-score, and support
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath='random_forest.pkl'):
    """
    Saves the model to the given path
    :param model: the fitted model
    :param model_filepath: the file path to save the model
    :return: none
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
