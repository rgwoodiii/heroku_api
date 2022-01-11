# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.metrics import precision_score, recall_score, fbeta_score
from joblib import load


df = pd.read_csv("../data/census_cleaned.csv")


def train_test_model(data=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # load data
    if data is None:
        df = pd.read_csv("../data/census_cleaned.csv")
    else:
        df = data
    # separate target
    x = df.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = df['salary']
    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    # model
    model = RandomForestClassifier(n_estimators=100)
    # fit
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))


def save_model(data=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # load data
    if data is None:
        df = pd.read_csv("../data/census_cleaned.csv")
    else:
        df = data
    # separate target
    x = df.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = df['salary']

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    # model
    model = RandomForestClassifier(n_estimators=100)

    # fit
    model.fit(x_train, y_train)

    # save model
    dump(model, "trainedmodel.pkl")


def inference(df):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # load model
    model = load("trainedmodel.pkl")

    # prep data
    x = df.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = df['salary']

    # predict
    pred = model.predict(x)
    return pred, y


def compute_model_metrics(pred, y_test):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    pred
    y

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y_test, pred, average='weighted', beta=0.5)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average=None)

    return precision, recall, fbeta


if __name__ == "__main__":
    train_test_model(df)
    pred, y_test = inference(df)
    compute_model_metrics(pred, y_test)
