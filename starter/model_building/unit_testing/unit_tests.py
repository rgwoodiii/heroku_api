# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score, recall_score, fbeta_score
from joblib import load
import pytest
import os.path

# functions

# load data


@pytest.fixture()
def df():
    df = pd.read_csv(os.path.join(root_dir, "data", "census_no_spaces.csv"))
    return df

# data


def test_data_shape(df):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert df.shape == df.dropna().shape, "Dropping null changes shape."


def test_slice_averages(df):
    """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
    for cat_feat in df["workclass"].unique():
        avg_value = df[df["workclass"] == cat_feat]["hours-per-week"].mean()
        assert (
            49 > avg_value > 28
        ), f"For {cat_feat}, average of {avg_value} not between 40 and 28"

# model


def test_perf(df):
    # separate target
    x = df.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = df['salary']

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    # model
    model = load("trainedmodel.pkl")
    # score
    assert model.score(x_test, y_test) >= .8, f"score is lower than expected."


def test_inference(df):
    # separate target
    x = df.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = df['salary']
    # model
    model = load("trainedmodel.pkl")
    # predict
    pred = model.predict(x)

    print(model.score(x, y))

    assert pred.shape[0] == y.shape[0], f"number of predictions are different from expected."

# metric review


def test_compute_model_metrics(df):
    x = df.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = df['salary']
    # model
    model = load("trainedmodel.pkl")
    # predict
    pred = model.predict(x)

    fbeta = fbeta_score(y, pred, average='weighted', beta=0.5)
    precision = precision_score(y, pred, average=None)
    recall = recall_score(y, pred, average=None)
    assert fbeta >= .96, f"fbeta is lower than expected."
    assert precision.mean() >= .95, f"precision is lower than expected."
    assert recall.mean() >= .94, f"recall is lower than expected."
