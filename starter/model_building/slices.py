# load libraries
from training import compute_model_metrics  # , train_test_model, inference
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load data
df = pd.read_csv("../data/census_cleaned.csv")
report = []
# slice
for value in df.sex.unique():
    data_slice = df[df["sex"] == value]

    # split
    x = data_slice.drop(['salary'], axis=1)
    x = pd.get_dummies(x)
    y = data_slice['salary']

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    # model
    model = RandomForestClassifier(n_estimators=100)
    # fit
    model.fit(x_train, y_train)

    # pred
    pred = model.predict(x_test)

    precision, recall, fbeta = compute_model_metrics(pred, y_test)
    print("Value: {value}".format(value=value))
    print("Precision: {precision}".format(precision=precision))
    print("recall: {recall}".format(recall=recall))
    print("fbeta: {fbeta}".format(fbeta=fbeta))

    summ_dict = {
        "value": value,
        "precision": precision.mean(),
        "recall": recall.mean(),
        "fbeta": fbeta}
    report.append(summ_dict)

with open("slice_output.txt", "w") as report_file:
    for item in report:
        report_file.write(str(item))
