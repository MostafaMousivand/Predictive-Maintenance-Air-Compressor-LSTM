#4
import pandas as pd
from sklearn.preprocessing import StandardScaler
from failure_intervals import *
import numpy as np

failure_events = failure_event()
TEST_BEFORE = pd.Timedelta(days=4, hours=1)
TEST_AFTER = pd.Timedelta(days=4, hours=1)
GAP = pd.Timedelta(hours = 1)

def fold(df_10min):
    X = df_10min.drop(columns = "failure_max").copy()
    y = df_10min["failure_max"].astype(int)
    X = X.sort_index()
    y = y.loc[X.index]
    test_events = failure_events[1:]
    folds = []
    for event in test_events:
        test_start = event["start"] - TEST_BEFORE
        test_end = event["end"] + TEST_AFTER
        test_mask = ((X.index >= test_start) & (X.index <= test_end))
        train_mask = X.index < (test_start - GAP)
        folds.append({"failure_id": event["failure_id"],"train_mask": train_mask,"test_mask": test_mask})
    return (folds, X, y)

""" def prepare_fold_data(X, y, folds):

    # 1. دریافت اطلاعات همین Fold
    failure_id = folds["failure_id"]
    train_mask = folds["train_mask"]
    test_mask = folds["test_mask"]
    
    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()

    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_scaled = scaler.transform(X).astype(np.float32)

    return (X_scaled,y_train,y_test,X_train,X_test,scaler,failure_id) """
