import pandas as pd
from sklearn.preprocessing import StandardScaler
from failure_intervals import *
import numpy as np

SEQ_LEN = 12

def build_sequences(X_scaled, X, y, X_train, X_test):
    X_seq = np.array([X_scaled[i - SEQ_LEN + 1 : i + 1]for i in range(SEQ_LEN - 1, len(X_scaled))],dtype=np.float32)

    y_seq = y.iloc[SEQ_LEN - 1:].to_numpy(dtype=np.int32)

    sequence_times = X.index[SEQ_LEN - 1:]

    train_seq_mask = sequence_times.isin(X_train.index)
    test_seq_mask = sequence_times.isin(X_test.index)

    X_train_seq = X_seq[train_seq_mask]
    y_train_seq = y_seq[train_seq_mask]

    X_test_seq = X_seq[test_seq_mask]
    y_test_seq = y_seq[test_seq_mask]

    test_times = sequence_times[test_seq_mask]
    return (X_train_seq,y_train_seq,X_test_seq,y_test_seq,test_times)
