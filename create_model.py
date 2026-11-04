#4
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import (classification_report,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM,Dense,Dropout,Input)
from tensorflow.keras.callbacks import EarlyStopping

SEQ_LEN = 12
THRESHOLD = 0.5
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

#Build LSTM model
def build_lstm_model(n_features):
    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")])

    model.compile(optimizer="adam",loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.Precision(name="precision"),tf.keras.metrics.Recall(name="recall")])
    return model

def train_model(model, X_train_seq, y_train_seq, class_weight):
    early_stopping = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

    history = model.fit(X_train_seq,y_train_seq,validation_split=0.15,epochs=20,batch_size=64,
        class_weight=class_weight,callbacks=[early_stopping],shuffle=False,verbose=1)

    return history




