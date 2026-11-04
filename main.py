from  load_data import *
from create_model import *
from failure_intervals import *
from evolution import *
from features_engineering import *
from preprocessing import *
from seq_gen import *
from class_weight import *
from evolution import *
from expriment import *

df = load_data()

failure_events = failure_event()
df = failure_intervals(failure_events,df)

df_10min = feature_engineer1(df)
df_10min = feature_engineer2(df_10min,sensor_cols)
print(df_10min["failure_max"].value_counts())
print(df_10min.shape)
folds, X, y = fold(df_10min)
print(df_10min["failure_max"].value_counts())
print(df_10min.shape)
""" X_scaled, y_train, y_test, X_train, X_test, scaler, failure_id = prepare_fold_data(X, y, folds)

X_train_seq,y_train_seq,X_test_seq,y_test_seq,test_times = build_sequences(X_scaled, X, y, X_train, X_test)
class_weight = calculate_class_weights(y_train_seq, failure_id)
model = build_lstm_model(48)
history = train_model(model, X_train_seq, y_train_seq, class_weight)
y_prob, y_pred, metrics = evaluate_model(model, X_test_seq, y_test_seq) 
lstm_models, lstm_predictions, lstm_results = run_experiment(folds, X, y) """


SEED = 42

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


SEQ_LEN = 12
THRESHOLD = 0.5


X = X.sort_index()
y = y.loc[X.index]


lstm_results = []
lstm_models = {}
lstm_predictions = {}

for fold in folds:

    failure_id = fold["failure_id"]

    print("\n" + "=" * 65)
    print(f"LSTM - Failure {failure_id}")
    print("=" * 65)

    train_mask = fold["train_mask"]
    test_mask = fold["test_mask"]

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    # --------------------------------------------------------
    # 4.1: Fit scaler on training data only
    # --------------------------------------------------------

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_scaled = scaler.transform(X).astype(np.float32)

    # --------------------------------------------------------
    # 4.2: Build chronological sequences
    # --------------------------------------------------------

  
    X_seq = np.array([X_scaled[i - SEQ_LEN + 1 : i + 1] for i in range(SEQ_LEN - 1, len(X_scaled))],dtype=np.float32)

    y_seq = y.iloc[SEQ_LEN - 1:].to_numpy(dtype=np.int32)

    sequence_times = X.index[SEQ_LEN - 1:]

    # Select sequences by their END timestamps.
    # Training sequences end in the training period.
    # Test sequences end in the test period.

    train_seq_mask = sequence_times.isin(X_train.index)
    test_seq_mask = sequence_times.isin(X_test.index)

    X_train_seq = X_seq[train_seq_mask]
    y_train_seq = y_seq[train_seq_mask]

    X_test_seq = X_seq[test_seq_mask]
    y_test_seq = y_seq[test_seq_mask]

    test_times = sequence_times[test_seq_mask]

    print("Training sequences:", X_train_seq.shape)

    print("Test sequences:", X_test_seq.shape)

    print("Training class counts:",pd.Series(y_train_seq).value_counts().to_dict())

    print("Test class counts:",pd.Series(y_test_seq).value_counts().to_dict())

    # --------------------------------------------------------
    # 4.3: Handle class imbalance
    # --------------------------------------------------------

    n_negative = int((y_train_seq == 0).sum())
    n_positive = int((y_train_seq == 1).sum())

    if n_positive == 0 or n_negative == 0:
        raise ValueError(f"Training fold {failure_id} does not contain both classes.")

    class_weight = {0: 1.0,1: n_negative / n_positive}

    print("Class weights:", class_weight)

    # --------------------------------------------------------
    # 4.4: Build LSTM model
    # --------------------------------------------------------

    model = Sequential([
        Input(shape=(SEQ_LEN, X.shape[1])),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=[tf.keras.metrics.Precision(name="precision"),
                                                                       tf.keras.metrics.Recall(name="recall")])

    # --------------------------------------------------------
    # 4.5: Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)

    # Validation uses the last 15% of training sequences.
    # shuffle=False preserves chronological order.

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.15,
        epochs=20,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=1
    )

    # --------------------------------------------------------
    # 4.6: Predict on test data
    # --------------------------------------------------------

    y_prob = model.predict(X_test_seq,verbose=0).ravel()

    y_pred = (y_prob >= THRESHOLD).astype(int)

    # --------------------------------------------------------
    # 4.7: Evaluate model
    # --------------------------------------------------------

    precision = precision_score(y_test_seq, y_pred, zero_division=0)

    recall = recall_score(y_test_seq, y_pred, zero_division=0)

    f1 = f1_score(y_test_seq, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test_seq, y_prob)

    pr_auc = average_precision_score(y_test_seq, y_prob)

    cm = confusion_matrix(y_test_seq, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    print("\nClassification Report:")
    print(classification_report(y_test_seq,y_pred,labels=[0, 1],target_names=["Healthy", "Failure"],zero_division=0,digits=4) )

    print("Confusion Matrix:")
    print(cm)

    print(f"Threshold: {THRESHOLD}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    # --------------------------------------------------------
    # 4.8: Save fold results
    # --------------------------------------------------------

    lstm_results.append({
        "failure_id": failure_id,
        "train_samples": len(X_train_seq),
        "test_samples": len(X_test_seq),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "best_epoch": len(history.history["loss"])
    })

    lstm_models[failure_id] = {
        "model": model,
        "scaler": scaler,
        "history": history
    }

    lstm_predictions[failure_id] = pd.DataFrame({
        "y_true": y_test_seq,
        "y_prob": y_prob,
        "y_pred": y_pred
    }, index=test_times)

# ------------------------------------------------------------
# STEP 5: Summary table
# ------------------------------------------------------------

lstm_results_df = pd.DataFrame(lstm_results)
lstm_results_df.to_excel(r"D:\python\Air_compressor_project\project\lstm_results.xlsx",index=False)

print(lstm_predictions)




print("\n" + "=" * 65)
print("LSTM - EXPANDING WINDOW SUMMARY")
print("=" * 65)
print(lstm_results_df.round(4).to_string(index=False))
