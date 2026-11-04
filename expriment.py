""" 
from create_model import build_lstm_model, train_model
from evolution import evaluate_model
from preprocessing import prepare_fold_data
from seq_gen import build_sequences
from class_weight import calculate_class_weights
import pandas as pd

def run_experiment(folds, X, y):
    lstm_results = []
    lstm_models = {}
    lstm_predictions = {}

    for fold in folds:
        failure_id = fold["failure_id"]

        # 1. Prepare data
        (X_scaled, y_train, y_test, X_train, X_test, scaler, failure_id) = prepare_fold_data(X, y, fold)

        # 2. Build sequences
        (X_train_seq,y_train_seq,X_test_seq,y_test_seq,test_times) = build_sequences(X_scaled, X, y, X_train, X_test)

        # 3. Calculate class weights
        class_weight = calculate_class_weights(y_train_seq, failure_id)

        # 4. Build model
        model = build_lstm_model(X.shape[1])

        # 5. Train model
        history = train_model(model,X_train_seq,y_train_seq,class_weight)

        # 6. Evaluate model
        y_prob, y_pred, metrics = evaluate_model(model,X_test_seq,y_test_seq)

        # 7. Save results
        lstm_results.append({"failure_id": failure_id,"train_samples": len(X_train_seq),
                             "test_samples": len(X_test_seq),**metrics,"best_epoch": len(history.history["loss"])})

        lstm_models[failure_id] = {"model": model,"scaler": scaler,"history": history}

        lstm_predictions[failure_id] = pd.DataFrame({"y_true": y_test_seq,"y_prob": y_prob,"y_pred": y_pred},index=test_times)

    return lstm_models, lstm_predictions, pd.DataFrame(lstm_results)
 """