from sklearn.metrics import (classification_report,precision_score,recall_score,f1_score,roc_auc_score,
                             average_precision_score,confusion_matrix)

THRESHOLD = 0.5

def evaluate_model(model, X_test_seq, y_test_seq):
    y_prob = model.predict(X_test_seq, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)
    precision = precision_score(y_test_seq, y_pred, zero_division=0)
    recall = recall_score(y_test_seq, y_pred, zero_division=0)
    f1 = f1_score(y_test_seq, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_seq, y_prob)
    pr_auc = average_precision_score(y_test_seq, y_prob)
    cm = confusion_matrix(y_test_seq, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {"precision": precision,"recall": recall,"f1": f1,"roc_auc": roc_auc,"pr_auc": pr_auc,"TN": tn,"FP": fp,"FN": fn,"TP": tp}
    return y_prob, y_pred, metrics
