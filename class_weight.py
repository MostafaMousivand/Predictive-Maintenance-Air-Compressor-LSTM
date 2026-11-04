
def calculate_class_weights(y_train_seq, failure_id):
    n_negative = int((y_train_seq == 0).sum())
    n_positive = int((y_train_seq == 1).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError(f"Training fold {failure_id} ""does not contain both classes.")
    class_weight = {0: 1.0,1: n_negative / n_positive}
    return class_weight
