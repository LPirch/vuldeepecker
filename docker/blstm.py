from __future__ import print_function

import warnings
from pathlib import Path
import json

from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_curve
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax

from sklearn.model_selection import train_test_split

"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""
class BLSTM:
    n_epochs = 4
    lr = 0.002
    hidden_dim = 300
    dropout = 0.5
    batch_size = 64

    def __init__(self, X_train, y_train, X_test, y_test, result_dir: Path, name="", subsample_train=1.0, seed=0):
        self.X_train = np.stack(X_train)
        self.X_test = np.stack(X_test)
        self.y_train = y_train
        self.y_test = y_test

        if subsample_train < 1.0:
            self.X_train, _, self.y_train, _ = train_test_split(self.X_train, self.y_train, test_size=1 - subsample_train, random_state=seed)

        self.name = name
        self.result_dir = result_dir
        self.weights_file = result_dir / (name + ".weights.h5")
        self.class_weight = {i: weight for i, weight in enumerate(compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=self.y_train))}
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        model = Sequential()
        model.add(Bidirectional(LSTM(self.hidden_dim), input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(self.hidden_dim))
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout))
        model.add(Dense(self.hidden_dim))
        model.add(LeakyReLU())
        model.add(Dropout(self.dropout))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(learning_rate=self.lr)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.n_epochs, class_weight=self.class_weight)
        self.model.save_weights(self.weights_file)

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        self.model.load_weights(self.weights_file)
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()
        pred_scores = (predictions[:,1]-predictions[:,0])/2 + 0.5
        report = get_report(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1))
        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        precision = 0 if np.isnan(precision) else precision
        f1_score = (2 * precision * recall) / (precision + recall)
        f1_score = 0 if np.isnan(f1_score) else f1_score
        pr, rec, ts = precision_recall_curve(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1))
        auprec = auc(rec, pr)
        vd_score, vd_threshold = calculate_vul_det_score(pred_scores, np.argmax(self.y_test, axis=1))
        vd_threshold = "nan" if np.isnan(vd_threshold) else vd_threshold
        vd_threshold = vd_threshold.item() if isinstance(vd_threshold, np.generic) else vd_threshold
        scores = {
            "accuracy": values[1],
            "false_positive_rate": fp / (fp + tn),
            "false_negative_rate": fn / (fn + tp),
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "auprec": auprec,
            "report": report,
            "vds": vd_score,
            "vds_threshold": vd_threshold,
        }
        with open(self.result_dir / (self.name + ".json"), "w") as f:
            json.dump(scores, f, indent=4)
        with open(self.result_dir / f"{self.name}-preds.csv", "w") as f:
            for pred, target in zip(predictions, np.argmax(self.y_test, axis=1)):
                score = pred[1] - pred[0]
                f.write(f"{score.item()},{target.item()}\n")


def get_report(target, preds):
    report = classification_report(target, preds, output_dict=True)
    return _serialize(report)


def _serialize(d):
    for k, v in d.items():
        if isinstance(v, np.generic):
            d[k] = v.item()
        elif isinstance(v, dict):
            d[k] = _serialize(v)
    return d


def calculate_vul_det_score(predictions, ground_truth, target_fpr=0.005):
    """
    Calculate the vulnerability detection score (VD-S) given a tolerable FPR.

    Args:
    - predictions: List of model prediction probabilities for the positive class.
    - ground_truth: List of ground truth labels, where 1 means vulnerable class, and 0 means benign class.
    - target_fpr: The tolerable false positive rate.

    Returns:
    - vds: Calculated vulnerability detection score given the acceptable .
    - threshold: The classification threashold for vulnerable prediction.
    """

    # Calculate FPR, TPR, and thresholds using ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predictions)

    # Filter thresholds where FPR is less than or equal to the target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]

    # Choose the threshold with the largest FPR that is still below the target FPR, if possible
    if len(valid_indices) > 0:
        idx = valid_indices[-1]  # Last index where FPR is below or equal to target FPR
    else:
        # If no such threshold exists (unlikely), default to the closest to the target FPR
        idx = np.abs(fpr - target_fpr).argmin()

    chosen_threshold = thresholds[idx]

    # Classify predictions based on the chosen threshold
    classified_preds = [1 if pred >= chosen_threshold else 0 for pred in predictions]

    # Calculate VD-S
    fn = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 0])
    tp = sum([1 for i in range(len(ground_truth)) if ground_truth[i] == 1 and classified_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0

    return vds, chosen_threshold
