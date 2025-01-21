from __future__ import print_function

import warnings
from pathlib import Path
import json

from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
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
    def __init__(self, X_train, y_train, X_test, y_test, result_dir: Path, name="", batch_size=64, subsample_train=1.0, seed=42):
        self.X_train = np.stack(X_train)
        self.X_test = np.stack(X_test)
        self.y_train = y_train
        self.y_test = y_test

        if subsample_train < 1.0:
            self.X_train, _, self.y_train, _ = train_test_split(self.X_train, self.y_train, test_size=1 - subsample_train, random_state=seed)

        self.name = name
        self.result_dir = result_dir
        self.weights_file = result_dir / (name + ".weights.h5")
        self.batch_size = batch_size
        self.class_weight = {i: weight for i, weight in enumerate(compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=self.y_train))}
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(learning_rate=0.002)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=4, class_weight=self.class_weight)
        self.model.save_weights(self.weights_file)

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        self.model.load_weights(self.weights_file)
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy is...", values[1])
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()
        report = classification_report(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1), output_dict=True)
        report = {key: val.item() if isinstance(val, np.ndarray) else val for key, val in report.items()}
        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print('False positive rate is...', fp / (fp + tn))
        print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True positive rate is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))
        scores = {
            "accuracy": values[1],
            "false_positive_rate": fp / (fp + tn),
            "false_negative_rate": fn / (fn + tp),
            "true_positive_rate": recall,
            "precision": precision,
            "f1_score": (2 * precision * recall) / (precision + recall),
            "report": report,
        }
        with open(self.result_dir / (self.name + ".json"), "w") as f:
            json.dump(scores, f, indent=4)
