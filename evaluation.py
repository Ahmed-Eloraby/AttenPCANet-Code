"""P300 detection metrics and character spelling accuracy."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:
    def test_p300(self, output, targets):
        pass

    def test_character(
        self,
        output,
        test_characters,
        number_of_trials,
        number_of_columns,
        number_of_flashes,
        character_matrix,
    ):
        pass


class SingleScoreEvaluator(Evaluator):
    """Evaluate a one-dimensional P300 score with a 0.5 threshold."""

    def test_p300(self, output, targets):
        predictions = [1 if score > 0.5 else 0 for score in output]
        self.p300_acc = accuracy_score(targets, predictions) * 100
        self.p300_f1_score = f1_score(targets, predictions) * 100
        self.p300_f1_macro = f1_score(targets, predictions, average="macro") * 100
        print(f"Macro F1 Score: {self.p300_f1_macro:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(targets, predictions))
        print(classification_report(targets, predictions))
        print("________________________________________")
        return self.p300_f1_macro

    def test_character(
        self,
        output,
        test_characters,
        number_of_trials,
        number_of_columns,
        number_of_flashes,
        character_matrix,
    ):
        test_outputs = np.reshape(
            output,
            (len(test_characters), number_of_flashes, number_of_trials),
        )
        test_outputs = np.sum(test_outputs, axis=-1)

        rows = np.argmax(test_outputs[:, number_of_columns:], axis=1)
        cols = np.argmax(test_outputs[:, :number_of_columns], axis=1)
        predicted_characters = character_matrix[rows, cols]
        print(list(zip(predicted_characters, test_characters)))

        self.character_accuracy = (
            accuracy_score(test_characters, predicted_characters) * 100
        )
        print(f"Character Accuracy: {self.character_accuracy:.2f}%")
        return self.character_accuracy


class TwoClassEvaluator(Evaluator):
    """Evaluate the two class scores produced by AttenPCA-Net.

    Character decisions use the P300-class score (column 1). Scores for the
    same row, and the same column, are summed across repetitions. The predicted
    character is the intersection of the winning row and the winning column.
    """

    def test_p300(self, output, targets):
        print("TwoClassEvaluator.test_p300")
        print(output.shape)
        predictions = np.argmax(output, axis=1)
        print(output.shape)
        self.p300_acc = accuracy_score(targets, predictions) * 100
        self.p300_f1_score = f1_score(targets, predictions) * 100
        self.p300_f1_macro = f1_score(targets, predictions, average="macro") * 100
        self.p300_precision = precision_score(targets, predictions) * 100
        self.p300_recall = recall_score(targets, predictions) * 100
        self.p300_roc_auc = roc_auc_score(targets, output[:, 1]) * 100

        is_p300 = targets == 1
        output_p300 = output[is_p300, 1]
        output_p300 = output_p300 / np.max(np.abs(output_p300))
        self.p300_diff = np.sum(np.abs(1 - output_p300))
        output_non_p300 = output[~is_p300, 0]
        output_non_p300 = output_non_p300 / np.max(np.abs(output_non_p300))
        self.non_p300_diff = np.sum(np.abs(1 - output_non_p300))
        self.total_diff = self.p300_diff + self.non_p300_diff

        print(f"self.p300_acc: {self.p300_acc:.2f}%")
        print(f"self.p300_f1_score: {self.p300_f1_score:.2f}%")
        print(f"self.p300_f1_macro: {self.p300_f1_macro:.2f}%")
        print(f"self.p300_precision: {self.p300_precision:.2f}%")
        print(f"self.p300_recall: {self.p300_recall:.2f}%")
        print(f"self.p300_roc_auc: {self.p300_roc_auc:.2f}%")
        print(f"self.p300_diff: {self.p300_diff:.4f}")
        print(f"self.non_p300_diff: {self.non_p300_diff:.4f}")
        print(f"self.total_diff: {self.total_diff:.4f}")

        print(f"Macro F1 Score: {self.p300_f1_macro:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(targets, predictions))
        print(classification_report(targets, predictions))
        print("________________________________________")
        return self.p300_f1_macro

    def test_character(
        self,
        output,
        test_characters,
        number_of_trials,
        number_of_columns,
        number_of_flashes,
        character_matrix,
        normalize=False,
    ):
        if normalize:
            output = output / output.sum(axis=1, keepdims=True)

        output = output[:, 1]
        test_outputs = np.reshape(
            output,
            (len(test_characters), number_of_flashes, number_of_trials),
        )
        test_outputs = np.sum(test_outputs, axis=-1)

        rows = np.argmax(test_outputs[:, number_of_columns:], axis=1)
        cols = np.argmax(test_outputs[:, :number_of_columns], axis=1)
        predicted_characters = character_matrix[rows, cols]
        print(
            [
                (str(predicted), str(actual))
                for predicted, actual in zip(predicted_characters, test_characters)
            ]
        )

        self.character_accuracy = (
            classification_report(
                test_characters, predicted_characters, output_dict=True
            )["accuracy"]
            * 100
        )
        print("Character Accuracy")
        print(self.character_accuracy)
        return self.character_accuracy


if __name__ == "__main__":
    evaluator = TwoClassEvaluator()
    targets = np.random.randint(0, 2, size=1000)
    print(targets.shape)
    outputs = np.random.rand(1000, 2)
    print(outputs.shape)
    evaluator.test_p300(outputs, targets)
