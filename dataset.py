"""Loader for BCI Competition III Dataset II."""

import numpy as np
from scipy.io import loadmat


class DatasetLoader:
    """Read one subject's training and test recordings.

    Expected files, relative to ``parent``:

    - ``Benchmark dataset/Subject_{A|B}_Train.mat``
    - ``Benchmark dataset/Subject_{A|B}_test.mat``
    - ``Benchmark dataset/true_labels_{A|B}.txt``
    """

    def __init__(self, subject):
        assert subject in ["A", "B"]
        self.subject = subject

        self.channels = list(range(64))
        # Channel order of the 64 electrodes in the competition ``.mat`` files.
        self.channel_names = [
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "Fp1",
            "Fpz",
            "Fp2",
            "AF7",
            "AF3",
            "AFz",
            "AF4",
            "AF8",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FT8",
            "T7",
            "T8",
            "TP7",
            "TP8",
            "T9",
            "T10",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Iz",
        ]
        self.fs = 240

    def load(self, parent="."):
        train = loadmat(
            parent + r"/Benchmark dataset/Subject_" + self.subject + "_Train.mat"
        )

        train_eeg_samples = train["Signal"].reshape((-1, 64))[:, self.channels]
        train_epoch_number = np.repeat(
            np.arange(1, 1 + train["Signal"].shape[0]), train["Signal"].shape[1]
        )
        train_codes = train["StimulusCode"].astype("int8", copy=False).ravel()
        train_targets = np.array(list(train["TargetChar"][0]))

        train_data = {
            "signal": train_eeg_samples,
            "epochs": train_epoch_number,
            "codes": train_codes,
            "targets": train_targets,
        }

        test = loadmat(
            parent + r"/Benchmark dataset/Subject_" + self.subject + "_test.mat"
        )
        test_eeg_samples = test["Signal"].reshape((-1, 64))[:, self.channels]
        test_epoch_number = np.repeat(
            np.arange(1, 1 + test["Signal"].shape[0]), test["Signal"].shape[1]
        )
        test_codes = test["StimulusCode"].astype("int8", copy=False).ravel()
        labels_path = (
            parent + r"/Benchmark dataset/true_labels_" + self.subject + ".txt"
        )
        with open(labels_path, "r") as handle:
            test_targets = np.array(list(handle.read()))

        test_data = {
            "signal": test_eeg_samples,
            "epochs": test_epoch_number,
            "codes": test_codes,
            "targets": test_targets,
        }
        return train_data, test_data, self.fs
