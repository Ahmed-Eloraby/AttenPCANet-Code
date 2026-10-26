"""Train AttenPCA-Net on BCI Competition III Dataset II.

The sweep follows the published experiment. For each subject the network is
trained once for every principal-component count, using all 15 repetitions.
It is then scored again at 15, 14, ..., 1 repetitions. Only the repetition
count used at test time changes; the fitted model is not retrained.
"""

import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch

from attenpca_net import AttenPCANetTrainer
from config import BenchmarkConfiguration as config
from dataset import DatasetLoader
from dimensionality_reduction import PCA_Dimensionality_Reduction
from electrode_grid import TopographicGrid
from evaluation import TwoClassEvaluator
from feature_extraction import FeatureExtractor
from preprocessing import MNEPreprocessor

SEED = 0
LOW_FREQ = 0.5
HIGH_FREQ = 25
# Kept in the results filename. They are not architecture arguments.
EXPANSION_LAYERS = 2
CONV_LAYERS = 4
WINDOW_SECONDS = 0.6
ONSET_OFFSET_SECONDS = 0
PCA_CACHE_DIR = r"C:\Users\CSE\Desktop\EMBC Experiments\Template\Template\PCA"
RESULT_COLUMNS = [
    "n_pca_components",
    "p300_f1_macro",
    "p300_f1_score",
    "p300_acc",
    "p300_recall",
    "p300_precision",
    "p300_roc_auc",
    "total_diff",
    "p300_diff",
    "non_p300_diff",
    "character_accuracy",
    "subject",
    "trial",
    "number_of_parameters",
]


def seed_numpy(seed):
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_segments(
    signal,
    recording,
    sampling_rate,
    n_trials,
    targets,
    channels_before_stack,
    window_seconds,
):
    """Return ``(segments, channels, time)`` and one P300 label per segment.

    Training moves channels in front of time before merging epochs. Testing
    merges epochs first. Both orders produce the same array.
    """
    extractor = (
        FeatureExtractor(
            signal=signal,
            epochs=recording["epochs"],
            codes=recording["codes"],
            fs=sampling_rate,
            window=window_seconds,
            delta=ONSET_OFFSET_SECONDS,
            number_of_flashes=config.NUMBER_OF_FLASHES,
            number_of_trials=config.NUMBER_OF_TRIALS,
        )
        .segment_flashes()
        .keep_trials(n_trials)
        .stack_flashes_and_trials()
    )
    if channels_before_stack:
        extractor.channels_before_time().stack_epochs()
    else:
        extractor.stack_epochs().channels_before_time()
    extractor.label_targets(
        target_character=targets,
        character_index=config.CHARACTER_INDEX,
    ).repeat_labels_over_trials()
    return extractor.data, extractor.labels


def cached_array(path, build):
    if os.path.exists(path):
        return np.load(path)
    array = build()
    np.save(path, array)
    return array


def to_topography(samples, grid):
    return np.array(
        [grid.place(sample, expand=False, switch=True) for sample in samples]
    )


def preprocess(signal, channel_names):
    processor = MNEPreprocessor(
        eeg_data=signal, sfreq=240, channel_names=channel_names
    )
    return (
        processor.apply_notch_filter()
        .apply_bandpass_filter(low=LOW_FREQ, high=HIGH_FREQ)
        .apply_thresholding()
        .apply_car_filter()
        .get_processed_data()
    )


def main():
    seed_numpy(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_file = (
        f"attenPCA_{LOW_FREQ}_{HIGH_FREQ}_loss_{EXPANSION_LAYERS}_"
        f"{CONV_LAYERS}_nodropout_{time.time_ns()}.csv"
    )
    seed_numpy(SEED)
    seed_torch(SEED)

    if not os.path.exists(output_file):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(output_file, index=False)

    for subject in ["A", "B"]:
        for window_seconds in [WINDOW_SECONDS]:
            loader = DatasetLoader(subject)
            train_data, test_data, sampling_rate = loader.load()
            channel_names = loader.channel_names

            print(train_data["signal"].shape)
            processed_train = preprocess(train_data["signal"], channel_names)
            print("preprocessed train", processed_train.shape)
            processed_test = preprocess(test_data["signal"], channel_names)

            x_train_full, y_train = extract_segments(
                processed_train,
                train_data,
                sampling_rate,
                n_trials=15,
                targets=train_data["targets"],
                channels_before_stack=True,
                window_seconds=window_seconds,
            )
            print("X Train Shape:", x_train_full.shape)
            print("Y Train Shape:", y_train.shape)

            pca_path = os.path.join(
                PCA_CACHE_DIR, f"PCA_{subject}_{LOW_FREQ}_{HIGH_FREQ}.pkl"
            )
            if os.path.exists(pca_path):
                print("There")
                with open(pca_path, "rb") as handle:
                    reducer = pickle.load(handle)
            else:
                print("Here")
                reducer = PCA_Dimensionality_Reduction(x_train_full.shape[-1])
                reducer.fit(x_train_full)
                with open(pca_path, "wb") as handle:
                    pickle.dump(reducer, handle)

            train_pca_path = os.path.join(
                PCA_CACHE_DIR,
                f"X_train_pca_{subject}_{LOW_FREQ}_{HIGH_FREQ}.npy",
            )
            if os.path.exists(train_pca_path):
                print("There")
                x_train = np.load(train_pca_path)
            else:
                print("here")
                x_train = reducer.transform(x_train_full.copy())
                np.save(train_pca_path, x_train)
            print("X Train Shape:", x_train_full.shape)
            print("Y Train Shape:", y_train.shape)

            for n_components in range(40, 0, -1):
                seed_numpy(SEED)
                seed_torch(SEED)

                x_train_components = x_train[..., :n_components].copy()
                grid = TopographicGrid(
                    loader.channel_names,
                    segment_size=x_train_components.shape[-1],
                    three_d=True,
                )
                x_train_components = to_topography(x_train_components, grid)

                model = AttenPCANetTrainer(
                    frames=x_train_components.shape[1],
                    rows=x_train_components.shape[2],
                    cols=x_train_components.shape[3],
                )
                model.train(x_train_components, y_train)

                for n_trials in [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
                    x_train_view, y_train_view = extract_segments(
                        processed_train,
                        train_data,
                        sampling_rate,
                        n_trials=n_trials,
                        targets=train_data["targets"],
                        channels_before_stack=True,
                        window_seconds=window_seconds,
                    )
                    x_test, y_test = extract_segments(
                        processed_test,
                        test_data,
                        sampling_rate,
                        n_trials=n_trials,
                        targets=test_data["targets"],
                        channels_before_stack=False,
                        window_seconds=window_seconds,
                    )
                    print("X Test Shape:", x_test.shape)
                    print("Y Test Shape:", y_test.shape)

                    x_train_view = cached_array(
                        os.path.join(
                            PCA_CACHE_DIR,
                            f"X_train_pca_{n_trials}_{subject}_{LOW_FREQ}_{HIGH_FREQ}.npy",
                        ),
                        lambda view=x_train_view: reducer.transform(view),
                    )
                    x_test = cached_array(
                        os.path.join(
                            PCA_CACHE_DIR,
                            f"X_test_pca_{n_trials}_{subject}_{LOW_FREQ}_{HIGH_FREQ}.npy",
                        ),
                        lambda view=x_test: reducer.transform(view),
                    )

                    x_train_view = to_topography(
                        x_train_view[..., :n_components].copy(), grid
                    )
                    x_test = to_topography(x_test[..., :n_components].copy(), grid)
                    print("X Train Shape:", x_train_components.shape)
                    print("X Test Shape:", x_test.shape)

                    evaluator = TwoClassEvaluator()
                    seed_numpy(SEED)
                    seed_torch(SEED)

                    train_scores = model.test(x_train_view)
                    evaluator.test_p300(train_scores, y_train_view)
                    evaluator.test_character(
                        train_scores,
                        train_data["targets"],
                        n_trials,
                        config.NUMBER_OF_COLS,
                        config.NUMBER_OF_FLASHES,
                        config.MATRIX,
                    )

                    test_scores = model.test(x_test)
                    print("Output:", test_scores)
                    evaluator.test_p300(test_scores, y_test)
                    evaluator.test_character(
                        test_scores,
                        test_data["targets"],
                        n_trials,
                        config.NUMBER_OF_COLS,
                        config.NUMBER_OF_FLASHES,
                        config.MATRIX,
                    )

                    row = {
                        "n_pca_components": n_components,
                        "p300_f1_macro": evaluator.p300_f1_macro,
                        "p300_f1_score": evaluator.p300_f1_score,
                        "p300_acc": evaluator.p300_acc,
                        "p300_recall": evaluator.p300_recall,
                        "p300_precision": evaluator.p300_precision,
                        "p300_roc_auc": evaluator.p300_roc_auc,
                        "total_diff": evaluator.total_diff,
                        "p300_diff": evaluator.p300_diff,
                        "non_p300_diff": evaluator.non_p300_diff,
                        "character_accuracy": evaluator.character_accuracy,
                        "subject": subject,
                        "trial": n_trials,
                        "number_of_parameters": model.calculate_model_parameters(),
                    }
                    pd.DataFrame([row]).to_csv(
                        output_file, mode="a", index=False, header=False
                    )


if __name__ == "__main__":
    main()
