"""Temporal PCA used before the topographic mapping.

One PCA is fit on the time axis of each EEG channel. Components are ordered
by explained variance, so ``transformed[..., :k]`` keeps the first ``k``
components of every channel. This is the reduction described for AttenPCA-Net:
a separate temporal PCA for each electrode, fit on the training segments only.
"""

import numpy as np
from sklearn.decomposition import PCA


class PCA_Dimensionality_Reduction:
    """Per-channel PCA along the last axis of ``(segments, channels, time)``."""

    def __init__(self, n_components):
        self.n_components = n_components
        self.models = []

    def fit(self, data):
        if data.ndim != 3:
            raise ValueError(
                "expected data shaped (segments, channels, time), "
                f"got {data.shape}"
            )
        self.models = []
        for channel in range(data.shape[1]):
            model = PCA(n_components=self.n_components)
            model.fit(data[:, channel, :])
            self.models.append(model)
        return self

    def transform(self, data):
        if not self.models:
            raise RuntimeError("call fit before transform")
        if data.shape[1] != len(self.models):
            raise ValueError(
                f"data has {data.shape[1]} channels, but {len(self.models)} "
                "PCA models were fit"
            )
        transformed = [
            model.transform(data[:, channel, :])
            for channel, model in enumerate(self.models)
        ]
        return np.stack(transformed, axis=1)
