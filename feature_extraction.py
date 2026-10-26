"""Cut continuous EEG into one segment per row/column flash."""

import numpy as np


class FeatureExtractor:
    """Epoch a recording into ``(segments, channels, samples)``.

    Each call returns ``self`` so the steps can be chained. The segmented
    array is stored on ``data`` and the P300 labels on ``labels``.
    """

    def __init__(
        self,
        signal,
        epochs,
        codes,
        fs,
        window,
        delta,
        number_of_flashes,
        number_of_trials,
    ):
        self.window = int(window * fs)
        self.delta = int(delta * fs)
        number_of_channels = signal.shape[-1]
        self.signal = signal
        self.epochs = epochs
        self.codes = codes
        self.number_of_flashes = number_of_flashes
        self.number_of_trials = number_of_trials
        self.number_of_epochs = epochs[-1]
        self.data = np.empty(
            (
                self.number_of_epochs,
                number_of_flashes,
                number_of_trials,
                self.window,
                number_of_channels,
            ),
            dtype=self.signal.dtype,
        )
        self.rowcol_code = np.tile(
            np.arange(1, number_of_flashes + 1, dtype=np.int8),
            (self.number_of_epochs, 1),
        )
        self.labels = None

    def segment_flashes(self):
        """Copy the post-stimulus window at every flash onset.

        The array layout is
        ``(epoch, flash, trial, time, channel)``.
        """
        # Onset: stimulus code becomes larger than the sample before it.
        start_positions = np.argwhere(np.diff(np.append(0, self.codes)) > 0).ravel()
        tracker = np.zeros(
            (self.number_of_epochs, self.number_of_flashes), dtype=np.int8
        )

        for onset in start_positions:
            current_epoch = self.epochs[onset] - 1
            row_col = self.codes[onset] - 1
            trial = tracker[current_epoch][row_col]
            tracker[current_epoch][row_col] += 1
            self.data[current_epoch, row_col, trial] = self.signal[
                (onset + self.delta) : (onset + self.delta + self.window)
            ]
        return self

    def wavelet_kernels(self):
        import pywt

        wavelet = "cmor"
        coeffs = pywt.wavedec(self.data, wavelet, level=4)
        self.data = np.abs(coeffs)

    def channels_before_time(self):
        """Move channels in front of the time axis."""
        self.data = np.swapaxes(self.data, -1, -2)
        return self

    def label_targets(self, target_character, character_index):
        """Mark flashes whose code is the target row or the target column."""
        target_character_index = np.array(
            [character_index[character] for character in target_character]
        )
        self.labels = (
            (self.rowcol_code[:, :, None] == target_character_index[:, None, :])
            .any(-1)
            .astype(np.int8)
            .ravel()
        )
        return self

    def repeat_labels_over_trials(self):
        self.labels = np.repeat(self.labels, self.trials_per_epoch)
        return self

    def keep_trials(self, n):
        """Keep the first ``n`` repetitions of each character."""
        self.data = self.data[:, :, :n]
        self.trials_per_epoch = n
        return self

    def _average_groups(self, array, axis, group_size):
        shape = array.shape
        grouped_shape = list(shape)
        grouped_shape[axis] = shape[axis] // group_size
        grouped_shape.insert(axis + 1, group_size)
        grouped = np.reshape(array, grouped_shape)
        return grouped.mean(axis=axis + 1)

    def average_trials(self, n=None):
        if n is None:
            n = self.data.shape[2]
        elif (self.data.shape[2] % n) != 0:
            raise Exception("Incompatible n")

        self.data = self._average_groups(self.data, axis=2, group_size=n)
        self.trials_per_epoch = self.trials_per_epoch // n
        return self

    def stack_flashes_and_trials(self):
        """``(epoch, flash, trial, ...)`` to ``(epoch, flash * trial, ...)``."""
        shape = self.data.shape
        self.data = self.data.reshape([shape[0], -1, *shape[3:]])
        return self

    def stack_epochs(self):
        """``(epoch, segment, ...)`` to ``(epoch * segment, ...)``."""
        shape = self.data.shape
        self.data = self.data.reshape([-1, *shape[2:]])
        return self

    def flatten_time_and_channels(self):
        shape = self.data.shape
        self.data = self.data.reshape(shape[:-2] + (-1,))
        return self

    def downsample(self, factor=2):
        self.data = self.data[..., ::factor]
        return self
