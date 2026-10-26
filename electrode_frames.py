"""Frame-wise 10-20 grid.

Each time sample becomes one frame of shape ``(time, rows, columns)``. This
is the earlier mapping used to export ViViT clips. AttenPCA-Net training uses
``electrode_grid.py`` instead.
"""

import os

import numpy as np
import torch
from scipy.interpolate import interp1d


_GRID_LAYOUT = np.array(
    [
        [None, None, None, None, None, None, None, None, None, None, None],
        [None, None, None, None, "Fp1", "Fpz", "Fp2", None, None, None, None],
        [None, None, None, "AF7", "AF3", "AFz", "AF4", "AF8", None, None, None],
        ["F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10"],
        [
            "FT9",
            "FT7",
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "FT8",
            "FT10",
        ],
        ["T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10"],
        [
            "TP9",
            "TP7",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "TP8",
            "TP10",
        ],
        ["P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10"],
        [None, None, None, "PO7", "PO3", "POz", "PO4", "PO8", None, None, None],
        [None, None, None, None, "O1", "Oz", "O2", None, None, None, None],
        [None, None, None, None, None, "Iz", None, None, None, None, None],
    ]
)


class FrameGrid:
    def __init__(self, channels=None):
        self.channels = channels
        self.indices_mapping, self.grid_dimensions = self._index_mapping()

    def place(self, samples, channels=None):
        """Return ``(time, rows, columns)`` with each electrode at its site."""
        if channels is None:
            channels = self.channels

        grid = np.zeros((samples.shape[-1], *self.grid_dimensions))
        for sample, channel in zip(samples, channels):
            row, col = self.indices_mapping.get(channel)
            grid[:, row, col] = sample
        return grid

    def _index_mapping(self):
        channel_indices = {}
        for row_index, row in enumerate(_GRID_LAYOUT):
            for col_index, channel in enumerate(row):
                if channel is not None:
                    channel_indices[channel] = (row_index, col_index)
        return channel_indices, _GRID_LAYOUT.shape

    def remove_unneeded_channels(self, matrix, channels):
        matrix = np.array(matrix)
        valid_mask = np.isin(matrix, channels)
        matrix = matrix[np.any(valid_mask, axis=1)]
        matrix = matrix[:, np.any(valid_mask, axis=0)]
        return matrix

    def remove_zero_rows_and_columns(self, matrix):
        matrix = np.array(matrix)
        matrix = matrix[~np.all(matrix == 0, axis=1)]
        matrix = matrix[:, ~np.all(matrix == 0, axis=0)]
        return matrix

    def resample_video(self, video_array, target_frames=32):
        print(video_array.shape)
        num_frames, channels, height, width = video_array.shape
        original_indices = np.linspace(0, num_frames - 1, num=num_frames)
        target_indices = np.linspace(0, num_frames - 1, num=target_frames)
        resampled_video = np.zeros(
            (target_frames, channels, height, width), dtype=video_array.dtype
        )
        for channel in range(channels):
            for row in range(height):
                for col in range(width):
                    pixel_values = video_array[:, channel, row, col]
                    interpolator = interp1d(
                        original_indices,
                        pixel_values,
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    resampled_video[:, channel, row, col] = interpolator(
                        target_indices
                    )
        return resampled_video


class VideoSampleExporter:
    """Export topographic frames as tensors for a pretrained ViViT."""

    def __init__(self, grid: FrameGrid):
        from transformers import VivitImageProcessor

        self.grid = grid
        self.processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400"
        )

    def _scale_to_uint8(self, sample):
        scaled = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        return (scaled * 255).astype(np.uint8)

    def export(self, samples, labels, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "labels"), labels)
        for index, sample in enumerate(samples):
            placed = self.grid.place(sample)
            video_sample = self.grid.resample_video(
                np.expand_dims(placed, axis=-1), 32
            )
            video_sample = np.repeat(video_sample, 3, axis=-1)
            video_sample = self._scale_to_uint8(video_sample)
            frames = []
            for frame in video_sample:
                pixel_values = self.processor(frame, return_tensors="pt")[
                    "pixel_values"
                ]
                frames.append(np.squeeze(pixel_values))
            np.save(
                os.path.join(output_dir, f"sample_{index}.npy"),
                np.array(frames, dtype=np.uint8),
            )


class DiskDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(
            name
            for name in os.listdir(data_dir)
            if name.endswith(".npy") and not name.startswith("labels")
        )
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = np.load(os.path.join(self.data_dir, self.files[index]))
        label = self.labels[index]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )
