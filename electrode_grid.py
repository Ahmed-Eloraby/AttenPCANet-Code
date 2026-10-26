"""Place each electrode's features on a 10-20 grid.

This is the mapping used by AttenPCA-Net. With ``three_d=True`` the grid is
``(rows, columns, features)`` and each electrode owns the vector at its
scalp location. ``to_model_layout`` moves that vector onto the channel axis
so the tensor is ``(features, rows, columns)``.
"""

import os

import numpy as np
import torch
from scipy.interpolate import griddata, interp1d
from scipy.spatial.distance import cdist


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


class TopographicGrid:
    def __init__(self, channels=None, segment_size=240, three_d=False):
        self.channels = channels
        self.segment_size = segment_size
        self.three_d = three_d
        self.indices_mapping, self.grid_dimensions = self._index_mapping(
            segment_size
        )

    def place(self, samples, channels=None, expand=True, switch=False):
        """Scatter one segment onto the 10-20 grid.

        ``samples`` is ``(channels, features)``. With ``switch=True`` the
        returned array is ``(features, rows, columns)``, which is the layout
        consumed by AttenPCA-Net.
        """
        if channels is None:
            channels = self.channels

        grid = np.zeros(self.grid_dimensions)
        for sample, channel in zip(samples, channels):
            row, col = self.indices_mapping.get(channel)
            if self.three_d:
                grid[row, col] = sample
            else:
                grid[row, col : col + self.segment_size] = sample

        if expand:
            grid = np.expand_dims(grid, 0)
        if switch:
            if expand:
                grid = np.transpose(grid, (0, 3, 1, 2))
            else:
                grid = np.transpose(grid, (2, 0, 1))
        return grid

    def _index_mapping(self, segment_size=240):
        grid_layout = _GRID_LAYOUT
        if self.three_d:
            grid_dimensions = (
                grid_layout.shape[0],
                grid_layout.shape[1],
                segment_size,
            )
        else:
            grid_dimensions = (
                grid_layout.shape[0],
                grid_layout.shape[1] * segment_size,
            )

        channel_indices = {}
        for row_index, row in enumerate(grid_layout):
            for col_index, channel in enumerate(row):
                if channel is not None:
                    if self.three_d:
                        channel_indices[channel] = (row_index, col_index)
                    else:
                        channel_indices[channel] = (
                            row_index,
                            col_index * segment_size,
                        )
        return channel_indices, grid_dimensions

    def remove_unneeded_channels(self, matrix, channels=None):
        if channels is None:
            channels = self.channels

        matrix = np.array(matrix)
        valid_mask = np.isin(matrix, channels)
        print(valid_mask)
        print(valid_mask.shape)
        matrix = matrix[np.any(valid_mask, axis=1)]
        print(matrix.shape)
        matrix = matrix[:, np.any(valid_mask, axis=0)]
        print(matrix.shape)
        return matrix

    # Neighbor offsets used by ``fill_missing`` on the compact 5-row montage.
    directions = [
        [[(0, 1)], [], [(0, -1)]],
        [[(0, 1), (1, 0)], [], [(1, 0), (0, -1)]],
        [[], [], []],
        [[], [(0, 1), (1, 0), (-1, 0), (0, -1), (-1, -1), (-1, 1)], []],
        [[(0, 1), (-1, 0)], [], [(0, -1), (-1, 0)]],
    ]

    def fill_missing(self, grid):
        for row_index, row in enumerate(self.directions):
            for col_index, direction in enumerate(row):
                if len(direction) != 0:
                    for step in direction:
                        for column in range(
                            col_index * self.segment_size,
                            (col_index + 1) * self.segment_size,
                        ):
                            grid[row_index][column] += (
                                grid[row_index + step[0]][column + step[1]]
                                / len(direction)
                            )
        return grid

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

    def progressive_interpolation(self, array, max_iterations=50, threshold=1e-6):
        """Fill zeros from the nearest known samples, closest points first."""
        result = array.copy()
        y_index, x_index = np.indices(array.shape)
        known_mask = array != 0
        unknown_mask = ~known_mask
        known_points = np.column_stack((x_index[known_mask], y_index[known_mask]))
        known_values = array[known_mask]
        unknown_points = np.column_stack(
            (x_index[unknown_mask], y_index[unknown_mask])
        )

        if len(known_points) == 0 or len(unknown_points) == 0:
            return result
        print(result)
        previous = result.copy()
        for _ in range(max_iterations):
            interpolated_values = griddata(
                known_points,
                known_values,
                unknown_points,
                method="cubic",
                fill_value=0,
            )
            result[unknown_mask] = interpolated_values

            still_zero = result == 0
            if np.any(still_zero):
                zero_points = np.column_stack(
                    (x_index[still_zero], y_index[still_zero])
                )
                linear_values = griddata(
                    known_points,
                    known_values,
                    zero_points,
                    method="linear",
                    fill_value=0,
                )
                result[still_zero] = linear_values

            if np.max(np.abs(result - previous)) < threshold:
                break

            known_mask = result != 0
            unknown_mask = ~known_mask
            known_points = np.column_stack(
                (x_index[known_mask], y_index[known_mask])
            )
            known_values = result[known_mask]
            unknown_points = np.column_stack(
                (x_index[unknown_mask], y_index[unknown_mask])
            )
            previous = result.copy()
            if len(unknown_points) == 0:
                break
        print(result)
        return result

    def _valid_neighbors(self, array, zero_coord, step):
        zero_y, zero_x = zero_coord
        rows, cols = array.shape
        valid_cols = [
            col for col in range(cols) if abs(col - zero_x) % step == 0
        ]
        valid_coords = []
        valid_values = []
        for col in valid_cols:
            for row in range(rows):
                if array[row, col] != 0:
                    valid_coords.append([row, col])
                    valid_values.append(array[row, col])
        return np.array(valid_coords), np.array(valid_values)

    def interpolate_missing(self, array, step):
        """Fill zeros from up to four neighbors on columns a multiple of ``step`` apart."""
        result = array.copy()
        zero_coords = np.argwhere(array == 0)
        if len(zero_coords) == 0:
            return result

        for zero_coord in zero_coords:
            valid_coords, valid_values = self._valid_neighbors(
                array, zero_coord, step
            )
            if len(valid_coords) == 0:
                continue

            distances = cdist(zero_coord.reshape(1, -1), valid_coords)[0]
            neighbor_count = min(4, len(distances))
            nearest = np.argpartition(distances, neighbor_count)[:neighbor_count]
            weights = 1 / (distances[nearest] + 1e-10)
            weights = weights / np.sum(weights)
            result[zero_coord[0], zero_coord[1]] = np.sum(
                valid_values[nearest] * weights
            )
        return result


if __name__ == "__main__":
    TopographicGrid(
        ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"], segment_size=100
    )
