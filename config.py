"""6 x 6 P300 speller layout for BCI Competition III Dataset II."""

import numpy as np


class BenchmarkConfiguration:
    """Row and column flash codes for the Farwell–Donchin matrix.

    Column flashes are coded 1–6 and row flashes are coded 7–12, matching
    the ``StimulusCode`` values stored in the dataset.
    """

    NUMBER_OF_TRIALS = 15
    NUMBER_OF_COLS = 6
    NUMBER_OF_ROWS = 6
    NUMBER_OF_FLASHES = NUMBER_OF_ROWS + NUMBER_OF_COLS
    MATRIX = np.array(
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "_",
        ]
    ).reshape((NUMBER_OF_ROWS, NUMBER_OF_COLS))

    CHARACTER_INDEX = {}
    for row in range(NUMBER_OF_ROWS):
        for col in range(NUMBER_OF_COLS):
            CHARACTER_INDEX[MATRIX[row][col]] = [
                row + NUMBER_OF_COLS + 1,
                col + 1,
            ]
