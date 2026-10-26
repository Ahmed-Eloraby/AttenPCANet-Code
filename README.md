# AttenPCA-Net

Code for the P300-speller model in:

> Ahmed Eloraby and Seif Eldawlatly, “AttenPCA-Net: A Multi-Attention Deep Learning Framework with Spatial EEG Representation for P300 Spellers Enhancement,” *12th International IEEE/EMBS Conference on Neural Engineering (NER)*, San Diego, CA, USA, 2025.  
> DOI: [10.1109/NER61569.2025.11589085](https://doi.org/10.1109/NER61569.2025.11589085)

AttenPCA-Net turns each flash of a P300 speller into a scalp image. A separate PCA compresses the time samples of every electrode, those components are laid onto a 10–20 grid, and a 2D network with channel attention and spatial attention scores the flash as P300 or not. The spelled character is the row and column with the largest summed P300 score.

This repository is the subject-dependent experiment on **BCI Competition III Dataset II** (subjects A and B).

## Architecture

<p align="center">
  <img src="figures/architecture.svg" alt="AttenPCA-Net architecture" width="720">
</p>


The network takes an input of shape `(principal components, 11, 11)`:

1. Two pointwise blocks map the components to 16 channels, then to 32. Each block is a 1×1 convolution, batch normalization, and squeeze-and-excitation.
2. Four 3×3 blocks follow, with batch normalization, squeeze-and-excitation, dropout 0.3, and an ELU. The channel counts are 32→64, 64→128, 128→256, and 256→256. Valid convolution shrinks the grid from 11×11 to 3×3.
3. Spatial attention reweights the grid.
4. A linear layer maps the 2304 flattened units to 64, with an ELU, and a second linear layer returns two scores.

## Pipeline

For each subject:

1. Notch filter at 60 Hz, band-pass from 0.5 Hz to 25 Hz, clip amplitudes to ±50, then a common-average reference.
2. Cut a 600 ms segment from every row and column flash.
3. Fit one PCA per channel on the training segments, along time. The run keeps 40 components, then 39, down to 1.
4. Place each component on its electrode in an 11×11 10–20 grid. Empty sites stay zero. The component axis is the channel axis of the convolution.
5. Train on all 15 repetitions (20% of those segments are the validation split). Early stopping watches validation loss, with patience 50.
6. Test with the first 15, 14, …, 1 repetitions. The model is not retrained for this sweep.
7. Sum the P300 score of each row and each column across repetitions. The predicted character is where the winning row and winning column meet.

Training uses Adam at learning rate 0.0001, batch size 128, and cross-entropy with equal class weights. NumPy, Python, and PyTorch seeds are set to 0.

## Project layout

| File | Role |
| --- | --- |
| `train.py` | Dataset II experiment |
| `attenpca_net.py` | Network and its optimizer |
| `trainer.py` | Training loop, early stopping, checkpoints |
| `dataset.py` | Dataset II reader |
| `preprocessing.py` | Notch, band-pass, clipping, common-average reference |
| `feature_extraction.py` | Flash segmentation and P300 labels |
| `dimensionality_reduction.py` | Per-channel temporal PCA |
| `electrode_grid.py` | 10–20 map used by the network |
| `evaluation.py` | P300 scores and character accuracy |
| `config.py` | 6×6 speller matrix and flash codes |
| `mlp.py` | Shallow network from the earlier feature comparison |
| `electrode_frames.py` | Frame-wise 10–20 map used to export ViViT clips |

## Setup

The experiment was written for Python 3.13.

```bash
pip install -r requirements.txt
```

Download [BCI Competition III Dataset II](https://www.bbci.de/competition/iii/#data_set_ii) and arrange it as follows. The test recordings use a lowercase `test` in the filename, which is how the loader asks for them.

```text
Benchmark dataset/
  Subject_A_Train.mat
  Subject_A_test.mat
  Subject_B_Train.mat
  Subject_B_test.mat
  true_labels_A.txt
  true_labels_B.txt
```

`true_labels_A.txt` and `true_labels_B.txt` are the 100-character test strings published with the competition.

## Run

From this directory:

```bash
python train.py
```

The script trains both subjects and writes a CSV in the working directory:

```text
attenPCA_0.5_25_loss_2_4_nodropout_<timestamp>.csv
```

Each row is one subject, one component count, and one repetition count. Checkpoints for the epoch with the lowest validation loss are saved under `AttenPCANet_checkpoints/`.

Fitted PCA models and transformed arrays are cached in the directory set by `PCA_CACHE_DIR` at the top of `train.py`. Point that path at a local folder before running.

A forward-pass check that does not need the dataset:

```bash
python attenpca_net.py
```
