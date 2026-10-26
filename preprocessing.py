"""EEG preprocessing used before epoching.

``MNEPreprocessor`` is the path used by the AttenPCA-Net experiment.
``SciPyPreprocessor`` is the earlier filter chain kept alongside it.
"""

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, sosfilt
from sklearn.preprocessing import StandardScaler
import mne


class MNEPreprocessor:
    """Filter a continuous recording with MNE.

    ``eeg_data`` is shaped ``(samples, channels)``. Filters are applied on a
    copy held as an MNE ``RawArray`` and returned in the same orientation.
    """

    def __init__(self, eeg_data, sfreq=240, channel_names=None):
        self.eeg_data = eeg_data.T
        print(self.eeg_data.shape)
        self.sfreq = sfreq
        self.n_channels, self.n_samples = self.eeg_data.shape
        self.info = mne.create_info(
            ch_names=(
                channel_names
                if channel_names
                else [f"Ch{i}" for i in range(self.n_channels)]
            ),
            sfreq=self.sfreq,
            ch_types="eeg",
        )
        self.raw = mne.io.RawArray(self.eeg_data, self.info)

    def apply_notch_filter(self, freq=60):
        """Remove power-line noise. Dataset II was recorded at 60 Hz mains."""
        self.raw.notch_filter(freqs=freq, fir_design="firwin")
        return self

    def apply_bandpass_filter(self, low=0.5, high=25):
        self.raw.filter(l_freq=low, h_freq=high, fir_design="firwin")
        return self

    def apply_car_filter(self):
        """Common-average reference."""
        self.raw.set_eeg_reference("average")
        return self

    def apply_wavelet_denoising(self, wavelet="db4"):
        eeg_data = self.raw.get_data()

        def wavelet_denoise(signal, level=5):
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-level])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            coeffs[1:] = [
                pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
            ]
            return pywt.waverec(coeffs, wavelet)[: len(signal)]

        self.eeg_data = np.array(
            [wavelet_denoise(eeg_data[ch]) for ch in range(self.n_channels)]
        )
        self.raw = mne.io.RawArray(self.eeg_data, self.info)
        return self

    def apply_ica(self, n_components=8):
        ica = mne.preprocessing.ICA(
            n_components=n_components, random_state=97, max_iter=500
        )
        ica.fit(self.raw)
        self.raw = ica.apply(self.raw)
        return self

    def apply_thresholding(self, threshold=50):
        """Clip extreme amplitudes left after filtering."""
        self.eeg_data = self.raw.get_data()
        self.eeg_data = np.clip(self.eeg_data, -threshold, threshold)
        self.raw = mne.io.RawArray(self.eeg_data, self.info)
        return self

    def get_processed_data(self):
        """Return ``(samples, channels)``."""
        return self.raw.get_data().T


class SciPyPreprocessor:
    """Butterworth / notch chain that does not depend on MNE."""

    def __init__(self, eeg_data, sfreq=240, channel_names=None):
        self.signal = eeg_data
        self.fs = sfreq

    def apply_bandpass_filter(self, low_freq=0.1, high_freq=20, order=8):
        nyquist = self.fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        sos = butter(N=order, Wn=[low, high], btype="bandpass", output="sos")
        self.signal = sosfilt(sos, self.signal, axis=0)
        return self

    def apply_notch_filter(self, freq=50, quality=35):
        b_notch, a_notch = iirnotch(freq, quality, self.fs)
        self.signal = filtfilt(b_notch, a_notch, self.signal, axis=0)
        return self

    def apply_car_filter(self):
        self.signal -= np.mean(self.signal, axis=-1, keepdims=True)
        return self

    def standardize_all(self):
        center = np.mean(self.signal)
        scale = np.std(self.signal)
        self.signal = (self.signal - center) / scale
        return self

    def standardize_channels(self):
        self.signal = StandardScaler().fit_transform(self.signal)
        return self

    def apply_thresholding(self, threshold=50):
        self.signal = np.clip(self.signal, -threshold, threshold)
        return self

    def get_processed_data(self):
        return self.signal
