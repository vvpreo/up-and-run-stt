"""
Log-mel фичерайзер на numpy — замена torch/torchaudio FeatureExtractor
для ONNX-инференса (образ без PyTorch).

Численно повторяет vendored gigaam.preprocess.FeatureExtractor
(torchaudio.transforms.MelSpectrogram + log-clamp) для конфигурации
GigaAM v3: sr=16000, n_mels=64, n_fft=320, win=320, hop=160,
center=False, power=2.0, HTK mel-шкала, без нормализации фильтров.
Паритет проверяется тестом (сравнение с torch-фичами и итоговых
транскриптов).
"""

import numpy as np


def _hann_periodic(win_length: int) -> np.ndarray:
    """torch.hann_window(N, periodic=True): 0.5*(1-cos(2*pi*n/N))."""
    n = np.arange(win_length)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * n / win_length))).astype(np.float64)


def _hz_to_mel_htk(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz_htk(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(
    n_freqs: int, n_mels: int, sample_rate: int, f_min: float = 0.0
) -> np.ndarray:
    """
    Треугольный mel-фильтрбанк, повторяющий torchaudio.functional.melscale_fbanks
    (mel_scale='htk', norm=None). Возвращает матрицу (n_freqs, n_mels).
    """
    f_max = sample_rate / 2.0
    all_freqs = np.linspace(0.0, f_max, n_freqs)

    m_min = _hz_to_mel_htk(np.array(f_min))
    m_max = _hz_to_mel_htk(np.array(f_max))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz_htk(m_pts)

    # torchaudio: slopes между соседними точками
    f_diff = f_pts[1:] - f_pts[:-1]                        # (n_mels+1,)
    slopes = f_pts[None, :] - all_freqs[:, None]           # (n_freqs, n_mels+2)
    down = -slopes[:, :-2] / f_diff[:-1]                   # (n_freqs, n_mels)
    up = slopes[:, 2:] / f_diff[1:]                        # (n_freqs, n_mels)
    fb = np.maximum(0.0, np.minimum(down, up))
    return fb


class NumpyFeatureExtractor:
    """Log-mel спектрограмма: waveform float32 16kHz -> (n_mels, T) float32."""

    def __init__(
        self,
        sample_rate: int = 16000,
        features: int = 64,
        n_fft: int = 320,
        win_length: int = 320,
        hop_length: int = 160,
        center: bool = False,
        **_ignored,
    ):
        self.sample_rate = sample_rate
        self.n_mels = features
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.window = _hann_periodic(win_length)
        if win_length < n_fft:
            pad = (n_fft - win_length) // 2
            self.window = np.pad(self.window, (pad, n_fft - win_length - pad))
        self.fbank = _mel_filterbank(n_fft // 2 + 1, features, sample_rate)

    def out_len(self, num_samples: int) -> int:
        """Число фреймов (совпадает с torch.stft при тех же параметрах)."""
        if self.center:
            return num_samples // self.hop_length + 1
        return (num_samples - self.n_fft) // self.hop_length + 1

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """
        Args:
            wav: float32 numpy array, mono, 16 kHz, диапазон [-1, 1].

        Returns:
            (n_mels, T) float32 log-mel спектрограмма.
        """
        wav = np.asarray(wav, dtype=np.float64)
        if self.center:
            pad = self.n_fft // 2
            wav = np.pad(wav, (pad, pad), mode="reflect")

        n_frames = self.out_len(len(wav)) if not self.center else (
            (len(wav) - self.n_fft) // self.hop_length + 1
        )
        if n_frames <= 0:
            return np.zeros((self.n_mels, 0), dtype=np.float32)

        # Нарезка окон без копирования
        stride = wav.strides[0]
        frames = np.lib.stride_tricks.as_strided(
            wav,
            shape=(n_frames, self.n_fft),
            strides=(stride * self.hop_length, stride),
            writeable=False,
        )

        spec = np.abs(np.fft.rfft(frames * self.window, n=self.n_fft, axis=1)) ** 2
        mel = spec @ self.fbank                     # (T, n_mels)
        log_mel = np.log(np.clip(mel, 1e-9, 1e9))
        return log_mel.T.astype(np.float32)         # (n_mels, T)
