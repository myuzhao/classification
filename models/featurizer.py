import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC

class AudioFeaturizer(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, feature_method='MelSpectrogram'):
        super().__init__()
        self._feature_method = feature_method
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        if feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(self.sample_rate, self.n_fft, self.n_mels)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(n_fft)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(sample_rate=self.sample_rate, n_mfcc=self.n_mels)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def forward(self, waveforms, input_lens_ratio):
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)
        mean = torch.mean(feature, 1, keepdim=True)
        std = torch.std(feature, 1, keepdim=True)
        feature = (feature - mean) / (std + 1e-5)

        return feature

    @property
    def feature_dim(self):
        if self._feature_method == 'LogMelSpectrogram':
            return self._feature_conf.n_mels
        elif self._feature_method == 'MelSpectrogram':
            return self._feature_conf.n_mels
        elif self._feature_method == 'Spectrogram':
            return self._feature_conf.n_fft // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._feature_conf.n_mfcc
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
