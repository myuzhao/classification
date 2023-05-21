import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC

class AudioFeaturizer(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, audio_time, f_min, f_max, spec_width = None):
        super().__init__()
        self.sample_rate = sample_rate
        if spec_width != None:
            hop_length = int(sample_rate * audio_time / (spec_width - 1))
            melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels, "center": False, "f_min": f_min, "f_max": f_max}
        else:
            melkwargs={"n_fft": n_fft, "hop_length": int(n_fft/2), "n_mels": n_mels, "center": False, "f_min": f_min, "f_max": f_max}

        self.feat_fun = MFCC(sample_rate=self.sample_rate, n_mfcc=n_mels, log_mels=True, melkwargs = melkwargs)
        
    def forward(self, waveforms):
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)
        mean = torch.mean(feature, 1, keepdim=True)
        std = torch.std(feature, 1, keepdim=True)
        feature = (feature - mean) / (std + 1e-5)

        return feature
