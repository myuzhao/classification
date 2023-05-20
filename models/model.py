import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

from models.pooling import AttentiveStatsPool, TemporalAveragePooling
from models.pooling import SelfAttentivePooling, TemporalStatisticsPooling

class TDNN(nn.Module):
    def __init__(self, num_class, input_size=80, channels=512, embd_dim=192, pooling_type="ASP"):
        super(TDNN, self).__init__()
        self.emb_size = embd_dim
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=512, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.td_layer2 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.td_layer3 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.td_layer4 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.td_layer5 = torch.nn.Conv1d(in_channels=512, out_channels=channels, dilation=1, kernel_size=1, stride=1)

        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(channels, 128)
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn5 = nn.BatchNorm1d(channels)
            self.linear = nn.Linear(channels, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn5 = nn.BatchNorm1d(channels * 2)
            self.linear = nn.Linear(channels * 2, embd_dim)
            self.bn6 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.fc = nn.Linear(embd_dim, num_class)

    def forward(self, x):
        """
        Compute embeddings.

        Args:
            x (torch.Tensor): Input data with shape (N, time, freq).

        Returns:
            torch.Tensor: Output embeddings with shape (N, self.emb_size, 1)
        """
        x = x.transpose(2, 1)
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)
        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)
        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)
        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)
        x = F.relu(self.td_layer5(x))
        out = self.bn5(self.pooling(x))
        out = self.bn6(self.linear(out))
        out = self.fc(out)
        out = nn.functional.softmax(out, dim = 1)
        return out


def show_params(nnet):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in nnet.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size())
                i = 1
                for j in params.size():
                    i = i * j
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)
    
def show_model(nnet):
    print("=" * 40, "Model Structures", "=" * 40)
    for module_name, m in nnet.named_modules():
        if module_name == '':
            print(m)
    print("=" * 98)

def save_model(model,out_path):
    print(out_path)
    torch.save(model.module.state_dict(),out_path)

def load_model(model_path, gpu_idxs):
    try: 
        model = TDNN()
        model = torch.nn.DataParallel(model, device_ids = gpu_idxs)
        model.load_state_dict(torch.load(model_path))
    except:
        model = TDNN()
        model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    num_class = 2 
    input_size = 128
    channels = 512
    embd_dim = 192
    model = TDNN(num_class, input_size, channels, embd_dim)
    show_model(model)