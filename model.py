import torch.nn as nn
import warnings
from torch.nn import functional as F
import torch
from model_code.resnet3Dcode import generate_model
from model_code.GlobalAttention import TransformerEncoderLayer, TransformerEncoder

warnings.filterwarnings("ignore", category=UserWarning)


# Reset 3D model(32-64-128-256]) + upsample_bilinear + 2D global attention + cov2d(256-128-64-1)
class Reset3DAttention(nn.Module):
    def __init__(self, model_depth=10, d_model=256, nhead=2, num_layers=4, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.resnet = generate_model(model_depth=model_depth, n_input_channels=2)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.cov = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 1, kernel_size=3, padding=0),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        x = self.resnet(x)  # 3D Resnet
        x = torch.mean(x, dim=2)  # take mean of third channel
        x = F.upsample_bilinear(x, scale_factor=2)  # up sampling
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # reshape it
        x = self.encoder(x, (h, w))  # input to transformer/global attention
        x = x.permute(1, 2, 0).view(bs, c, h, w)  # torch.Size([bs, 256, 14, 14])
        x = self.cov(x)  # cov2d
        x = x.view(-1, 1)
        return x

