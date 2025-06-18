import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Simple variational encoder implemented in PyTorch."""

    def __init__(self, input_size, code_size, hidden_size,
                 activation=nn.ReLU(), depth=1, dropout=0.5):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.fc_loc = nn.Linear(in_features, code_size)
        self.fc_log_scale = nn.Linear(in_features, code_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.mlp(x)
        loc = self.fc_loc(h)
        log_scale = self.fc_log_scale(h)
        scale = torch.exp(log_scale)
        eps = torch.randn_like(scale)
        code = loc + eps * scale
        return code, loc, scale
