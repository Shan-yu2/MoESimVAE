import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, code_size, output_size, hidden_size,
                 activation=nn.ReLU(), depth=1):
        super().__init__()
        layers = []
        in_features = code_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(activation)
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Decoder(nn.Module):
    """Mixture of Experts decoder in PyTorch."""

    def __init__(self, num_experts, code_size, output_size, hidden_size,
                 activation=nn.ReLU(), depth=1):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(code_size, output_size, hidden_size, activation, depth)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(code_size, num_experts)

    def forward(self, code):
        gates = F.softmax(self.gate(code), dim=-1)
        outputs = torch.stack([expert(code) for expert in self.experts], dim=1)
        weighted = torch.einsum('bn,bnd->bd', gates, outputs)
        return weighted, gates
