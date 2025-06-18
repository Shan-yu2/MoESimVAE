"""Simplified training script using PyTorch modules."""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .encoder import Encoder
from .decoder import Decoder


def main():
    parser = argparse.ArgumentParser(description="PyTorch MoE-Sim-VAE example")
    parser.add_argument('--num_markers', type=int, default=100)
    parser.add_argument('--code_size', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_experts', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()

    # Placeholder random data - replace with real dataset
    data = torch.rand(1000, args.num_markers)
    dataloader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True)

    encoder = Encoder(input_size=args.num_markers, code_size=args.code_size,
                      hidden_size=args.hidden_size)
    decoder = Decoder(num_experts=args.num_experts, code_size=args.code_size,
                      output_size=args.num_markers, hidden_size=args.hidden_size)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                             lr=args.learning_rate)

    for epoch in range(args.epochs):
        for batch, in dataloader:
            code, loc, scale = encoder(batch)
            recon, gates = decoder(code)
            recon_loss = F.binary_cross_entropy_with_logits(recon, batch, reduction='mean')
            kl = -0.5 * torch.mean(1 + torch.log(scale ** 2) - loc ** 2 - scale ** 2)
            loss = recon_loss + kl
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")


if __name__ == '__main__':
    main()
