"""The two Umami-MRNN networks and how their outputs are combined.

Both networks predict a peptide's umami taste threshold in mmol/L. Bitter
(non-umami) peptides are trained with the value 40, so a prediction of 40 or
more means "not umami".
"""
import numpy as np
import torch
from torch import nn

import features

MLP_WEIGHT = 0.67  # final prediction = 0.67 * MLP + 0.33 * RNN
UMAMI_CUTOFF = 40  # mmol/L; predictions below this are classed as umami


class MLP(nn.Module):
    """Multilayer perceptron on the 1,093 sequence descriptors."""

    def __init__(self, n_features=1093, dropout=0.3):
        super().__init__()
        # Descriptors have very different scales (fractions, percentages,
        # z-scores), so each is standardised with the training-set mean and std.
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('std', torch.ones(n_features))
        self.layers = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def set_scaling(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)
        self.std[self.std < 1e-6] = 1  # descriptors that never vary are left unscaled

    def forward(self, x):  # x: (batch, 1093)
        return self.layers((x - self.mean) / self.std).squeeze(-1)


class RNN(nn.Module):
    """Two stacked recurrent layers reading the peptide one residue at a time."""

    def __init__(self, n_codes=10, hidden=32, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(n_codes, hidden, num_layers=2, nonlinearity='tanh',
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, lengths):  # x: (batch, max_len, 10), padded with zeros
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)  # h: (layers, batch, hidden), state after each peptide's last residue
        return self.out(self.dropout(h[-1])).squeeze(-1)


def encode(sequences):
    """Turn a list of sequences into the inputs of MLP and RNN."""
    x_mlp = torch.tensor(np.stack([features.descriptors(s) for s in sequences]))
    codes = [torch.tensor(features.residue_codes(s)) for s in sequences]
    x_rnn = nn.utils.rnn.pad_sequence(codes, batch_first=True)
    lengths = torch.tensor([len(c) for c in codes])
    return x_mlp, x_rnn, lengths


@torch.no_grad()
def predict(mlp, rnn, sequences):
    """Returns (mlp, rnn, combined) threshold predictions as numpy arrays."""
    mlp.eval(), rnn.eval()
    x_mlp, x_rnn, lengths = encode(sequences)
    a, b = mlp(x_mlp).numpy(), rnn(x_rnn, lengths).numpy()
    return a, b, MLP_WEIGHT * a + (1 - MLP_WEIGHT) * b
