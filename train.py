"""Train Umami-MRNN on data/peptides.csv and report test accuracy.

Usage: python train.py [--seed 0]

Writes models/mlp.pt, models/rnn.pt and models/split.csv (which peptides were
held out for testing).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from model import MLP, RNN, UMAMI_CUTOFF, encode, predict

# Bitter peptides have no umami threshold. They are trained towards a value well
# above the cutoff (40) so that predictions for them land clearly above it.
BITTER_LABEL = 50

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--test-fraction', type=float, default=0.2)
parser.add_argument('--mlp-epochs', type=int, default=300)
parser.add_argument('--rnn-epochs', type=int, default=300)
parser.add_argument('--rnn-lr', type=float, default=1e-2)
parser.add_argument('--bitter-label', type=float, default=BITTER_LABEL,
                    help='threshold (mmol/L) used as the training target for bitter peptides')
args = parser.parse_args()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

# ---- data: hold out the same fraction of umami and of bitter peptides ----
data = pd.read_csv(ROOT / 'data' / 'peptides.csv')
data['split'] = 'train'
for _, group in data.groupby('class'):
    test = rng.choice(group.index, size=round(len(group) * args.test_fraction), replace=False)
    data.loc[test, 'split'] = 'test'
train, test = data[data.split == 'train'], data[data.split == 'test']

# There are about twice as many bitter as umami peptides, so umami training
# peptides are repeated to give both classes similar weight.
repeat = round((train['class'] == 'bitter').sum() / (train['class'] == 'umami').sum())
train = pd.concat([train[train['class'] == 'bitter']] + [train[train['class'] == 'umami']] * repeat)

x_mlp, x_rnn, lengths = encode(train.sequence.tolist())
y = torch.tensor(np.where(train['class'] == 'bitter', args.bitter_label, train.threshold_mmol_L), dtype=torch.float32)
print(f'Training on {len(data) - len(test)} peptides ({len(train)} after repeating umami), testing on {len(test)}.')


def fit(net, inputs, epochs, lr):
    """Mini-batch training with mean squared error on the threshold."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        net.train()
        losses = []
        for batch in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(*(x[batch] for x in inputs)), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 50 == 0:
            net.eval()
            with torch.no_grad():
                error = (net(*inputs) - y).abs().mean().item()
            print(f'  epoch {epoch + 1}: train loss {np.mean(losses):.2f}, mean error {error:.2f} mmol/L')


print('Training MLP...')
mlp = MLP()
mlp.set_scaling(x_mlp)
fit(mlp, (x_mlp,), args.mlp_epochs, lr=1e-3)
print('Training RNN...')
rnn = RNN()
fit(rnn, (x_rnn, lengths), args.rnn_epochs, lr=args.rnn_lr)

# ---- evaluate on held-out peptides ----
is_umami = (test['class'] == 'umami').values
for name, pred in zip(['MLP', 'RNN', 'Combined'], predict(mlp, rnn, test.sequence.tolist())):
    accuracy = ((pred < UMAMI_CUTOFF) == is_umami).mean()
    mae = np.abs(pred - test.threshold_mmol_L.values)[is_umami].mean()
    print(f'{name:9s} test accuracy {accuracy:.1%}   threshold error on umami peptides {mae:.2f} mmol/L')

out = ROOT / 'models'
out.mkdir(exist_ok=True)
torch.save(mlp.state_dict(), out / 'mlp.pt')
torch.save(rnn.state_dict(), out / 'rnn.pt')
data[['sequence', 'class', 'split']].to_csv(out / 'split.csv', index=False)
print(f'Saved models to {out}')
