"""Predict umami peptides with the published Umami-MRNN models.

Usage:
    python predict.py EGS DQR GFP          sequences on the command line
    python predict.py peptides.fasta       FASTA file, or one sequence per line
    python predict.py --models models ...  use your own trained models (train.py)
"""
import argparse
from pathlib import Path

import torch

import model
import web_demo

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='sequences, or a file of sequences')
parser.add_argument('--models', help='folder with mlp.pt and rnn.pt (default: published models)')
args = parser.parse_args()

names, seqs = [], []
for item in args.inputs:
    if not Path(item).is_file():
        names.append(item), seqs.append(item)
        continue
    for line in Path(item).read_text().split():
        if line.startswith('>'):
            names.append(line[1:])
        else:
            if len(names) == len(seqs):
                names.append(line)
            seqs.append(line)

if args.models:
    mlp, rnn = model.MLP(), model.RNN()
    mlp.load_state_dict(torch.load(Path(args.models) / 'mlp.pt'))
    rnn.load_state_dict(torch.load(Path(args.models) / 'rnn.pt'))
else:
    mlp, rnn = web_demo.load(ROOT / 'published')

_, _, thresholds = model.predict(mlp, rnn, seqs)
print('name\tsequence\tprediction\tthreshold_mmol_L')
for name, seq, t in zip(names, seqs, thresholds):
    print(f'{name}\t{seq}\t{"umami" if t < model.UMAMI_CUTOFF else "non-umami"}\t{t:.2f}')
