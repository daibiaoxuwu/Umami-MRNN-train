"""Convert between the PyTorch models and the web demo's weight files.

Export trained models for the demo (https://github.com/daibiaoxuwu/umami-mrnn):
    python web_demo.py export path/to/umami-mrnn/model
This writes manifest.json and weights.bin into that folder.

load() goes the other way, e.g. to reuse the published demo weights in Python.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from model import MLP, RNN


def _tensors(mlp, rnn):
    """Name -> array in the demo's layout: kernels are (inputs, outputs)."""
    m, r = mlp.state_dict(), rnn.state_dict()
    # Fold the input standardisation into the first layer: W' = W / std, b' = b - W' . mean
    w1 = m['layers.0.weight'] / m['std']
    m['layers.0.bias'] = m['layers.0.bias'] - w1 @ m['mean']
    m['layers.0.weight'] = w1
    out = {}
    for i, layer in enumerate([0, 3, 6]):
        out[f'bp.d{i + 1}.bias'] = m[f'layers.{layer}.bias']
        out[f'bp.d{i + 1}.kernel'] = m[f'layers.{layer}.weight'].T
    for i in range(2):
        out[f'rnn.r{i + 1}.bias'] = r[f'rnn.bias_ih_l{i}'] + r[f'rnn.bias_hh_l{i}']
        out[f'rnn.r{i + 1}.kernel'] = r[f'rnn.weight_ih_l{i}'].T
        out[f'rnn.r{i + 1}.recurrent_kernel'] = r[f'rnn.weight_hh_l{i}'].T
    out['rnn.d.bias'] = r['out.bias']
    out['rnn.d.kernel'] = r['out.weight'].T
    return {k: v.numpy().astype('<f4') for k, v in out.items()}


def export(mlp, rnn, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest, chunks, offset = {}, [], 0
    for name, arr in _tensors(mlp, rnn).items():
        manifest[name] = {'shape': list(arr.shape), 'offset': offset}
        chunks.append(np.ascontiguousarray(arr).tobytes())
        offset += arr.size
    (folder / 'weights.bin').write_bytes(b''.join(chunks))
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=1))


def load(folder):
    """Build MLP and RNN from a demo manifest.json + weights.bin."""
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    data = np.fromfile(folder / 'weights.bin', dtype='<f4')
    w = {k: torch.tensor(data[v['offset']:v['offset'] + int(np.prod(v['shape']))].reshape(v['shape']))
         for k, v in manifest.items()}
    mlp, rnn = MLP(), RNN()
    state = {}
    for i, layer in enumerate([0, 3, 6]):
        state[f'layers.{layer}.weight'] = w[f'bp.d{i + 1}.kernel'].T
        state[f'layers.{layer}.bias'] = w[f'bp.d{i + 1}.bias']
    mlp.load_state_dict(state, strict=False)  # keeps mean 0, std 1
    state = {'out.weight': w['rnn.d.kernel'].T, 'out.bias': w['rnn.d.bias']}
    for i in range(2):
        state[f'rnn.weight_ih_l{i}'] = w[f'rnn.r{i + 1}.kernel'].T
        state[f'rnn.weight_hh_l{i}'] = w[f'rnn.r{i + 1}.recurrent_kernel'].T
        state[f'rnn.bias_ih_l{i}'] = w[f'rnn.r{i + 1}.bias']
        state[f'rnn.bias_hh_l{i}'] = torch.zeros_like(w[f'rnn.r{i + 1}.bias'])
    rnn.load_state_dict(state)
    return mlp, rnn


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'export':
        sys.exit(__doc__)
    models = Path(__file__).resolve().parent / 'models'
    mlp, rnn = MLP(), RNN()
    mlp.load_state_dict(torch.load(models / 'mlp.pt'))
    rnn.load_state_dict(torch.load(models / 'rnn.pt'))
    export(mlp, rnn, sys.argv[2])
    print(f'Wrote {sys.argv[2]}/manifest.json and weights.bin')
