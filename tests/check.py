"""Sanity checks. Run from the repository root: python tests/check.py

1. The Python features reproduce the descriptors stored in the original
   spreadsheet of bitter peptides.
2. The PyTorch models, loaded with the published web-demo weights, reproduce
   the original TensorFlow predictions (tests/reference_predictions.json).
3. Exporting those models back gives byte-identical demo weight files.
4. If you have trained models (models/), exporting them for the web demo
   keeps their predictions unchanged.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import features, model, web_demo  # noqa: E402

# 1. features
sheets = pd.read_excel(ROOT / 'data' / 'original' / 'BitterProtein.xlsx', sheet_name=None)
seqs = list(sheets[next(iter(sheets))].iloc[:, 0].str.strip())
ours = np.array([features.descriptors(s) for s in seqs], dtype=float)
starts = {'AAC': 0, 'CTDC': 20, 'CTDD': 59, 'CTDT': 254, 'DDE': 293, 'DPC': 693}
for name, sheet in sheets.items():
    block = next(k for k in starts if k in name)
    expected = sheet.iloc[:, 1:].to_numpy(float)
    i = starts[block]
    assert np.allclose(ours[:, i:i + expected.shape[1]], expected, atol=1e-5), block
print(f'features match the spreadsheet for {len(seqs)} bitter peptides')

# 2. published weights reproduce TensorFlow
mlp, rnn = web_demo.load(ROOT / 'published')
reference = json.loads((ROOT / 'tests' / 'reference_predictions.json').read_text())
_, _, combined = model.predict(mlp, rnn, [r['seq'] for r in reference])
diff = np.abs(combined - [r['threshold'] for r in reference]).max()
assert diff < 1e-3, diff
print(f'published models match TensorFlow on {len(reference)} peptides (max difference {diff:.1e} mmol/L)')

# 3. export round trip
with tempfile.TemporaryDirectory() as tmp:
    web_demo.export(mlp, rnn, tmp)
    assert (Path(tmp) / 'weights.bin').read_bytes() == (ROOT / 'published' / 'weights.bin').read_bytes()
print('export reproduces the published weight file exactly')

# 4. exporting newly trained models keeps their predictions
if (ROOT / 'models' / 'mlp.pt').exists():
    mlp, rnn = model.MLP(), model.RNN()
    mlp.load_state_dict(torch.load(ROOT / 'models' / 'mlp.pt'))
    rnn.load_state_dict(torch.load(ROOT / 'models' / 'rnn.pt'))
    _, _, before = model.predict(mlp, rnn, seqs)
    with tempfile.TemporaryDirectory() as tmp:
        web_demo.export(mlp, rnn, tmp)
        _, _, after = model.predict(*web_demo.load(tmp), seqs)
    diff = np.abs(before - after).max()
    assert diff < 1e-3, diff
    print(f'exporting models/ for the web demo keeps predictions (max difference {diff:.1e})')
