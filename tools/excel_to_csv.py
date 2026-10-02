"""Build data/peptides.csv from the original spreadsheets in data/original/.

Usage: python tools/excel_to_csv.py

Notes on the original data:
- Most umami peptides appear two or three times in AllUmami.xlsx (to balance
  them against the bitter peptides). The CSV keeps each peptide once; train.py
  does the balancing explicitly, so test peptides never also appear in training.
- Row 77 of AllUmami.xlsx is named VDV, but its descriptors (and
  PeptideSequence.xlsx) show it is VAV.
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / 'data' / 'original'
NON_UMAMI_THRESHOLD = 40  # bitter (non-umami) peptides are labelled with this threshold


def parse_threshold(value):
    """Turn a reported threshold such as '>10', '<0.5 mM' or 3 into mmol/L.

    Missing or unparseable values become 10, as in the original training code.
    """
    if isinstance(value, str):
        number = value.split('>')[-1].split('<')[-1].split(' ')[0].split('m')[0]
        return float(number) if number else 10.0
    return 10.0 if math.isnan(value) else float(value)


umami = pd.read_excel(ORIGINAL / 'AllUmami.xlsx')  # first sheet: sequence, threshold, AAC
bitter = pd.read_excel(ORIGINAL / 'BitterProtein.xlsx')

umami.iloc[76, 0] = 'VAV'

rows = [{'sequence': s.strip().upper(), 'class': 'umami', 'threshold_reported': t,
         'threshold_mmol_L': parse_threshold(t)}
        for s, t in zip(umami.iloc[:, 0], umami.iloc[:, 1])]
rows += [{'sequence': s.strip().upper(), 'class': 'bitter', 'threshold_reported': '',
          'threshold_mmol_L': NON_UMAMI_THRESHOLD}
         for s in bitter.iloc[:, 0]]

df = pd.DataFrame(rows).drop_duplicates('sequence')
df.to_csv(ROOT / 'data' / 'peptides.csv', index=False)
print(df['class'].value_counts().to_dict())
