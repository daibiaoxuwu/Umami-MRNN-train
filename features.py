"""Turn a peptide sequence into the numbers the two networks read.

- descriptors(seq): 1,093 numbers summarising the whole sequence, for the MLP.
  AAC (20) + CTDC (39) + CTDD (195) + CTDT (39) + DDE (400) + DPC (400).
  Definitions follow iFeature (https://github.com/Superzchen/iFeature).
- residue_codes(seq): a 10-number code per residue, for the RNN.

The web demo (predict.js in the umami-mrnn repository) implements the same
functions in JavaScript.
"""
import math

import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
MIN_LEN, MAX_LEN = 2, 39

# Each property splits the 20 amino acids into three groups.
PROPERTIES = {
    'hydrophobicity_PRAM900101': ('RKEDQN', 'GASTPHY', 'CLVIMFW'),
    'hydrophobicity_ARGP820101': ('QSTNGDE', 'RAHCKMV', 'LYPFIW'),
    'hydrophobicity_ZIMJ680101': ('QNGSWTDERA', 'HMCKV', 'LPFYI'),
    'hydrophobicity_PONP930101': ('KPDESNQT', 'GRHA', 'YMFWLCVI'),
    'hydrophobicity_CASG920101': ('KDEQPSRNTG', 'AHYMLV', 'FIWC'),
    'hydrophobicity_ENGD860101': ('RDKENQHYP', 'SGTAW', 'CVLIMF'),
    'hydrophobicity_FASG890101': ('KERSQD', 'NTPG', 'AYHWVMFLIC'),
    'normwaalsvolume': ('GASTPDC', 'NVEQIL', 'MHKFRYW'),
    'polarity': ('LIFWCMVY', 'PATGS', 'HQRKNED'),
    'polarizability': ('GASDT', 'CPNVEQIL', 'KMHFRYW'),
    'charge': ('KR', 'ANCQGHILMFPSTWYV', 'DE'),
    'secondarystruct': ('EALMQKRH', 'VIYCWFT', 'GNPSD'),
    'solventaccess': ('ALFCGIVW', 'RKQEND', 'MSPTHY'),
}

# Number of codons for each amino acid, used by DDE.
CODONS = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
          'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2}

# Overlapping property code of each residue (10 yes/no physicochemical properties).
OPF = {'A': '0000100110', 'R': '1101000000', 'N': '1000000100', 'D': '1011000100', 'C': '1000100110',
       'Q': '1000000000', 'E': '1011000000', 'G': '0000100110', 'H': '1101101000', 'I': '0000110000',
       'L': '0000110000', 'K': '1101100000', 'M': '0000100000', 'F': '0000101000', 'P': '0000000101',
       'S': '1000000110', 'T': '1000100100', 'W': '1000101000', 'Y': '1000101000', 'V': '0000110100'}


def count(group, seq):
    return sum(aa in group for aa in seq)


def aac(seq):
    """Amino-acid composition: fraction of each residue."""
    return [seq.count(aa) / len(seq) for aa in AA]


def ctdc(seq):
    """Composition: fraction of residues in each property group."""
    out = []
    for g1, g2, _ in PROPERTIES.values():
        c1, c2 = count(g1, seq) / len(seq), count(g2, seq) / len(seq)
        out += [c1, c2, 1 - c1 - c2]
    return out


def ctdd(seq):
    """Distribution: where (as % of length) the first, 25%, 50%, 75% and last
    residue of each group occurs."""
    out = []
    for groups in PROPERTIES.values():
        for group in groups:
            positions = [i + 1 for i, aa in enumerate(seq) if aa in group]
            n = len(positions)
            if n == 0:
                out += [0] * 5
                continue
            for cutoff in (1, math.floor(0.25 * n), math.floor(0.5 * n), math.floor(0.75 * n), n):
                out.append(positions[max(cutoff, 1) - 1] / len(seq) * 100)
    return out


def ctdt(seq):
    """Transition: fraction of neighbouring pairs that switch between two groups."""
    pairs = list(zip(seq, seq[1:]))
    out = []
    for g1, g2, g3 in PROPERTIES.values():
        def between(x, y, a, b):
            return (a in x and b in y) or (a in y and b in x)
        t12 = t13 = t23 = 0
        for a, b in pairs:
            if between(g1, g2, a, b):
                t12 += 1
            elif between(g1, g3, a, b):
                t13 += 1
            elif between(g2, g3, a, b):
                t23 += 1
        out += [t12 / len(pairs), t13 / len(pairs), t23 / len(pairs)]
    return out


def dpc(seq):
    """Dipeptide composition: fraction of each of the 400 neighbouring pairs."""
    counts = np.zeros(400)
    for a, b in zip(seq, seq[1:]):
        counts[AA.index(a) * 20 + AA.index(b)] += 1
    return list(counts / (len(seq) - 1))


def dde(seq, composition):
    """Dipeptide deviation from the frequency expected from codon usage."""
    out = []
    for k, observed in enumerate(composition):
        a, b = AA[k // 20], AA[k % 20]
        mean = (CODONS[a] / 61) * (CODONS[b] / 61)
        variance = mean * (1 - mean) / (len(seq) - 1)
        out.append((observed - mean) / math.sqrt(variance))
    return out


def check(seq):
    seq = seq.strip().upper()
    if not seq or any(aa not in AA for aa in seq):
        raise ValueError(f'{seq!r}: only the 20 standard amino-acid letters are allowed')
    if not MIN_LEN <= len(seq) <= MAX_LEN:
        raise ValueError(f'{seq!r}: length must be {MIN_LEN}-{MAX_LEN} residues')
    return seq


def descriptors(seq):
    seq = check(seq)
    pairs = dpc(seq)
    return np.array(aac(seq) + ctdc(seq) + ctdd(seq) + ctdt(seq) + dde(seq, pairs) + pairs,
                    dtype=np.float32)


def residue_codes(seq):
    """Array of shape (len(seq), 10)."""
    return np.array([[int(bit) for bit in OPF[aa]] for aa in check(seq)], dtype=np.float32)
