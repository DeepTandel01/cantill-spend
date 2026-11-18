"""Fuzzy clustering of strings (suppliers, items) with union-find over char-ngram TF-IDF neighbours."""
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_num = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_of(s):
    return tuple(sorted(_num.findall(s)))


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_strings(keys, threshold=0.88, ngram=(2, 4), numeric_guard=True, min_len=3, chunk=800):
    """keys: iterable of normalised strings (repeats allowed). Returns dict key -> cluster representative
    (the most frequent member). Strings with equal keys always share a cluster."""
    cnt = Counter(k for k in keys if k)
    uniq = sorted(cnt)
    n = len(uniq)
    if n == 0:
        return {}
    uf = UF(n)
    if n > 1 and threshold < 1.0:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, sublinear_tf=True, dtype=np.float32)
        X = vec.fit_transform(uniq).tocsr()
        XT = X.T.tocsr()
        nums = [numbers_of(u) for u in uniq] if numeric_guard else None
        for i in range(0, n, chunk):
            S = (X[i:i + chunk] @ XT).tocoo()
            m = (S.data >= threshold) & ((S.row + i) < S.col)
            for r, c in zip(S.row[m] + i, S.col[m]):
                if len(uniq[r]) < min_len or len(uniq[c]) < min_len:
                    continue
                if numeric_guard and nums[r] != nums[c]:
                    continue
                uf.union(int(r), int(c))
    groups = {}
    for i, u in enumerate(uniq):
        groups.setdefault(uf.find(i), []).append(u)
    out = {}
    for members in groups.values():
        rep = max(members, key=lambda k: (cnt[k], -len(k)))
        for m_ in members:
            out[m_] = rep
    return out


def pair_scores(truth, pred):
    """Pairwise precision/recall/F1 of a clustering vs truth (sampled counting via contingency)."""
    import pandas as pd
    df = pd.DataFrame({"t": truth, "p": pred})
    c2 = lambda s: (s * (s - 1) / 2).sum()
    tp = c2(df.groupby(["t", "p"]).size())
    p = c2(df.groupby("p").size())
    t = c2(df.groupby("t").size())
    prec, rec = tp / p if p else 1.0, tp / t if t else 1.0
    return dict(precision=round(prec, 4), recall=round(rec, 4), f1=round(2 * prec * rec / (prec + rec + 1e-12), 4))
