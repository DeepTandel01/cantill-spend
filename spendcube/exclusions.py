"""Decide what is NOT procurement spend.

Layers (all optional, all config-driven):
  1. status rules      (CANCELLED / NOT_APPROVED)           -> applied in pipeline
  2. keyword rules     on configured text columns            -> NON_PROCUREMENT
  3. learned model     trained on labelled exclusions if the company ships any (e.g. B's seed labels)
"""
import re

import numpy as np
import pandas as pd

from .utils import norm_text

# Generic, industry-neutral concepts of "real money that is not third-party spend".
DEFAULT_PATTERNS = {
    "intercompany": r"\bI/C\b|\bIC\s?(PAYABLE|RECEIVABLE|\d+)|INTER\s?-?COMPANY|INTERCO\b|AFFILIATE|GROUP COMPANY|RELATED PART",
    "tax": r"\b(SALES|INCOME|PAYROLL|PROPERTY|CORPORATE|WITHHOLDING|EXCISE|STAMP|CAPITAL GAINS)\s+TAX|TAX EXPENSE|"
           r"\bVAT\b|\bGST\b|\bPAYE\b|TAXES\b|TAX PAYABLE|TAX AUTHORIT|REVENUE SERVICE|\bSARS\b",
    "royalty": r"ROYALT|LICEN[CS]E FEE REVENUE",
    "revenue": r"\bREVENUE\b|CUSTOMER (PAYMENT|REFUND|DEPOSIT)|SALES (INCOME|REVENUE)",
    "financing": r"\bINTEREST\b|BANK (FEE|CHARGE)|FINANC(E|ING) (COST|CHARGE|EXPENSE)|\bLOAN\b|DIVIDEND|"
                 r"(FX|FOREIGN EXCHANGE|EXCHANGE) (GAIN|LOSS|DIFFERENCE)|LEASE LIABILIT",
    "partner": r"PARTNER (BILLING|RECHARGE|COST)|JOINT VENTURE|\bJV\b (BILLING|CALL|COST)|RE-?CHARGE|COST ALLOCATION|"
               r"MANAGEMENT FEE|CASH CALL",
    "non_cash": r"DEPRECIATION|AMORTI[SZ]ATION|\bWRITE[- ]?OFF\b|IMPAIRMENT|PROVISION FOR|\bACCRUAL\b|"
                r"PAYROLL|SALARIES|WAGES\b",
}


def compile_rules(cfg_rules):
    """cfg_rules: {'use_default': True, 'extra': [{'code': 'NON_PROCUREMENT', 'pattern': '...'}], 'disable': [names]}"""
    cfg_rules = cfg_rules or {}
    rules = []
    if cfg_rules.get("use_default", True):
        for name, pat in DEFAULT_PATTERNS.items():
            if name in cfg_rules.get("disable", []):
                continue
            rules.append((name, "NON_PROCUREMENT", re.compile(pat, re.I)))
    for i, r in enumerate(cfg_rules.get("extra", [])):
        rules.append((f"extra{i}", r.get("code", "NON_PROCUREMENT"), re.compile(r["pattern"], re.I)))
    return rules


def apply_rules(rule_text, rules):
    """rule_text: Series of strings. Returns (code Series, rule-name Series)."""
    code = pd.Series("", index=rule_text.index)
    why = pd.Series("", index=rule_text.index)
    uniq = rule_text.fillna("").unique()
    hit = {}
    for u in uniq:
        for name, c, rx in rules:
            if rx.search(u):
                hit[u] = (c, name)
                break
    if hit:
        m = rule_text.map(lambda x: hit.get(x))
        has = m.notna()
        code[has] = m[has].map(lambda t: t[0])
        why[has] = m[has].map(lambda t: t[1])
    return code, why


class LearnedExclusion:
    """Logistic model on GL / account / supplier-type tokens + description words, trained on labelled exclusions."""

    def __init__(self, min_prob=0.6):
        self.min_prob = min_prob
        self.model = None

    def fit(self, texts, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y).astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            return self
        self.vec = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_/]{2,}", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.model = LogisticRegression(C=5.0, max_iter=500, class_weight="balanced").fit(X, y)
        return self

    def predict_proba(self, texts):
        if self.model is None:
            return np.zeros(len(texts))
        return self.model.predict_proba(self.vec.transform(texts))[:, 1]
