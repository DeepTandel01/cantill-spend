"""Classification into a company's category tree. The tree is read from a file at run time; no category
names live in the code.

Backends (config `classify.method`):
  supervised : TF-IDF + linear model trained on whatever labels the company ships (seed / golden labels)
  llm        : Claude API, zero labels, unique descriptions batched + cached
  lexical    : no labels, no API: tree-name similarity + self-training (always available)
  embedding  : like lexical but with sentence-transformers if installed
  auto       : supervised if labels, else llm if key present, else embedding if installed, else lexical
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .utils import norm_text, GENERIC_DESCRIPTIONS


# ------------------------------------------------------------------ tree
class Tree:
    def __init__(self, df, level_cols=None):
        if level_cols is None:
            level_cols = [c for c in df.columns if re.fullmatch(r"(category_)?l\d+", c.strip().lower())]
            level_cols = sorted(level_cols, key=lambda c: int(re.findall(r"\d+", c)[0]))
        if not level_cols:  # fall back to all columns in file order
            level_cols = list(df.columns)
        self.level_cols = level_cols
        self.nlevels = len(level_cols)
        d = df[level_cols].copy()
        for c in level_cols:
            d[c] = d[c].where(d[c].notna(), "").astype(str).str.strip()
        d = d.drop_duplicates().reset_index(drop=True)
        self.paths = [tuple(x for x in row if x) for row in d.itertuples(index=False, name=None)]
        # keep only leaves (a path that is a strict prefix of another is not a leaf)
        pset = set(self.paths)
        self.leaves = [p for p in self.paths if not any(q != p and q[:len(p)] == p for q in pset)]
        self.leaf_index = {p: i for i, p in enumerate(self.leaves)}
        self.path_text = [" ".join(p) for p in self.leaves]

    def encode(self, row_levels):
        """tuple/list of level names (blank ok) -> leaf idx or -1; tolerant of repeated names."""
        p = tuple(x for x in row_levels if isinstance(x, str) and x.strip())
        if p in self.leaf_index:
            return self.leaf_index[p]
        # try prefix match to a unique leaf
        c = [i for i, l in enumerate(self.leaves) if l[:len(p)] == p]
        return c[0] if len(c) == 1 else -1

    def levels(self, leaf_idx):
        p = self.leaves[leaf_idx]
        return list(p) + [""] * (4 - len(p)) if len(p) <= 4 else list(p[:4])


# ------------------------------------------------------------------ feature helpers
def _softmax(z, t=3.0):
    z = z * t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Featurizer:
    def __init__(self):
        self.w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"[A-Z0-9]{2,}", dtype=np.float32)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 max_features=200000, dtype=np.float32)
        self.x = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, sublinear_tf=True, dtype=np.float32)

    def fit_transform(self, text, ctx):
        return hstack([self.w.fit_transform(text), self.c.fit_transform(text), 0.6 * self.x.fit_transform(ctx)]).tocsr()

    def transform(self, text, ctx):
        return hstack([self.w.transform(text), self.c.transform(text), 0.6 * self.x.transform(ctx)]).tocsr()


# ------------------------------------------------------------------ supervised
def supervised_predict(tree, train_text, train_ctx, train_leaf, text, ctx, weights=None):
    ok = np.asarray(train_leaf) >= 0
    f = Featurizer()
    Xtr = f.fit_transform(pd.Series(train_text)[ok], pd.Series(train_ctx)[ok])
    clf = LinearSVC(C=0.5, class_weight=None)
    clf.fit(Xtr, np.asarray(train_leaf)[ok], sample_weight=None if weights is None else np.asarray(weights)[ok])
    X = f.transform(text, ctx)
    sc = clf.decision_function(X)
    if sc.ndim == 1:
        sc = np.vstack([-sc, sc]).T
    P = _softmax(sc, 3.0)
    full = np.zeros((len(text), len(tree.leaves)))
    full[:, clf.classes_] = P
    return full


# ------------------------------------------------------------------ lexical / embedding zero-shot
def _zero_shot_lexical(tree, text):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    vec.fit(list(text) + tree.path_text)
    A = vec.transform(text)
    B = vec.transform(tree.path_text)
    return (A @ B.T).toarray()


def _zero_shot_embedding(tree, text, model_name):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    a = m.encode(list(text), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    b = m.encode(tree.path_text, normalize_embeddings=True)
    return a @ b.T


def self_train(tree, text, ctx, S0, rounds=2, top_frac=0.4):
    """Bootstrap: confident zero-shot predictions -> pseudo labels -> supervised model -> blend."""
    S0 = np.asarray(S0)
    z = (S0 - S0.mean(axis=1, keepdims=True)) / (S0.std(axis=1, keepdims=True) + 1e-9)
    P0 = _softmax(z, 1.0)
    P = P0
    for _ in range(rounds):
        conf = np.sort(P, axis=1)[:, -2:]
        margin = conf[:, 1] - conf[:, 0]
        pred = P.argmax(1)
        keep = np.zeros(len(pred), bool)
        for k in np.unique(pred):  # per-class top fraction so rare classes survive
            idx = np.where(pred == k)[0]
            n = max(1, int(len(idx) * top_frac))
            keep[idx[np.argsort(-margin[idx])[:n]]] = True
        if keep.sum() < 20 or len(np.unique(pred[keep])) < 2:
            break
        Ps = supervised_predict(tree, text, ctx, np.where(keep, pred, -1), text, ctx)
        P = 0.4 * P0 + 0.6 * Ps
    return P


# ------------------------------------------------------------------ LLM
LLM_SYSTEM = ("You are a procurement spend classifier. You map purchase-line descriptions (often abbreviated, "
              "multilingual, noisy) to the single best leaf category of a fixed category tree. "
              "Reply with JSON only.")


def _llm_prompt(tree, items, examples):
    leaf_lines = "\n".join(f"{i}: {' > '.join(p)}" for i, p in enumerate(tree.leaves))
    ex = ""
    if examples:
        ex = "Labelled examples:\n" + "\n".join(f"- {t[:100]!r} -> {i}" for t, i in examples) + "\n\n"
    body = "\n".join(f"{k}: {t[:240]}" for k, t in enumerate(items))
    return (f"Category tree leaves (id: path):\n{leaf_lines}\n\n{ex}"
            "Classify each numbered line to exactly one leaf id. If the text is uninformative, pick the most "
            "probable leaf and give a low confidence. Output a JSON array like "
            '[{"n":0,"leaf":12,"conf":0.8}, ...] covering every line, nothing else.\n\n'
            f"Lines:\n{body}")


def llm_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa
        return True
    except ImportError:
        return False


def llm_predict(tree, texts, cfg, cache_path, examples=None, log=print):
    """texts: list of unique strings. Returns dict text -> (leaf_idx, conf). Cached on disk."""
    import anthropic
    model = cfg.get("model", os.environ.get("SPEND_LLM_MODEL", "claude-haiku-5-5"))
    bs = int(cfg.get("batch_size", 40))
    tree_sig = hashlib.sha1("|".join(map(str, tree.leaves)).encode()).hexdigest()[:10]
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    todo = [t for t in texts if f"{tree_sig}|{t}" not in cache]
    usage = {"in": 0, "out": 0}
    client = anthropic.Anthropic()

    def run(batch):
        prompt = _llm_prompt(tree, batch, examples)
        for attempt in range(4):
            try:
                r = client.messages.create(model=model, max_tokens=4000, system=LLM_SYSTEM,
                                           messages=[{"role": "user", "content": prompt}])
                usage["in"] += r.usage.input_tokens
                usage["out"] += r.usage.output_tokens
                txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                m = re.search(r"\[.*\]", txt, re.S)
                arr = json.loads(m.group(0))
                out = {}
                for o in arr:
                    n, leaf = int(o["n"]), int(o["leaf"])
                    if 0 <= n < len(batch) and 0 <= leaf < len(tree.leaves):
                        out[batch[n]] = (leaf, float(o.get("conf", 0.6)))
                return out
            except Exception as ex:  # noqa
                time.sleep(2 * (attempt + 1))
        return {}

    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    log(f"LLM: {len(todo)} new unique texts in {len(batches)} batches (model={model})")
    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 6))) as ex:
        for res in ex.map(run, batches):
            for t, v in res.items():
                cache[f"{tree_sig}|{t}"] = list(v)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w"))
    pin, pout = float(cfg.get("usd_per_mtok_in", 1.0)), float(cfg.get("usd_per_mtok_out", 5.0))
    cost = usage["in"] / 1e6 * pin + usage["out"] / 1e6 * pout
    log(f"LLM tokens in={usage['in']} out={usage['out']} est_cost=${cost:.3f} (rates in config; verify)")
    return {t: tuple(cache[f"{tree_sig}|{t}"]) for t in texts if f"{tree_sig}|{t}" in cache}, cost


# ------------------------------------------------------------------ orchestrator
def classify(tree, df, cfg, labels=None, cache_dir="outputs/.cache", log=print):
    """df needs cls_text, cls_ctx, sup_name (optional) for the rows to classify. labels: DataFrame aligned to df
    index with column 'leaf' (>=0 where known) or None. Returns (leaf Series, conf Series, method)."""
    method = cfg.get("method", "auto")
    have_labels = labels is not None and (labels >= 0).sum() >= int(cfg.get("min_labels", 200))
    if method == "auto":
        method = "supervised" if have_labels else ("llm" if llm_available() else
                                                  ("embedding" if _has_st() else "lexical"))
    log(f"classifier method: {method}")
    text = df["cls_text"].fillna("")
    ctx = df["cls_ctx"].fillna("")
    n = len(df)
    P = None
    if method == "supervised":
        if not have_labels:
            raise SystemExit("method=supervised but no labels")
        known = labels >= 0
        P = supervised_predict(tree, text[known], ctx[known], labels[known].to_numpy(), text, ctx)
    elif method == "llm":
        text = df["llm_text"].fillna("") if "llm_text" in df.columns else text
        uniq = list(pd.unique(text))
        generic = {t for t in uniq if norm_text(t) in GENERIC_DESCRIPTIONS}
        ex = None
        if have_labels:
            kk = labels[labels >= 0].sample(min(40, int((labels >= 0).sum())), random_state=0)
            ex = [(text[i], int(labels[i])) for i in kk.index]
        res, cost = llm_predict(tree, [t for t in uniq if t not in generic], cfg.get("llm", {}),
                                os.path.join(cache_dir, f"llm_{cfg.get('cache_name', 'x')}.json"), ex, log)
        # fallback for texts the LLM missed / generic: lexical self-training
        S0 = _zero_shot_lexical(tree, text)
        P = self_train(tree, text, ctx, S0)
        for i, t in enumerate(text):
            r = res.get(t)
            if r:
                P[i] = 0.0
                P[i, r[0]] = max(r[1], 0.05)
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
        cf = np.array([res[t][1] if t in res else np.nan for t in text])
        leaf = pd.Series(P.argmax(1), index=df.index)
        conf = pd.Series(np.where(np.isnan(cf), P.max(1) * 0.6, cf), index=df.index)
        return leaf, conf.clip(0, 1), method
    else:
        if method == "embedding":
            S0 = _zero_shot_embedding(tree, text, cfg.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))
        else:
            S0 = _zero_shot_lexical(tree, text)
        P = self_train(tree, text, ctx, S0)
    leaf = pd.Series(P.argmax(1), index=df.index)
    conf = pd.Series(P.max(1), index=df.index)
    return leaf, conf.clip(0, 1), method


def _has_st():
    try:
        import sentence_transformers  # noqa
        return True
    except ImportError:
        return False
