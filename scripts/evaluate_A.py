"""Evaluate A against labels_A.csv: supplier/item de-dup (pairwise F1), classification with labels hidden
(zero-label mode) and with 3-fold supervised CV."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from spendcube.pipeline import run
from spendcube.dedupe import pair_scores
from spendcube import classify as clf

lab = pd.read_csv("data/labels_A.csv", dtype=str)
out, A, tree, info = run("configs/company_A.yaml", "data", "outputs", "A_eval")
erp = out[out.record_id.str.startswith("ERP:")].merge(lab, on="record_id", suffixes=("", "_t"))
print("supplier_key pairwise (ERP):", pair_scores(erp.supplier_key_t, erp.supplier_key))
print("item_key pairwise (ERP):    ", pair_scores(erp.item_key_t, erp.item_key))

def acc(o, tag):
    e = o[o.record_id.str.startswith("ERP:")].merge(lab, on="record_id", suffixes=("", "_t"))
    e = e[e.include == 1]
    w = e.spend.abs()
    r = {}
    for k, (p, t) in enumerate([("l1", "category_l1"), ("l2", "category_l2"), ("l3", "category_l3")], 1):
        ok = (e[p] == e[t].fillna(""))
        r[f"L{k}"] = (round(ok.mean(), 3), round((ok * w).sum() / w.sum(), 3))
    print(tag, "accuracy (count, spend-weighted):", r)

# zero-label mode: labels hidden, tree only
t0 = time.time()
o0, A0, _, i0 = run("configs/company_A.yaml", "data", "outputs", "A_nolabels", hide_labels=True)
print("zero-label method:", i0["method"], "time", round(time.time() - t0), "s")
acc(o0, "ZERO-LABEL")

# supervised 3-fold CV over ERP lines
inc = A[(A.include == 1) & (A.src == "P")].copy()
l = lab.copy()
l["leaf"] = [tree.encode(list(r)) for r in l[["category_l1", "category_l2", "category_l3"]].itertuples(index=False, name=None)]
d = inc.merge(l[["record_id", "leaf"]], on="record_id")
res = {1: [], 2: [], 3: []}
for tr, te in KFold(3, shuffle=True, random_state=0).split(d):
    P = clf.supervised_predict(tree, d.cls_text.iloc[tr], d.cls_ctx.iloc[tr], d.leaf.iloc[tr].to_numpy(), d.cls_text.iloc[te], d.cls_ctx.iloc[te])
    pr = P.argmax(1)
    for lv in (1, 2, 3):
        res[lv].append(np.mean([tree.leaves[p][:lv] == tree.leaves[t][:lv] for p, t in zip(pr, d.leaf.iloc[te])]))
print("SUPERVISED 3-fold CV:", {f"L{k}": round(float(np.mean(v)), 3) for k, v in res.items()})
