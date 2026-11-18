"""Evaluate B: exclusions vs seed labels, and classification accuracy by 5-fold CV on the seed labels."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from spendcube.pipeline import run, _read
from spendcube import classify as clf

out, A, tree, info = run("configs/company_B.yaml", "data", "outputs", "B_eval")
s = pd.read_csv("data/seed_labels.csv", dtype=str)
m = s.merge(out, left_on="line_id", right_on="record_id", suffixes=("_t", ""))
print("exclusion confusion (rows=truth include, cols=pred include)")
print(pd.crosstab(m.include_t.astype(int), m.include))
# CV classification on included seeds
inc = A[A.include == 1].copy()
lab = s[s.include == "1"].copy()
lab["leaf"] = [tree.encode(list(r)) for r in lab[["category_l1", "category_l2", "category_l3"]].itertuples(index=False, name=None)]
d = inc.merge(lab[["line_id", "leaf"]], left_on="record_id", right_on="line_id")
accs = {1: [], 2: [], 3: []}
for tr, te in KFold(5, shuffle=True, random_state=0).split(d):
    P = clf.supervised_predict(tree, d.cls_text.iloc[tr], d.cls_ctx.iloc[tr], d.leaf.iloc[tr].to_numpy(), d.cls_text.iloc[te], d.cls_ctx.iloc[te])
    pred = P.argmax(1)
    for lv in (1, 2, 3):
        accs[lv].append(np.mean([tree.leaves[p][:lv] == tree.leaves[t][:lv] for p, t in zip(pred, d.leaf.iloc[te])]))
print({f"L{k}": round(float(np.mean(v)), 3) for k, v in accs.items()})
