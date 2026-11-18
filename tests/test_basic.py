"""Run:  python tests/test_basic.py   (no pytest needed)"""
import os, sys, types, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from spendcube.utils import parse_amount, parse_vat, parse_dates, norm_supplier
from spendcube.dedupe import cluster_strings

def test_amounts():
    assert parse_amount("1,234.56") == 1234.56
    assert parse_amount("1.234,56") == 1234.56
    assert parse_amount("(12.30)") == -12.30
    assert parse_amount("12,5") == 12.5
    assert parse_amount("1,234") == 1234
    assert np.isnan(parse_amount("abc"))
    assert parse_vat(None) == 0.15 and parse_vat("15%") == 0.15 and parse_vat("0.15") == 0.15 and parse_vat("0") == 0.0

def test_dates():
    s = pd.Series(["2024-03-29", "27-Mar-24", "11/10/2022", "13/10/2022", "garbage", None, "1900-01-01"])
    d, f = parse_dates(s, min_year=2000)
    assert list(f) == ["ok", "ok", "ok", "ok", "unparseable", "missing", "out_of_range"], list(f)
    assert d.iloc[2].month == 10  # day-first by default

def test_suppliers():
    assert norm_supplier("KEL MONT (PTY) LTD") == norm_supplier("Kelmont") == "KELMONT"
    m = cluster_strings([norm_supplier(x) for x in ["OAKENT MINING SA (PTY) LTD", "OAKENT  MINING SA", "OAK ENT MINING"]], 0.9)
    assert len(set(m.values())) == 1

def test_llm_plumbing(tmp="/tmp/spend_llm_test"):
    """Stub the anthropic SDK; checks prompt -> JSON parsing -> cache -> tree mapping (no network)."""
    from spendcube import classify as clf
    class R:  # fake response
        def __init__(s, t): s.content = [types.SimpleNamespace(type="text", text=t)]; s.usage = types.SimpleNamespace(input_tokens=100, output_tokens=20)
    class M:
        def create(self, **kw):
            n = kw["messages"][0]["content"].count("\n") and [l for l in kw["messages"][0]["content"].split("Lines:\n")[1].split("\n")]
            return R(json.dumps([{"n": int(l.split(":")[0]), "leaf": 1, "conf": 0.9} for l in n]))
    fake = types.ModuleType("anthropic"); fake.Anthropic = lambda: types.SimpleNamespace(messages=M())
    sys.modules["anthropic"] = fake
    tree = clf.Tree(pd.DataFrame({"category_l1": ["A", "A", "B"], "category_l2": ["x", "y", "z"]}))
    res, cost = clf.llm_predict(tree, ["valve 2in", "pump"], {"batch_size": 1}, os.path.join(tmp, "c.json"))
    assert res["valve 2in"][0] == 1 and os.path.exists(os.path.join(tmp, "c.json"))

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
