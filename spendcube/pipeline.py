"""End-to-end spend cube pipeline, driven entirely by a YAML/JSON company config."""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from . import classify as clf
from .dedupe import cluster_strings
from .exclusions import LearnedExclusion, apply_rules, compile_rules
from .utils import (GENERIC_DESCRIPTIONS, norm_item, norm_supplier, norm_text, parse_amount, parse_dates,
                    parse_rate, parse_vat)

OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)


def _read(data_dir, name):
    p = os.path.join(data_dir, name)
    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""], encoding_errors="replace")
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df.replace("", np.nan)


def _col(df, name, default=""):
    if not name or name not in df.columns:
        return pd.Series(default, index=df.index, dtype=object)
    return df[name].fillna(default)


def _ctx(df, cols):
    """Context columns -> space separated tokens like 'gl_account_GL40102' (empty values skipped)."""
    parts = []
    for c in cols or []:
        if c not in df.columns:
            continue
        v = df[c].fillna("").astype(str).map(norm_text).str.replace(" ", "_")
        parts.append(np.where(v != "", c + "_" + v, ""))
    if not parts:
        return pd.Series("", index=df.index)
    return pd.Series([" ".join(x for x in t if x) for t in zip(*parts)], index=df.index)


def _join(df, cols, sep=" "):
    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        return pd.Series("", index=df.index)
    return df[cols].fillna("").astype(str).agg(sep.join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()


# ================================================================= standardise one source
def standardise(df, s, cfg, fx=None, log=print):
    """Map one raw table to the standard frame using its column map `s`."""
    o = pd.DataFrame(index=df.index)
    o["record_id"] = s.get("id_prefix", "") + _col(df, s["id_col"]).astype(str)
    o["system"] = s.get("system_name", s.get("id_prefix", "").strip(":") or "SRC")
    # --- amount
    amt = _col(df, s["amount_col"], np.nan).map(parse_amount)
    basis = s.get("amount_basis", "net")
    if basis == "gross":
        vat = _col(df, s.get("vat_col"), "").map(lambda v: parse_vat(v, s.get("default_vat", 0.0)))
        o["vat_rate"] = vat
        amt = amt / (1 + vat)
    o["amount_orig"] = amt
    # --- currency / fx
    rep = cfg.get("reporting_currency", "USD")
    cur = _col(df, s.get("currency_col"), s.get("currency", rep)).astype(str).str.upper().replace("", rep)
    cur = cur.replace({"NAN": rep})
    o["currency"] = cur
    # --- dates (first usable of date_cols)
    d_all, f_all = None, None
    for dc in s.get("date_cols", [s["date_col"]] if s.get("date_col") else []):
        if dc not in df.columns:
            continue
        d, f = parse_dates(df[dc], s.get("date_formats"), s.get("dayfirst", True),
                           s.get("min_year", 2000), s.get("max_year", 2035))
        if d_all is None:
            d_all, f_all = d.copy(), f.copy()
        else:
            miss = f_all != "ok"
            take = miss & (f == "ok")
            d_all[take] = d[take]
            f_all[take] = "ok"
            f_all[miss & ~take & (f_all == "missing")] = f[miss & ~take & (f_all == "missing")]
    if d_all is None:
        d_all = pd.Series(pd.NaT, index=df.index)
        f_all = pd.Series("missing", index=df.index)
    o["date"], o["date_flag"] = d_all, f_all
    # --- convert to reporting currency
    spend = amt.copy()
    o["fx_source"] = np.where(cur == rep, "none", "")
    non = cur != rep
    if non.any():
        rate = _col(df, s.get("rate_col"), np.nan).map(parse_rate)
        sem = s.get("rate_semantics", "units_per_reporting")  # B: units of line currency per 1 USD
        have = non & rate.notna()
        spend[have] = amt[have] / rate[have] if sem == "units_per_reporting" else amt[have] * rate[have]
        o.loc[have, "fx_source"] = "line_rate"
        need = non & rate.isna()
        if need.any() and fx is not None:
            r = _lookup_fx(fx, cur[need], o.loc[need, "date"])
            ok = r.notna()
            idx = r.index[ok]
            spend[idx] = amt[idx] / r[ok]
            o.loc[idx, "fx_source"] = "fx_table"
            o.loc[r.index[~ok], "fx_source"] = "NO_FX"
        elif need.any():
            o.loc[need, "fx_source"] = "NO_FX_ASSUMED_PARITY"   # no rate and no table: keep amount, flag it
    o["spend_calc"] = spend
    # --- keys / text
    o["po"] = _col(df, s.get("po_col"), "").astype(str).str.extract(cfg.get("match", {}).get("po_regex", r"(\d{8,12})"))[0]
    o["invoice"] = _col(df, s.get("invoice_col"), "").astype(str).map(norm_text).str.replace(" ", "")
    o["sup_name"] = _col(df, s.get("supplier_name_col"), "")
    o["vendor_id"] = _col(df, s.get("supplier_id_col"), "").astype(str)
    o["item_code"] = _col(df, s.get("item_code_col"), "").astype(str)
    o["desc"] = _join(df, s.get("description_cols"))
    first = pd.Series("", index=df.index)           # item identity text: first non-empty of item_text_cols
    for c in s.get("item_text_cols", s.get("description_cols", [])):
        first = first.where(first != "", _col(df, c).astype(str).str.strip())
    o["item_desc"] = first
    o["qty"] = _col(df, s.get("qty_col"), np.nan).map(parse_amount)
    o["group_val"] = _col(df, s.get("item_group_col"), "").astype(str)
    o["rule_text"] = _join(df, s.get("rule_cols") or s.get("description_cols"), " | ")
    o["cls_text"] = _join(df, s.get("text_cols") or s.get("description_cols"))
    o["cls_ctx"] = _ctx(df, s.get("context_cols", []))
    o["ctx_excl"] = _ctx(df, s.get("exclusion_context_cols", s.get("context_cols", [])))
    # --- status
    st = _col(df, s.get("status_col"), "").astype(str).str.lower().str.strip()
    flag = pd.Series("", index=df.index)
    if s.get("status_col"):
        ok = set(x.lower() for x in s.get("ok_status", []))
        canc = set(x.lower() for x in s.get("cancelled_status", ["cancelled", "canceled", "void", "voided", "reversed"]))
        flag[st.isin(canc) | st.str.contains(r"cancel|void", na=False)] = "CANCELLED"
        flag[(flag == "") & ~st.isin(ok) & (st != "")] = "NOT_APPROVED"
        flag[(flag == "") & (st == "") & (not s.get("blank_status_ok", True))] = "NOT_APPROVED"
    o["status_flag"] = flag
    return o


def _lookup_fx(fx, cur, dates):
    """fx: DataFrame currency, month(YYYY-MM), units_per_usd. Exact month, else nearest month of that currency."""
    fx = fx.copy()
    fx["p"] = pd.PeriodIndex(fx["month"], freq="M")
    out = pd.Series(np.nan, index=cur.index)
    for c, g in cur.groupby(cur):
        t = fx[fx.currency == c].sort_values("p")
        if t.empty:
            continue
        ords = t["p"].map(lambda p: p.ordinal).to_numpy()
        rates = t["units_per_usd"].astype(float).to_numpy()
        for i in g.index:
            d = dates[i]
            if pd.isna(d):
                out[i] = float(np.median(rates))
                continue
            o = pd.Period(d, freq="M").ordinal
            out[i] = rates[np.abs(ords - o).argmin()]
    return out


# ================================================================= two-system matching
def match_systems(P, S, mcfg, log=print):
    """Greedy 1:1 assignment of secondary (tool) records to primary (ERP) lines.
    tier1: same PO + amount; tier2: amount + same supplier cluster; tier3: amount + similar text."""
    tol = float(mcfg.get("amount_tol", 0.02))
    min_sim = float(mcfg.get("text_sim_min", 0.5))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([P["item_text"], S["item_text"]]))
    XP, XS = vec.transform(P["item_text"]), vec.transform(S["item_text"])
    pa = P["spend_calc"].to_numpy(dtype=float)
    ok = ~np.isnan(pa)
    order = np.argsort(np.where(ok, pa, np.inf))
    pa_sorted = np.where(ok, pa, np.inf)[order]
    taken = np.zeros(len(P), bool)
    p_po = P["po"].to_numpy(object)
    p_sc = P["sup_cluster"].to_numpy(object)
    res_p = np.full(len(S), -1)
    res_t = np.zeros(len(S), int)
    s_amt = S["spend_calc"].to_numpy(dtype=float)
    s_po = S["po"].to_numpy(object)
    s_sc = S["sup_cluster"].to_numpy(object)
    elig = np.where((S["status_flag"] == "") & ~np.isnan(s_amt))[0]
    for tier in (1, 2, 3):
        for i in elig:
            if res_p[i] >= 0:
                continue
            a = s_amt[i]
            w = tol + 1e-6 * abs(a)
            lo, hi = np.searchsorted(pa_sorted, a - w, "left"), np.searchsorted(pa_sorted, a + w, "right")
            cands = [j for j in order[lo:hi] if not taken[j]]
            if not cands:
                continue
            if tier > 1 and abs(a) < float(mcfg.get("min_amount_tier23", 5.0)):
                continue                      # tiny amounts coincide by chance; only trust PO-backed matches
            if tier == 1:
                cands = [j for j in cands if s_po[i] is not np.nan and isinstance(s_po[i], str) and p_po[j] == s_po[i]]
            elif tier == 2:
                cands = [j for j in cands if p_sc[j] == s_sc[i] and s_sc[i]]
            else:
                if a == 0:
                    continue
                sims = [(XP[j].multiply(XS[i]).sum(), j) for j in cands]
                cands = [j for sm, j in sims if sm >= min_sim]
                cands.sort(key=lambda j: -XP[j].multiply(XS[i]).sum())
            if not cands:
                continue
            if tier != 3 and len(cands) > 1:  # prefer same supplier then closest amount
                cands.sort(key=lambda j: (p_sc[j] != s_sc[i], abs(pa[j] - a)))
            j = cands[0]
            taken[j] = True
            res_p[i], res_t[i] = j, tier
    return res_p, res_t


# ================================================================= main run
def run(cfg_path, data_dir=None, out_dir="outputs", out_name=None, hide_labels=False, log=print):
    t0 = time.time()
    cfg = load_config(cfg_path)
    data_dir = data_dir or cfg.get("data_dir", "data")
    company = cfg.get("company", "X")
    mode = cfg.get("mode", "single")
    rep = cfg.get("reporting_currency", "USD")
    fx = _read(data_dir, cfg["fx_file"]) if cfg.get("fx_file") else None
    if fx is not None:
        fx["units_per_usd"] = fx["units_per_usd"].astype(float)

    # ---------------- ingest + standardise
    if mode == "two_system":
        raw_p, raw_s = _read(data_dir, cfg["primary"]["file"]), _read(data_dir, cfg["secondary"]["file"])
        P = standardise(raw_p, cfg["primary"], cfg, fx, log)
        S = standardise(raw_s, cfg["secondary"], cfg, fx, log)
        P["src"], S["src"] = "P", "S"
        raw_p_group = _col(raw_p, cfg["primary"].get("item_group_col"), "")
        log(f"ingested primary={len(P)} secondary={len(S)}")
        A = pd.concat([P, S], ignore_index=True)
    else:
        raw = _read(data_dir, cfg["file"])
        A = standardise(raw, cfg, cfg, fx, log)
        A["src"] = "P"
        if cfg.get("duplicates", {}).get("full_row"):   # identical in every column except the record id
            cols = [c for c in raw.columns if c != cfg["id_col"] and c not in cfg["duplicates"].get("ignore_cols", [])]
            A["dup_key"] = pd.util.hash_pandas_object(raw[cols].fillna("").astype(str), index=False).to_numpy()
        P, S = A, None
        log(f"ingested {len(A)} rows")
    A["spend_calc"] = A["spend_calc"].astype(float)

    # ---------------- suppliers
    scfg = cfg.get("supplier", {})
    A["sup_norm"] = A["sup_name"].map(norm_supplier)
    has_name = (A["sup_norm"] != "").mean() > 0.3
    if has_name:
        cl = cluster_strings(A["sup_norm"], threshold=scfg.get("cluster_threshold", 0.9), ngram=(2, 3))
        A["sup_cluster"] = A["sup_norm"].map(cl).fillna("")
        if scfg.get("unmapped_bucket", False):
            erp = A[(A.src == "P")]
            cov = erp.groupby("sup_cluster")["vendor_id"].apply(lambda s: (s != "").mean())
            unmapped = set(cov[cov < scfg.get("min_vendor_coverage", 0.5)].index)
            A["sup_key"] = np.where(A["sup_cluster"].isin(unmapped), "UNMAPPED_SUPPLIERS", A["sup_cluster"])
            log(f"suppliers: {A.sup_cluster.nunique()} name clusters, {len(unmapped)} unmapped (vendor-id coverage < "
                f"{scfg.get('min_vendor_coverage', 0.5)}) lumped into one bucket")
        else:
            A["sup_key"] = A["sup_cluster"]
    else:
        A["sup_cluster"] = A["vendor_id"].str.upper().str.lstrip("0")
        A["sup_key"] = A["sup_cluster"]
    A.loc[A["sup_key"] == "", "sup_key"] = "UNKNOWN_SUPPLIER"

    # ---------------- items
    icfg = cfg.get("item", {})
    A["item_text"] = A["item_desc"].map(norm_item)
    for pat, rep_ in icfg.get("strip_patterns", []):
        A["item_text"] = A["item_text"].str.replace(pat, rep_, regex=True)
    A["item_text"] = A["item_text"].str.strip()
    A["generic"] = A["item_text"].isin(GENERIC_DESCRIPTIONS) | (A["item_text"].str.len() < 3)
    master = pd.Series(False, index=A.index)
    mw = icfg.get("master_when")
    if mw and mode == "two_system":
        gv = pd.concat([_col(raw_p, mw["column"], ""), pd.Series("", index=S.index)], ignore_index=True)
        master = gv.isin(mw["values"]) & (A.src == "P").to_numpy()
    elif mw:
        master = _col(raw, mw["column"], "").isin(mw["values"])
    elif icfg.get("use_item_code", True):
        master = A["item_code"] != ""
    A["is_master"] = master.to_numpy()
    txt_cl = cluster_strings(A.loc[~A.generic, "item_text"], threshold=icfg.get("cluster_threshold", 0.93), ngram=(2, 4))
    A["item_key"] = np.where(A["is_master"] & (A["item_code"] != ""), "M:" + A["item_code"],
                             np.where(A["generic"], "UNSPECIFIED", "T:" + A["item_text"].map(txt_cl).fillna("")))
    uw = icfg.get("unmapped_when")                  # lines with no stable item identity (free text / catalogue)
    if uw and mode == "two_system":
        um = (A["group_val"].isin(uw["values"]) & (A.src == "P")).to_numpy()
        A.loc[um, "item_key"] = "UNMAPPED_ITEMS"
    if mode == "two_system":  # secondary rows: adopt nearest primary item if similar enough
        pm = A[(A.src == "P") & ~A.generic]
        sm = A[(A.src == "S") & ~A.generic]
        if len(pm) and len(sm):
            u_p = pm.drop_duplicates("item_text")
            v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
            v.fit(pd.concat([u_p["item_text"], sm["item_text"]]))
            XP = v.transform(u_p["item_text"])
            u_s = sm.drop_duplicates("item_text")
            XS = v.transform(u_s["item_text"])
            best, bsim = [], []
            for i in range(0, XS.shape[0], 1000):
                M = (XS[i:i + 1000] @ XP.T).tocsr()
                for r in range(M.shape[0]):
                    row = M.getrow(r)
                    if row.nnz:
                        k = row.data.argmax()
                        best.append(row.indices[k]); bsim.append(row.data[k])
                    else:
                        best.append(-1); bsim.append(0.0)
            best, bsim = np.array(best), np.array(bsim)
            from .dedupe import numbers_of
            mp = {}
            ptexts = u_p["item_text"].to_numpy(); pkeys = u_p["item_key"].to_numpy()
            for k, (t, b, sc_) in enumerate(zip(u_s["item_text"], best, bsim)):
                if b >= 0 and sc_ >= icfg.get("adopt_threshold", 0.85) and numbers_of(t) == numbers_of(ptexts[b]):
                    mp[t] = pkeys[b]
            adopt = A.index[(A.src == "S") & A.item_text.isin(mp.keys())]
            mp = {k: v for k, v in mp.items()}
            A.loc[adopt, "item_key"] = A.loc[adopt, "item_text"].map(mp)

    # ---------------- exclusions: status, rules, learned
    A["excl"] = A["status_flag"].copy()
    rules = compile_rules(cfg.get("exclusions"))
    code, why = apply_rules(A["rule_text"], rules)
    m = (A["excl"] == "") & (code != "")
    A.loc[m, "excl"] = code[m]
    A["excl_why"] = np.where(A["status_flag"] != "", "status", why)
    A["excl_conf"] = np.where(A["excl"] != "", 0.95, 1.0)
    ecfg = cfg.get("exclusions", {}) or {}
    labels_dir = data_dir
    if ecfg.get("learn_from") and not hide_labels:
        L = ecfg["learn_from"]
        lab = _read(labels_dir, L["file"])
        lab = lab.rename(columns={L["id_col"]: "record_id"})
        lab["y"] = (lab[L["include_col"]].astype(str).isin(["0", "0.0"])).astype(int)
        j = A[["record_id", "cls_text", "rule_text", "ctx_excl"]].merge(lab[["record_id", "y"]], on="record_id") \
            if "ctx_excl" in A.columns else None
        if j is not None and len(j):
            le = LearnedExclusion(min_prob=ecfg.get("min_prob", 0.6)).fit(j["ctx_excl"] + " " + j["rule_text"], j["y"])
            p = le.predict_proba(A["ctx_excl"] + " " + A["rule_text"])
            m = (A["excl"] == "") & (p >= le.min_prob)
            A.loc[m, "excl"] = "NON_PROCUREMENT"
            A.loc[m, "excl_why"] = "learned"
            A.loc[m, "excl_conf"] = np.clip(p[m], 0.5, 0.99)
            log(f"learned exclusion model flagged {int(m.sum())} extra rows")

    # ---------------- two-system matching / duplicates
    A["match_tier"] = 0
    A["match_to"] = ""
    if mode == "two_system":
        Pp = A[A.src == "P"].reset_index()
        Ss = A[A.src == "S"].reset_index()
        Ss["status_flag"] = np.where(Ss["excl"] != "", Ss["excl"], "")  # any exclusion blocks matching
        res_p, res_t = match_systems(Pp, Ss, cfg.get("match", {}), log)
        mt = res_p >= 0
        idx_s = Ss.loc[mt, "index"].to_numpy()
        A.loc[idx_s, "excl"] = "DUPLICATE"
        A.loc[idx_s, "excl_why"] = "matched_to_primary"
        A.loc[idx_s, "match_tier"] = res_t[mt]
        A.loc[idx_s, "match_to"] = Pp.loc[res_p[mt], "record_id"].to_numpy()
        A.loc[idx_s, "excl_conf"] = np.select([res_t[mt] == 1, res_t[mt] == 2], [0.98, 0.9], 0.75)
        for k_, c_ in (("sup_key", "sup_key"), ("item_key", "item_key")):
            A.loc[idx_s, k_] = Pp.loc[res_p[mt], c_].to_numpy()
        # tool invoice exported more than once: identical key among remaining included secondary rows
        rem = A[(A.src == "S") & (A.excl == "")]
        key = rem["invoice"] + "|" + rem["po"].fillna("") + "|" + rem["sup_cluster"] + "|" + rem["item_text"] + "|" + \
            rem["spend_calc"].round(2).astype(str)
        dup = key.duplicated(keep="first")
        A.loc[dup[dup].index, ["excl", "excl_why", "excl_conf"]] = ["DUPLICATE", "export_duplicate", 0.9]
        log(f"matching: tier counts {dict(pd.Series(res_t[mt]).value_counts().sort_index())}; "
            f"secondary-only included={int(((A.src == 'S') & (A.excl == '')).sum())}; "
            f"export duplicates={int(dup.sum())}")
    else:
        if "dup_key" in A.columns:
            dup = A["dup_key"].duplicated(keep="first") & (A["excl"] == "")
            A.loc[dup, ["excl", "excl_why", "excl_conf"]] = ["DUPLICATE", "exact_duplicate_row", 0.9]
            log(f"exact duplicate rows flagged: {int(dup.sum())}")

    A["include"] = (A["excl"] == "").astype(int)
    A["spend"] = np.where(A["include"] == 1, A["spend_calc"].fillna(0.0), 0.0).round(2)

    # ---------------- classification (included, non-duplicate rows only)
    tcfg = cfg.get("classify", {})
    tree = clf.Tree(_read(data_dir, tcfg["taxonomy"]))
    inc = A[A.include == 1].copy()
    labels = None
    if tcfg.get("training_labels") and not hide_labels:
        L = tcfg["training_labels"]
        lab = _read(data_dir, L["file"]).rename(columns={L["id_col"]: "record_id"})
        if L.get("include_col") and L["include_col"] in lab.columns:
            lab = lab[lab[L["include_col"]].astype(str).isin(["1", "1.0"])]
        cols = L["level_cols"]
        lab["leaf"] = [tree.encode(list(r)) for r in lab[cols].itertuples(index=False, name=None)]
        lm = dict(zip(lab["record_id"], lab["leaf"]))
        labels = inc["record_id"].map(lm).fillna(-1).astype(int)
        log(f"training labels usable: {(labels >= 0).sum()}")
    c2 = dict(tcfg)
    c2["cache_name"] = company
    sup_hint = inc["sup_name"].fillna("")
    inc["llm_text"] = (inc["cls_text"] + np.where(sup_hint != "", " [supplier: " + sup_hint + "]", "")).str.strip()
    inc["cls_text"] = np.where(inc["cls_text"].str.strip() == "", "", inc["cls_text"])
    if tcfg.get("add_supplier_to_text", True) and has_name:
        inc["cls_ctx"] = (inc["cls_ctx"].fillna("") + " SUP_" + inc["sup_key"].str.replace(" ", "_")).str.strip()
    leaf, conf, method = clf.classify(tree, inc, c2, labels, cache_dir=os.path.join(out_dir, ".cache"), log=log)
    L4 = [tree.levels(i) for i in leaf]
    A["l1"] = A["l2"] = A["l3"] = A["l4"] = ""
    for k in range(4):
        A.loc[inc.index, f"l{k + 1}"] = [r[k] for r in L4]
    A["conf"] = np.where(A.include == 1, 0.0, A["excl_conf"])
    A.loc[inc.index, "conf"] = conf.round(3).to_numpy()
    A["exclusion_reason"] = A["excl"]

    out = A.rename(columns={"sup_key": "supplier_key", "conf": "confidence"})
    out = out[["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
               "l1", "l2", "l3", "l4", "confidence"]]
    os.makedirs(out_dir, exist_ok=True)
    name = out_name or company
    path = os.path.join(out_dir, f"{name}.csv")
    out.to_csv(path, index=False)
    audit = A[["record_id", "src", "date", "date_flag", "currency", "fx_source", "amount_orig", "spend_calc",
               "match_tier", "match_to", "excl_why"]]
    audit.to_csv(os.path.join(out_dir, f"{name}__audit.csv"), index=False)
    secs = time.time() - t0
    log(f"wrote {path}: {len(out)} rows, included={int(out.include.sum())}, spend={out.spend.sum():,.0f} {rep}, "
        f"time={secs:.0f}s, classifier={method}")
    return out, A, tree, dict(seconds=secs, method=method)
