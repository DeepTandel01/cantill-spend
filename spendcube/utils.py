"""Parsing + normalisation helpers (amounts, dates, names, descriptions)."""
import re
import unicodedata

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- amounts
def parse_amount(x):
    """'1,234.56' '1.234,56' '(12.30)' '12.30-' ' 5 000,10' -> float. NaN if unparseable."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    t = str(x).strip()
    if not t or t.lower() in ("nan", "none", "null", "-", "n/a"):
        return np.nan
    neg = t.startswith("-") or t.startswith("(") or t.endswith("-") or t.endswith(")")
    t = re.sub(r"[^\d,.]", "", t)
    if not t:
        return np.nan
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        thou = "." if dec == "," else ","
        t = t.replace(thou, "").replace(dec, ".")
    elif "," in t:
        head, tail = t.rsplit(",", 1)
        if t.count(",") == 1 and len(tail) != 3:
            t = head + "." + tail  # decimal comma
        else:
            t = t.replace(",", "")  # thousands
    elif t.count(".") > 1:
        t = t.replace(".", "")
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_amounts(s):
    return s.map(parse_amount)


def parse_rate(x):
    v = parse_amount(x)
    return v if (v == v and v > 0) else np.nan


def parse_vat(x, default=0.15):
    """'15' '15%' '0.15' '0' '' -> fraction. Blank/unparseable -> default."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
        return default
    v = parse_amount(str(x).replace("%", ""))
    if v != v:
        return default
    return v / 100.0 if v > 1 else v


# ---------------------------------------------------------------- dates
DEFAULT_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%b-%y", "%d-%b-%Y", "%d/%b/%Y",
                   "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%Y%m%d",
                   "%d/%m/%y", "%m/%d/%y", "%b %d, %Y", "%d %b %Y", "%d %B %Y", "%B %d, %Y"]


def parse_dates(s, formats=None, dayfirst=True, min_year=2000, max_year=2035):
    """Return (datetime Series, flag Series). flag: ok | missing | unparseable | out_of_range."""
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = s.notna() & (s != "") & (s.str.lower() != "nan")
    fmts = list(formats or DEFAULT_FORMATS)
    if not dayfirst:  # month-first formats tried first
        fmts = sorted(fmts, key=lambda f: 0 if f.startswith("%m") else 1)
    for f in fmts:
        idx = (todo & out.isna()).fillna(False).to_numpy()
        if not idx.any():
            break
        try:
            p = pd.to_datetime(s[idx], format=f, errors="coerce")
        except Exception:
            continue
        out[idx] = p.astype("datetime64[ns]")
    flag = pd.Series("ok", index=s.index)
    flag[~todo.fillna(False)] = "missing"
    flag[todo.fillna(False) & out.isna()] = "unparseable"
    bad = out.notna() & ((out.dt.year < min_year) | (out.dt.year > max_year))
    flag[bad] = "out_of_range"
    return out, flag


# ---------------------------------------------------------------- text
def fold(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = fold(s).upper()
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


LEGAL = {"PTY", "LTD", "LIMITED", "CC", "SA", "GMBH", "INC", "LLC", "CO", "COMPANY", "CORP", "CORPORATION",
         "BV", "SRL", "SPA", "AG", "SAS", "PLC", "LP", "PVT", "SDN", "BHD", "OOO", "AB", "AS", "OY", "KG", "NV",
         "SARL", "SL", "DOO", "SP", "ZOO", "SAB", "DE", "CV", "THE", "AND", "OF", "LLP", "PRIVATE", "SRO", "KFT",
         "LTDA", "EIRELI", "SAC", "UAB", "ASA"}


def norm_supplier(s):
    """Upper, ascii, drop punctuation + legal suffixes, close up spaces ('KEL MONT (PTY) LTD' -> 'KELMONT')."""
    t = norm_text(s)
    toks = [w for w in t.split() if w not in LEGAL]
    return "".join(toks) if toks else t.replace(" ", "")


def norm_item(s):
    """Normalise a description for item matching: glue '22 KW' -> '22KW', drop punctuation."""
    t = norm_text(s)
    t = re.sub(r"(\d)\s+([A-Z]{1,3})\b", r"\1\2", t)
    t = re.sub(r"\b(P N|PN)\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


GENERIC_DESCRIPTIONS = {"", "PRODUCT", "PARTS", "PART", "ITEM", "ITEMS", "MISC", "MISCELLANEOUS", "NO OBJECT",
                        "NOOBJECT", "N0 OBJECT", "NO OBJCCT", "GOODS", "SERVICE", "SERVICES", "OTHER", "NA", "N A",
                        "TBA", "TBC", "NONE", "NULL", "GENERAL", "MATERIAL", "MATERIALS", "SPARES"}
