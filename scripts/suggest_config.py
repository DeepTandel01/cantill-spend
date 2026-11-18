#!/usr/bin/env python3
"""Draft a config for a NEW single-file company by guessing columns from names and values.
   python scripts/suggest_config.py data/c_invoices.csv data/c_taxonomy.csv --name C > configs/company_C.yaml
ALWAYS read the draft and fix column names by hand (takes 2 minutes); this only saves typing."""
import argparse, re, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("file"); ap.add_argument("taxonomy"); ap.add_argument("--name", default="C")
ap.add_argument("--currency", default="USD")
a = ap.parse_args()
df = pd.read_csv(a.file, dtype=str, nrows=5000)
cols = list(df.columns)
def pick(pats, pool=None, exclude=()):
    for p in pats:
        for c in (pool or cols):
            if c in exclude: continue
            if re.search(p, c, re.I): return c
    return None
num_cols = [c for c in cols if pd.to_numeric(df[c].str.replace(",", ""), errors="coerce").notna().mean() > .8]
idc = pick([r"^(line_?id|record_?id|id)$", r"line.*id", r"_id$|^id"])
amt = pick([r"net.*(amount|value)", r"(amount|value|spend|total|cost|price)"], num_cols)
cur = pick([r"^currency$", r"curr"])
rate = pick([r"rate|fx"], num_cols)
dt = [c for c in cols if re.search(r"date", c, re.I)]
sup_n = pick([r"(supplier|vendor).*(name|desc)"], exclude=(idc,))
sup_i = pick([r"supplier.*id|vendor.*id|vendor.*(no|code)|supplier.*(no|code)"], exclude=(sup_n,))
desc = [c for c in cols if re.search(r"desc|text|item.*name|material.*name|product|memo|narration|line.*text", c, re.I) and c not in (sup_n, sup_i)]
item = pick([r"material.*(no|number|code)|item.*(code|no|number)|sku|part"], exclude=tuple(desc))
status = pick([r"status|state"])
gl = [c for c in cols if re.search(r"gl_|account|cost.?cent|category|commodity|uom|plant", c, re.I) and c not in desc]
tax_basis = "gross" if pick([r"gross|incl"], num_cols) == amt else "net"
N = lambda v: "null" if v is None else v
print(f"""company: {a.name}
mode: single
reporting_currency: {a.currency}
data_dir: data
# fx_file: fx_rates.csv        # only if the data has non-{a.currency} lines without a usable rate

file: {a.file.split('/')[-1]}
id_col: {N(idc)}
amount_col: {N(amt)}
amount_basis: {tax_basis}   # gross -> also set vat_col / default_vat
currency_col: {N(cur)}
rate_col: {N(rate)}
rate_semantics: units_per_reporting   # or reporting_per_unit
date_cols: {dt[:2]}
supplier_name_col: {N(sup_n)}
supplier_id_col: {N(sup_i)}
item_code_col: {N(item)}
description_cols: {desc[:2]}
text_cols: {desc[:2]}
context_cols: {gl[:4]}
rule_cols: {desc[:2] + gl[:2]}
{'status_col: ' + status if status else '# status_col: (none found)'}
# ok_status: [approved, paid]
duplicates: {{full_row: true}}
item: {{use_item_code: true, cluster_threshold: 0.9}}
supplier: {{cluster_threshold: 0.9}}
exclusions: {{use_default: true}}
classify:
  method: auto            # llm if ANTHROPIC_API_KEY is set, else embedding/lexical fallback
  taxonomy: {a.taxonomy.split('/')[-1]}
  add_supplier_to_text: true
""")
