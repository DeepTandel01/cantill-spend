#!/usr/bin/env bash
# Blind-test day helper.  NO CODE CHANGES after 10:00 -- this script only runs the frozen tool.
#   NAME=yourname bash scripts/blind_test.sh  data/blind          # folder holding A2, B2 and C files
# It copies the support files (taxonomies, labels, seed labels, fx) into the blind folder if missing, runs A2, B2
# (existing configs) and C (configs/company_C.yaml that YOU write on the day), and puts results in submission/.
set -euo pipefail
NAME="${NAME:?set NAME=yourname}"
B="${1:?folder with the blind-test files}"
for f in taxonomy_A.csv labels_A.csv taxonomy_B.csv seed_labels.csv fx_rates_usd.csv; do
  [ -f "$B/$f" ] || cp "data/$f" "$B/$f"
done
mkdir -p submission
[ -f "$B/erp_goods_receipts.csv" ] && python run.py --config configs/company_A.yaml --data-dir "$B" --out-dir submission --out-name "${NAME}__A2"
[ -f "$B/invoice_lines.csv" ]      && python run.py --config configs/company_B.yaml --data-dir "$B" --out-dir submission --out-name "${NAME}__B2"
[ -f configs/company_C.yaml ]      && python run.py --config configs/company_C.yaml --data-dir "$B" --out-dir submission --out-name "${NAME}__C"
cp configs/company_C.yaml submission/ 2>/dev/null || true
rm -f submission/*__audit.csv
ls -la submission
