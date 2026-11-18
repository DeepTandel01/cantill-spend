# spendcube: spend data pipeline + classifier (Cantill.ai Round 1)

Takes a company's raw purchase data and category tree, combines systems, cleans amounts/dates, decides what is
spend, de-duplicates suppliers and items, and classifies every line into the tree. **A new company = a new config
file, not new code.**

## Setup (VS Code, any OS)
```bash
git clone <your repo> && cd cantill-spend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir data                                                # put the task CSVs here (data/ is git-ignored)
```
Data files needed in `data/`: `erp_goods_receipts.csv invoice_tool_export.csv labels_A.csv taxonomy_A.csv
invoice_lines.csv seed_labels.csv taxonomy_B.csv fx_rates_usd.csv`.

## The one command
```bash
python run.py --config configs/company_A.yaml --out-name yourname__A     # -> outputs/yourname__A.csv
python run.py --config configs/company_B.yaml --out-name yourname__B
```
Other flags: `--data-dir DIR`, `--out-dir DIR`, `--hide-labels` (zero-label test).
Runtime: ~30 s for A (65k rows), ~11 s for B (41k rows) on a laptop, supervised mode.

## Layout
```
run.py                      CLI entry point
configs/company_A.yaml      two-system company (ERP + AP tool)
configs/company_B.yaml      single file, 25 currencies, vendor IDs
configs/company_C.template.yaml   starting point for a new company
spendcube/utils.py          amount / date / VAT / name parsing
spendcube/dedupe.py         fuzzy clustering (char n-gram TF-IDF + union-find, numeric guard)
spendcube/exclusions.py     keyword rules + learned exclusion model
spendcube/classify.py       tree loader; supervised | llm | embedding | lexical classifiers
spendcube/pipeline.py       ingest -> match -> clean -> exclude -> dedupe -> classify -> write
scripts/evaluate_A.py       dedup F1 + zero-label accuracy + CV on A
scripts/evaluate_B.py       exclusion check + CV on B seeds
scripts/suggest_config.py   drafts a config for a new company from column names
scripts/blind_test.sh       blind-test day runner
tests/test_basic.py         unit tests (no network)
```

## Classifier modes (`classify.method`, default `auto`)
| mode | needs | when |
|---|---|---|
| supervised | labels file in config | A and B (labels shipped) |
| llm | `pip install anthropic` + `ANTHROPIC_API_KEY` | unseen tree (company C), best accuracy |
| embedding | `pip install sentence-transformers` (model downloads once) | no key, better than lexical |
| lexical | nothing | always-available fallback |

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
export SPEND_LLM_MODEL=claude-haiku-5-5  # optional; default shown
```
Without a key the tool still runs end to end using the lexical fallback (much lower accuracy on unseen trees).

## Blind-test day (Thu 19 Nov)
1. 10:00: save A2/B2/C files in `data/blind/` (same file names as before; if names differ, edit the `file:` lines
   in the configs - config edits are not code).
2. Write `configs/company_C.yaml`: `python scripts/suggest_config.py data/blind/<c_file>.csv data/blind/<c_tree>.csv > configs/company_C.yaml`, then fix the column names by hand.
3. `export ANTHROPIC_API_KEY=...` then `NAME=yourname bash scripts/blind_test.sh data/blind`
4. `git add submission && git commit -m "blind test" && git push` before 12:00.
Do not change any `.py` file after 10:00.
