# spendcube: spend data pipeline and classifier

Cantill.ai Data & ML Intern Task, Round 1. Author: DeepTandel.

Takes a company's raw purchase data and category tree, combines systems, cleans amounts and dates, decides what counts as spend, de-duplicates suppliers and items, and classifies every line into the tree. A new company needs a new config file, not new code.

## Setup

Requires Python 3.10+.

```bash
git clone <this repo> && cd cantill-spend
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\activate
pip install -r requirements.txt
mkdir data                           # put the task CSVs here (git-ignored, never committed)
```

## The one command

```bash
python run.py --config configs/company_A.yaml --out-name DeepTandel__A
```

Output: `outputs/DeepTandel__A.csv` (one row per input record, in the required format).
For another company, use its config:

```bash
python run.py --config configs/company_B.yaml --out-name DeepTandel__B
python run.py --config configs/company_C.yaml --out-name DeepTandel__C
```

## Running without or with an API key

The tool runs end to end with no key, using a local fallback classifier.
For an unseen tree, an optional LLM classifier gives better accuracy:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

Estimated cost: about $0.1 to $0.3 per 1,000 lines (an estimate, not measured; real token use is printed on each run).

## Tools used

- AI coding assistant: Claude (Anthropic) helped write the code and documentation.
- Optional LLM inside the tool: Claude via the Anthropic API (`claude-haiku-5-5`), only when a key is set.

## Documentation

- `APPROACH.md`: pipeline design, method, measured accuracy, runtime, cost.
- `DECISIONS.md`: what I found in the data and what I decided.
