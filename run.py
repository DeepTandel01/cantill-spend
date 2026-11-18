#!/usr/bin/env python3
"""One command:  python run.py --config configs/company_A.yaml --out-name yourname__A"""
import argparse
from spendcube.pipeline import run

ap = argparse.ArgumentParser(description="Spend cube pipeline")
ap.add_argument("--config", required=True)
ap.add_argument("--data-dir", default=None, help="folder with the company's csv files (default: config data_dir)")
ap.add_argument("--out-dir", default="outputs")
ap.add_argument("--out-name", default=None, help="output file stem, e.g. yourname__A2")
ap.add_argument("--hide-labels", action="store_true", help="ignore any labels (zero-label test)")
a = ap.parse_args()
run(a.config, a.data_dir, a.out_dir, a.out_name, a.hide_labels)
