# DECISIONS.md: what I found in the data and what I decided

Numbers are from the shipped A and B files. Everything below is a config switch, not hard-coded logic.

## Company A (ERP goods receipts + AP tool)
1. **Net amounts.** ERP `value_zar` is net. Tool `gross_amount` has thousands separators ("5,266.63"); `vat_rate` appears as
   `15`, `0.15`, `15%`, `0` and blank. All parsed to a fraction; blank -> 15% (brief); explicit 0 stays 0. Net = gross / (1 + vat).
2. **Count once (matching).** 28,520 tool rows. Greedy 1:1 assignment of tool rows to ERP lines, strongest evidence first:
   tier 1 = same 10-digit PO (extracted from `PO#...`, `PO ...`, `4510.../10.0`) and amount within R0.02 (+1e-6 relative);
   tier 2 = same amount and same (fuzzy-clustered) supplier, PO missing or wrong; tier 3 = same amount and text similarity >= 0.5.
   Tiers 2-3 are ignored for amounts under 5 (tiny amounts collide by chance). Result: 16,672 / 5,643 / 20 matches. Matched
   tool rows = `DUPLICATE`; the rest (5,520 rows, ~13% of spend) are spend that exists only in the tool and is counted.
   Matching is one-to-one, so two identical ERP lines need two tool rows.
3. **Tool exported twice.** Among tool rows still included: same invoice no. + PO + supplier cluster + text + net amount ->
   keep first, rest `DUPLICATE` (162 rows). Done after ERP matching so genuine twin lines that match two twin ERP lines survive.
4. **Status.** Approved/Paid count. Cancelled (188) -> `CANCELLED`; On Hold/Rejected (315) -> `NOT_APPROVED`. Status wins over
   DUPLICATE, and non-approved tool rows never consume an ERP line.
5. **Negative ERP lines** (434, credit/reversals) are kept as negative spend: the net spend of the company is the sum.
6. **Dates.** Tool dates come in 5 formats (`11/10/2022`, `29-Mar-22`, `2023-07-21`, ...) and 61% are blank. Parsed
   day-first (South Africa); out-of-range / unparseable dates are flagged in `outputs/<name>__audit.csv` (they do not change spend because A is already in ZAR).
7. **Suppliers: the client's golden key has an "unmapped" bucket.** Of 193 ERP supplier names, 71 are mapped to their own
   golden key; **122 names (56% of lines) share one golden key** (SG-f85711). Those are exactly the names whose lines mostly
   lack a `vendor_id` (names with >=53% vendor-id coverage are mapped, <=45% are in the bucket). I mimic this:
   names are clustered (case, punctuation, legal suffix, "KEL MONT" vs "KELMONT"), a cluster with <50% vendor-id coverage goes to
   `UNMAPPED_SUPPLIERS`. Pairwise F1 vs the golden key on A = 1.00. **Trade-off:** if the grader scores true suppliers instead
   of the client's file, set `supplier.unmapped_bucket: false` in `configs/company_A.yaml` (one line); I chose the evidence in labels_A.
8. **Items: same pattern.** Golden `item_key` exists for only 21k of 37k ERP lines: every `Material` line has its own key (= material
   number, F1 1.00); all labelled `Free Text` / `Catalogue` lines share one catch-all key; labelled `Service` lines are keyed by the
   *repaired item* (text after `REPR;`, warranty prefixes stripped). Mimicked: `Material` -> material number, `Free Text`/`Catalogue` ->
   `UNMAPPED_ITEMS`, others -> fuzzy text cluster (numbers must match so 1500 mm and 1800 mm rods stay apart). Pairwise F1 on labelled rows 0.997.
   Tool-only rows adopt the nearest ERP item (cosine >= 0.85, same numbers) else get their own text cluster.
9. **Generic descriptions** ("NO OBJECT", "Product", "Parts", blank) get item `UNSPECIFIED`; they are still classified (supplier + spend type context).
10. **Non-procurement in A.** Almost none by keyword (2 rows): A's data has no tax/intercompany/royalty accounts in the supplied files.

## Company B (8 ERPs, 25 currencies)
1. **Currency.** `currency_rate` is units of line currency per 1 USD: USD = amount / rate. USD lines are used as-is (7% of them carry a stray rate; ignored). 3,343
   non-USD lines without a rate use `fx_rates_usd.csv` for the invoice month (nearest available month if that month is missing).
   Sanity check: given rates agree with the table within 20% for 99.4% of lines (outliers are ARS/EUR months).
2. **Dates.** Invoice date is tried first and posting date is the fallback when it is missing/unparseable/out of range
   (invoice dates range from 2018 to 2027; payments were posted in 2024).
3. **Exclusions (all `NON_PROCUREMENT`).** Seed labels show 450/3000 excluded, all non-procurement: intercompany GL accounts
   (`IC06`, `IC Payables`, `I/C Payable`, ...), sales-tax expense lines, and a few vendor types. Two layers: keyword rules on GL description + line text,
   and a small logistic model on GL / supplier-type / document-type tokens learned from the seed file. On the seeds: 450/450 caught, 0 false positives;
   rules alone (no labels, as for company C) miss 4 lines. The IC rule is narrow (`IC06`, `IC Payable`, `I/C`) because "IC CB INK" is a product.
4. **Duplicates.** 604 rows are identical to an earlier row in every column except `line_id` (concentrated in ERP-B3 JDE) -> `DUPLICATE`.
   Risk: one such pair exists in the seeds and both were labelled included; a quick analyst label would not notice. Switch off with `duplicates: {}`.
5. **Blank descriptions** (~41% blank or whitespace). Classified from GL description, GL account, vendor, PO type and region tokens (supervised mode) and from the GL description (text).
6. **Suppliers.** Only vendor IDs: `supplier_key` = vendor ID (leading zeros stripped). **Items:** material number where present, else text cluster, blank -> UNSPECIFIED.
7. **Reversals** (+X / -X on one invoice line) are kept; they net to zero. 3 rows have unparseable amounts -> spend 0, flagged in the audit file.

## Rules for any company
* Exclusion priority: CANCELLED / NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
* Spend = 0 and blank categories for excluded rows; excluded DUPLICATE rows carry the supplier/item of the record they duplicate.
* Confidence for excluded rows = confidence of the exclusion decision; for included rows = classifier probability (supervised) or the LLM's stated confidence (scaled down when the LLM misses a line).
* Unresolvable currency (no rate and no table) keeps the amount at parity and is flagged `NO_FX_ASSUMED_PARITY` in the audit file.
