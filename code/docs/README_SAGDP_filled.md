# SAGDP compensation + GOS, suppressed cells filled — 2022-2024

Built 2026-09-19. Source vintage: SAGDP2-7 `__ALL_AREAS_1997_2024/2025` (new-vintage, no "N" suffix).

## Files
- `sagdp_comp_gos_filled_2022_2024.csv` — one row per year x region x SAGDP LineCode.
  Regions: 00000 (US **domestic** — overseas lines 100-103 netted out so US = sum of states),
  91000-98000 BEA regions, 50 states + DC. Lines 1-86 solved; 87-92 recomputed from filled
  leaves (`status = derived`). All money in **thousands of current dollars** (GDP converted from millions).
  Columns: `status` (published / D), `gdp_note` ((L) handling), `comp_published`, `comp_proxy_sainc`, `comp_filled`,
  `gos_published`, `gos_identity`, `gos_filled`, `gdp`, `taxes_less_subsidies`.
- `sagdp_solver_diagnostics_2022_2024.csv`, `sagdp_mask_backtest_scores.csv`, `sagdp_mask_backtest_cells.csv`
- `sagdp_fill_code/` — `prep.py`, `run.py`, `validate.py`. Engine imported from
  `raw/CAINC/filled/bea_map_code/mapsolve.py` (the county compensation fill).

## Method
1. **Compensation (SAGDP4).** Proxy = BEA personal-income compensation by place of work
   (SAINC6N/CAINC6N concept), from the already-filled `bea_comp_earn_county_{year}_filled` state rows,
   mapped to SAGDP lines. On published state cells SAGDP4 / proxy has median 1.0000
   (p10 0.9990, p90 1.00003); lines 61-63 have no counterpart (proxy 0, RAS-form seed).
   Row-normalised exact-MAP LSQR, published cells hard-pinned, non-negative, swept over
   (sector level x region level), then a full-tree polish pass.
2. **GOS (SAGDP7).** Every suppressed SAGDP4 cell is also suppressed in SAGDP7; SAGDP2 and SAGDP3 are
   fully published. So GOS = GDP x 1000 − taxes less subsidies − compensation, then a MAP pass
   (negatives allowed — 188 published GOS cells are negative) to restore exact additivity
   against published GOS despite GDP's rounding to millions.

## Results
Cells filled: 407 (2022), 413 (2023), 446 (2024) in lines 1-86.
Adding-up: largest remaining residual 80–95 thousand $ (comp) and 12–35 thousand $ (GOS); BEA's own
published slack is up to 6 thousand $.

Masking backtest (10% of published state cells, size-matched to suppressed population, 5 reps/year):

| year | comp median / p90 / weighted APE | GOS median / p90 / weighted APE | SAINC as-is comp p90 |
|---|---|---|---|
| 2022 | 0.000% / 0.09% / 0.02% | 0.004% / 0.34% / 0.02% | 0.79% |
| 2023 | 0.001% / 0.09% / 0.02% | 0.004% / 0.24% / 0.01% | 0.64% |
| 2024 | 0.000% / 0.05% / 0.01% | 0.003% / 0.20% / 0.01% | 0.61% |

Worst lines (p90 1-2%): 38 rail, 16 primary metals, 22 other transp. equip., 42 pipelines, 49 data processing.

## Caveats
- Lines 46 (publishing) and 48 (broadcasting + telecom) are suppressed in **all 51 states** in 2024,
  so their state split is entirely the SAINC proxy's; the backtest cannot score them.
- 28–35 filled GOS cells per year are negative (largest: Rocky Mountain region, insurance, 2022, −$888M).
  Negative GOS is published elsewhere, but check these before use.
- Masked backtest scores cells scattered at random; true (D) cells cluster in the lines where SAGDP and
  SAINC diverge most, so treat the scores as optimistic.

## Follow-up checks (2026-09-19, second pass)

**(L) cells in SAGDP2** ("less than $50,000"). Five across 2022-24. Where compensation and GOS are both
published, GDP = comp + taxes + GOS exactly (HI pipelines 2022 = $38k; WI oil & gas 2024 = $21k).
Otherwise GDP is set to the $25k midpoint (DC pipelines 2023 and 2024, VT mining support 2023) and
feeds the GOS identity. Flagged in `gdp_note`.

**Negative filled GOS** — `sagdp_negative_gos_review.csv`. 2022: 29, 2023: 35, 2024: 27.
- 81 of 91 are forced by BEA's own published numbers: published GDP minus published taxes is already
  smaller than SAINC compensation for that cell, so GOS < 0 whatever the solver does
  (all 27 in 2024 are of this kind). 53 of the 91 cells have published negative GOS in earlier years.
  The largest, Rocky Mountain insurance 2022 (−$888M), was negative in 10 of 26 published years.
- 10 (6 in 2022, 4 in 2023, none in 2024) are the solver's doing: comp pushed above the SAINC proxy to
  satisfy adding-up. All small; the largest is NM motor vehicles 2022 (−$1.35M).

**Identity comp + taxes + GOS = GDP.** BEA's own published cells miss it by median $27k, p99 $238k,
max $2.0M (GDP is published in millions to one decimal). Filled cells: median $36k, p99 $2.0M, max $2.4M
— the same size, because the residual the GOS pass places on suppressed cells is that rounding.

**Against the summary Use table** — `sagdp_vs_sut_{2022,2023,2024}.csv`. All 64 bridged lines
(SAGDP leaf lines to the 71 SUT industries; federal = GFGD+GFGN+GFE vs lines 84+85, state & local =
GSLG+GSLE) agree within $1M for compensation (V001), GOS (V003), value added (VAPRO) and taxes less
subsidies (VAPRO − V001 − V003) in 2024 and 2023; within $3M in 2022. Filled state sums (plus overseas)
match the SUT within $1.1M. SAGDP and the SUT are the same BEA vintage, and the line-to-industry bridge
is exact.
