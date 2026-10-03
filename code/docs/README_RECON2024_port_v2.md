# RECON2024 — porting the RECON2022 package to 2024

**Started 2026-09-20.** `code/` is João Rodrigues' RECON2022 package (from
`RECON2022.zip`, commit b8c6282) renamed `recon2024`, with the base directory and
reference year moved into `stages.yaml` and every `pipeline.yaml` reading `{year}` /
`{prev_year}` tokens. Inputs are read from `..\raw\`; this package writes only under
`RECON2024\process\`, `RECON2024\results\`, `RECON2024\logs\`.
`reference_2022\` holds João's 2022 `process\`, `combine\` and sample `results\` for
step-by-step comparison.

Run from `RECON2024\code` with `src` on `PYTHONPATH`:
`python -c "from recon2024.main import main; main()"`. Steps are switched on in each
stage's `pipeline.yaml` `steps:` list, exactly as in 2022.

Code fixes needed to run at all: Python-3.12 f-strings in `sut/steps.py` and
`bea/steps.py`; a variable clobber in `qcew/steps.py:process_codes` (the archived
source overwrote `codes` with a DataFrame); `__version__` fallback when not pip-installed.

---

## 1. What changes, stage by stage

The 2022 design was: national 2017 detail SUT scaled to the 2022 71-sector summary and
balanced; one national A matrix; regional imports and exports = national × wage share;
BEA-balanced employment; RPCs from the FAF regression. Three things are different in 2024
and they all land in the `sut` stage:

1. **The detail SUT is the 421-sector construction package**
   (`Construction_IO_2024\outputs\Use_revised_NAICS421_2024.csv`,
   `Supply_revised_NAICS421_2024.csv`, `A_total`/`A_domestic_NAICS421_2024.csv`).
   It is already at 2024 current prices and balanced to the 2024 summary, so
   `process_suts` + `create_target` (2017→2022 projection and Balance-engine fit)
   are not run. Construction rows are NAICS industries, not BEA structure types,
   so the QCEW construction allocator (`qcew/construction.xlsx`, 50 QCEW × 12 BEA)
   is replaced by a direct NAICS map.
2. **Each state gets its own international imports vector** — retained imports by BEA
   commodity from `raw\Trade\out_2024\retained_imports_state_bea_2024_SUTscaled.csv`
   (goods; Census state HS6 general imports net of re-exports, SAMNJ22 rules) — and its
   own exports vector (`netted_exports_state_bea_2024.csv`). The 2022 rule
   (state imports = national imports × state wage share) goes away for goods.
3. **State industry-by-industry A matrices.** With state-specific m_s and e_s the
   domestication is done per state:
   `d_s(c) = (q_s(c) − e_s(c)) / (q_s(c) − e_s(c) + m_s(c))`,
   `A_dom(s) = D · diag(d_s) · B` on the 421 axis, one file per state
   (`results\Amatrices\recon2024_Amatrix_SS000.csv`), counties inheriting their
   state's A. In 2022 there was one national A and the state/county difference was
   carried entirely by the RPC vectors.

Everything else is a substitution of inputs.

### 1.1 `qcew` — done except the bridge target
| 2022 step | 2024 |
|---|---|
| `process_data`, `balance` (Balance engine on the raw singlefile) | **replaced** by `import_filled`: reads `raw\QCEW\qcew_{source,tree,balanced}_2024.parquet` (v6 row-normalised exact-MAP fill for own 5, `stage_a_gov_fill` for own 1–3), already in João's process layout. `qcew_previous.csv` = national rows of the 2022 balanced panel. |
| `process_codes` | unchanged (2024 area titles include CT planning regions 09110–09190). |
| `process_qcew_bridge`, `process_naics_bridge`, `test_bridge`, `apply_bridge` | unchanged code; **the SUT target must become the 421 axis** (`Construction_IO_2024\outputs\A_naics_axis_2024.npz`) — open, see §2. |

### 1.2 `bea` — inputs exist, three steps to rewrite
| 2022 step | 2024 |
|---|---|
| `process_structure`, `process_codes`, `process_data` | same code, 2024 files (`SAGDP2/3/4/7__ALL_AREAS_1997_2024`, `CAINC5N/6N 2024`, `CAINC4 2024`). `SAEMP25N`/`CAEMP25N` are read for structure only (they end in 2022). |
| `balance_gdp` | **replaced** by import of `raw\SAGDP\filled\sagdp_comp_gos_filled_2022_2024.csv` (MAP fill, checked to the 2024 SUT within $1M). |
| `balance_other` | **replaced** by import of `raw\CAINC\filled\bea_comp_earn_county_2024_filled.parquet`. |
| `balance_employment` | **replaced** by `assemble_employment` (to write): county × sector = QCEW covered (all ownerships, filled) + uncovered W&S (`raw\Proprietors\uncovered_ws_state_sector_2022_2024.csv`, state → county by compensation) + proprietors (`FINAL_prop_jobs_county_sector_2022_2024.csv`) + rail (`rail_jobs_county_2022_2024_v5.csv`) + farm (`farm_panel\`) + military (CAEMP25N lc 2002 carried from 2022). This is the "total-jobs control" — BEA's 2024 county employment no longer exists. |
| `process_pce_income` | 413-commodity PCE split must be re-run on `SAPCE 2024` (`raw\PCE\final_work_PCE.g` is the 2022 GAUSS; port to Python). |

### 1.3 `sut` — the redesign (§1 above)
| 2022 step | 2024 |
|---|---|
| `process_suts`, `create_bridge`, `create_target` | **replaced** by `import_construction_sut` (to write): load the 421 Use/Supply, build the 421↔71 aggregation bridge from the package's own bridge. |
| `process_margins` | FAF6 (`raw\FAF\FAF6.0_access.accdb`) 2024 extract replaces `faf_selected_subset.csv` (FAF5, 2022). |
| `process_bea` | same logic (VA by detail industry from state GDP components) on the 421 axis. |
| `apply_margins`, `create_symmetric`, `process_government` | same code; usaspending pull for 2024. |
| `create_vectors` | **rewritten**: state imports/exports from `raw\Trade\out_2024\` for goods; services per §3; counties within state by wage share as before. |
| `export_A_matrices` | per-state A_dom (§1.3). |

### 1.4 `rpc` — inputs mostly missing for 2024
| step | 2024 |
|---|---|
| `process_area`, `process_rpc_regression` | unchanged (GEOINFO 2020, FAF regression). |
| `process_earnings` | CAINC5N 2024, done in config. |
| `process_total_income_tax`, `process_business_income_tax`, `process_state_local_revenue` | need the 2024 IRS SOI state shares and Census State & Local Government Finance tables (`24in01stateshares`, `24dbs01t05co`, `24slsstab1` equivalents). 2022 files are still wired. `recon2024_workspace.zip` has `irs_fy2024_state_collections.csv` and `gov_finance.py` from August that may cover part of this. |
| `calculate_rpc` | supply = output − e_s, demand = supply + m_s now use the **state-specific** trade vectors. |
| `export_results` | prefix `recon2024`; metadata year from config. |

---

## 2. Open design decisions

1. **QCEW → 421 bridge.** `process_naics_bridge` builds NAICS→SUT from the 2017
   benchmark NAICS sheet (405 industries). The 421 axis adds construction detail
   (31 NAICS 236/237/238 industries in place of the 12 BEA structure types) and keeps
   everything else. Plan: generate the bridge from the 2017 sheet as now, then replace
   the construction block with the identity map to the NAICS-31 construction
   industries. `apply_bridge.government` (own 1–3 → S00xxx/GSLGx) unchanged.
2. **PCE 413 split.** Either port `final_work_PCE.g` or re-use the 2022 commodity
   shares within each SAPCE line scaled to 2024 SAPCE totals. The second is a
   30-minute job and probably adequate; the first is a day.
3. **Government finance 2024** — whether Census has released the 2024 Annual Survey of
   State & Local Government Finances (as of 2026-09 it should exist); otherwise carry
   2023.

---

## 3. Services in the state trade vectors

Goods (BEA commodities 111CA–339) come from the Census state pipeline. Services
imports and exports have **no state-level source**: BEA's only state series is the 2022
BE-120 exploratory paper (WP2026-12; eight business-service categories, both
directions, no tables yet; travel, transport, education, financial and insurance
explicitly not covered). So:

- **Services exports e_s(c)**: state share of national exports of service commodity c =
  state share of national **output** of c (from the state SUT / bridged QCEW wages), with
  overrides where a real allocator exists — travel (NTTO international visitor spending
  by state), education (IIE Open Doors international-student spending by state).
- **Services imports m_s(c)** — *states' shares of imports of international services*
  (implemented in `apply_state_trade` as the demand proxy on the industry axis:
  share_s(i) = [Σ_j A(i,j)·x_j(s) + pce_s(i)] / national; the overrides below are still to do):
  state share of national **use** of c (intermediate use from the state A-matrix
  projection + state PCE for the household part), i.e. the demand proxy; overrides:
  passenger fares and travel by state PCE on foreign travel (SAPCE net foreign travel
  line), freight/port services by the state's goods-import totals, insurance and
  financial by state use of those commodities. When BEA publishes the BE-120 state
  tables, their 2022 shares replace the proxy for the eight covered categories.
- National control: the 2024 summary SUT imports vector (services rows), so
  Σ_s m_s(c) = M(c). Goods rows are controlled to the same SUT vector by the
  `_SUTscaled` file already.

The demand proxy needs the state use table, so services shares are computed **after**
the state SUT projection inside `create_vectors`, not before it.

---

## 4. Running it

```
cd RECON2024\code
set PYTHONPATH=%CD%\src
python -m recon2024.main                       # every stage, steps as enabled in each pipeline.yaml
python -m recon2024.main --stages bea          # one stage
python -m recon2024.main --stages bea --steps import_gdp_filled assemble_employment   # named steps
```
Dependencies beyond João's list: `pyarrow`, `xmltodict`, `beautifulsoup4`, `statsmodels`
(pandas 2.3 / numpy 2.2 tested; the archived code needs Python >= 3.10 — two 3.12-only
f-strings were rewritten).

## 5. Status log

**2026-09-20**
- Package copied, renamed `recon2024`, parameterised (`stages.yaml`: base dir per OS,
  `year`, `prev_year`; `{year}` tokens in every pipeline.yaml; `--stages/--steps` CLI).
- `qcew` stage runs end to end on the **421 axis**: `process_codes` -> `import_filled`
  (v6/stage-A panel; published cells reproduced exactly) -> `process_qcew_bridge` ->
  `process_naics_bridge` -> `retarget_sut_axis` (12 BEA structure types -> 31 NAICS,
  universe 402 -> 421, no gaps) -> `test_bridge` -> `apply_bridge` (construction sheet
  `naics31`; establishments added to the output). US 2024: 155.0M covered jobs,
  $11.71T wages, 99.94% bridged.
  *Connecticut*: `raw\QCEW\area-titles-csv.csv` lacked the nine planning regions
  (9110-9190) so they were appended (original kept as `area-titles-csv_pre2024CT.csv`);
  `bea\bea_config.xlsx` sheet `region` now lists 09110-09190 instead of the eight
  counties (original workbook kept as `bea_config_RECON2022_original.xlsx`).
- `bea` stage: `process_structure/codes/data` run on the 2024 files (SAEMP25N/CAEMP25N
  read with `value_year: 2022` for structure). New steps in place of the Balance-engine
  fits: `import_gdp_filled` (4,680/4,680 cells; published wages/GOS reproduced within
  $1.4M — the MAP fill pins published cells softly; components vs published GDP total
  within $6M), `import_other_filled` (345,203/345,203 cells, published reproduced
  exactly), `assemble_employment` (v1 rules below), `project_pce_413` (v1: the 2022
  413-commodity state PCE carried by matched SAPCE4 line growth — 58 of 77 bottom
  lines matched within 5%, the rest use state total-PCE growth; US +13.2% vs SAPCE
  total +12.5%).
- Cloud/device note: the device shell kills anything over 3 minutes, so the long steps
  were run on a cloud mirror of `raw\` + `RECON2024\` and the outputs copied back; on
  Windows the package runs directly.

### assemble_employment v1 rules (to review)
- private lines 100-1900: covered (QCEW all ownerships, bridged) + uncovered W&S
  (state file, to counties by covered share, compensation share fallback) + proprietors
  (county FINAL file); state sub-line detail gets the non-covered part in 2022
  (published - covered) proportions.
- farm: farm panel `farm_employment` (2024 carried by NASS); CT planning regions from
  the separate CT file.
- GC, GSL: covered + (2022 published - 2022 covered)+ x covered growth.
- GM: 2022 CAEMP25N military x county military compensation growth (CAINC6N line 2002).
- counties are scaled to the state line totals; national = sum of states; integer
  bottom-up correction as in João's code.
Not yet in: student workers / commission workers / religious-org adjustments beyond
what the 2022 residual carries; CT planning-region history (none exists) — GC/GSL/GM
there are the state figure split by covered jobs / compensation.

**Control (added later the same day): BEA SAINC4.** State employment did not die with
SAEMP25N — `raw\SAINC\SAINC4__ALL_AREAS_1929_2025.csv` carries total (7010), wage & salary
(7020) and proprietors (7040) employment by state through 2024 (RECON2024 §67). The
assembly is now controlled to it: per state, nonfarm uncovered W&S is scaled so
covered + uncovered = 7020, nonfarm proprietors so farm + nonfarm = 7040 (farm held to the
NASS panel). Factors: uncovered median 1.05 [0.54, 1.43], proprietors median 0.96
[0.94, 1.00]. US total = 218,181,200 = SAINC4. Before the control the free assembly was
218.84M (+0.3%). The FINAL proprietor files are broken for Connecticut in 2024 (4,742
jobs; the planning-region switch) — the 2023 CT pattern is used with the SAINC4 level.

Three things learned while getting the free assembly to +3.0% (218.8M vs 212.4M):
covered jobs must be taken **by ownership straight from the filled panel** (own 5 for the
private lines, own 1 = GC, own 2+3 = GSL; the bridged `bea_qcew` puts government-owned
water/sewer, parks etc. into private industries); on the **county-sum basis** (the uncovered
series was pinned in 2022 without the unknown-county rows, so those 5.5M jobs travel with
"uncovered"); and BEA's SAEMP code `3364-3466,3369` is a typo for `3364-3366,3369`
(taken literally it double-counts furniture and misc. manufacturing). NAICS-2022 recoding
of retail (44-45) and information (51) means their sub-lines are split by 2022 published
shares. National check still wanted: BEA NIPA 6.7D / SAEMP-equivalent for 2024 by industry.

**sut stage — runs end to end** (`extract_summary_year` → `process_suts` → `create_bridge`
→ `process_bea` → `create_target` → `process_margins` → `apply_margins` →
`create_symmetric` → `process_government` → `create_vectors` → `export_A_matrices`):
- `extract_summary_year` writes `Use_2024.xlsx`/`Supply_2024.xlsx` from the multi-year
  summary workbooks ('...' → '---').
- `process_suts` reads the 421 construction package through the new
  `operations.extract_construction` (`load_construction` config); source_agg = target_agg =
  the 2024 summary. The package is balanced on commodities (max $0.2M) but **not on
  industries** (input ≠ output by up to $45B, nonstore retail; $21B industrial building
  construction) — João's `create_target` Balance engine closes that: 86,950 variables,
  6,922 constraints, 79 s, max residual $8M, aggregates within $5M.
- `process_bea`: `sut\bea_config.xlsx` `tax_det` expanded to the 31 NAICS (2017 product
  taxes split by the package's translator G) and `tax_agg` refreshed with the 2024 summary's
  T00TOP/T00SUB (was hard-coded 2022; original workbook kept as
  `bea_config_RECON2022_original.xlsx`). PCE from the 413 file is re-axed (12 construction
  commodities, all zero, dropped).
- `process_margins`: archived code called a `read_sut()` that no longer exists — patched to
  read the source SUT container. FAF field still the FAF5 2022 subset (see the parallel
  FAF6 work below). Pharmaceutical preparations (325412) keep an $18B use/supply gap after
  the trade-margin RAS (5,000 iterations, drug wholesaling column) — to look at.
- `process_government`: usaspending API is blocked from the cloud shell; the 2022
  `government.csv` state shares were copied in (they are only used as shares).
- `create_vectors` → new `operations.apply_state_trade` (§3 below): 243 goods commodities
  from the Census state pipeline cover 86% of national imports and 60% of exports; states
  are controlled to the national SUT vectors; exports capped at 90% of a state industry's
  output (1,695 cells, mostly port pass-through; NJ autos was e > q); counties keep their
  within-state wage pattern.
- `export_A_matrices`: 52 files (`results\Amatrices\recon2024_Amatrix_SS000.csv`, 422×422
  with the households row/column) — João's own code already domesticated per region via
  `supplydemand = (q−e)/(q−e+m)`, so with state-specific m and e each state's A now differs
  (e.g. NJ autos 0.003, TX 0.50, US 0.62; NJ semiconductors 0.25 vs US 0.79).

**rpc stage — runs** with the 2022 tax/revenue inputs (flagged) and the archived
beta-regression: `results\states\recon2024_SS000.csv` (52) and
`results\SS000\recon2024_SSCCC.csv` (3,219 counties), same 35 columns as 2022.
`raw\RPC\industry_sctg_conversion_421.csv` adds the 31 NAICS rows (sctg 0).

**Collision, 2026-09-20 ~18:15 PC time.** A second session was editing the same tree on
the PC (FAF6 margins and regression data, fractional-probit RPC with `post_processing`
switches, an A/B script). My whole-tree copy overwrote its `rpc\operations.py`,
`rpc\steps.py`, `rpc\pipeline.yaml`, `sut\steps.py`, `sut\pipeline.yaml`,
`sut\margins_config.xlsx`. Its `.pyc` files were saved to `_xfer\_recovered_pyc\` and
decompile cleanly (Decompyle++) except `rpc\steps.py:export_results`; its raw inputs and
scripts are intact. Merge pending — see the chat.

**2026-09-20, later.** The parallel session re-saved its `rpc\` files (fractional-probit QMLE
on the FAF6 panel, clustered SEs; `post_processing` switches incl. `structural_zero`). Adopted
unchanged; the rpc stage was rerun on the 2024 vectors with them (R² on the rpc scale 0.464,
MAE 0.140; 785,059 county cells set to rpc = 0 for no local supply) and `results\` refreshed.
Its `sut\` changes (FAF6 margins config) were not re-saved and remain to be redone by it.

**2026-09-20, services imports by state (§3 implemented).** `operations.services_import_shares`
(called from `apply_state_trade`) allocates every import commodity without a Census state
source by use component under import-use proportionality: household part by a BEA SAPCE4
line (foreign travel 129, air fares 64, water 65, tuition 97/95, hospitals 51, insurance 109,
financial 105; else the state's PCE of the commodity), intermediate part by the **QCEW payroll
of the buying industries** (Mike's location-quotient basis, `intermediate_weight: payroll`),
port/freight services by the state's goods imports, government by the government proxy;
then multiplied by the **producing industry's payroll LQ^-0.5** (`producer_lq_alpha`; states
without the producer import more per unit of demand). 64 commodities, $472bn. Noncomparable
imports (S00300, $173bn after create_symmetric) stay as João's primary-input row split by the
using industries' wages — the same payroll basis. Outputs in `process\SUT\`:
`services_imports_method_2024.csv` (allocator per commodity), `services_imports_state_commodity_2024.csv`,
and the full `imports_state_commodity_2024.csv` / `exports_state_commodity_2024.csv`
(state x 421 commodity, $M) — the per-state import vectors. Results and A matrices rerun.

**2026-09-20, Use-side domestication (`export_state_use_A`).** Per state, on the commodity x
industry axis of the balanced margin-adjusted SUT: B (national total-use coefficients) ->
columns scaled by the state/US intermediate-use fraction ratio (João's productivity
adjustment, bounds 0.25-4) -> B_dom = (I - P_s) B_s with P_s = diag(m_s(c)/U_s(c)), the
international import share of the state's total use of c (m_s from
`imports_state_commodity`, U_s = B_s x_s + state PCE + government by income + other FD by
producer share; cap 0.95) -> A_s = D' B_dom + households row/column. Outputs
`results\Bmatrices\recon2024_Bdom_SS000.csv.gz` (+ `Pimport_SS000.csv`, the diagonal) and
`results\Amatrices_use\recon2024_Amatrix_SS000.csv`; `Amatrices\` (João's route, supply/
demand ratio on A rows) is kept for comparison. Use-weighted international import share of
state use: mean 4.2% (MI 10%, NJ 8.7%); NJ A column sums 0.440 (use-side) vs 0.419 (João).
RPC (interregional) domestication still follows in the rpc stage and is not in these A's.

**2026-09-21, basic-price import shares.** Use is at purchasers' prices (João endogenises
margins on the supply side; D carries wholesale/retail/transport as partial producers of
goods) while imports are at customs/CIF value, so `export_state_use_A` now measures the
import share against use net of trade and transport margins (per-commodity margin totals
from the `margin_trade`/`margin_transport` sheets of `sut_det_2024_margins.xlsx`):
p_basic = m_s / (U_s x basic_share), capped 0.95; the share removed from the purchasers'-price
coefficient is p_basic x basic_share, the margin part being domestic service. 258 margined
commodities, basic share median 0.73 (apparel 0.39, furniture 0.46). `Pimport_SS000.csv` now
lists p_import_basic, basic_share_of_use and p_applied. A and B_dom files regenerated.

**2026-09-21, state output and value added (Mike: plan 2, plan 1 as back-up).** In
`create_vectors`, state (and county) output is the national output/compensation ratio times
the region's compensation (João), except in industries whose compensation is < 15% of value
added nationally (owner-occupied housing, lessors, petroleum refining, oil & gas, some farm
lines), where output is split by GDP share. New `operations.reconcile_va_output` then handles
regions whose VA exceeds 98% of output (`va_output.f_min: 0.02`, or the parent's own VA/output
where higher, e.g. private households = 1): **plan 2** keeps the region's GDP and raises its
output to VA / (parent's VA/output ratio), taking the extra from the same industry in the
other regions of the parent pro rata to their headroom; **plan 1** (back-up, when the others
lack headroom) cuts the region's GOS to fit and gives it to the others. States: 740 cells,
671 by plan 2, $672bn of output moved (mostly banking 52A000/522A00, federal defence, insurance
agencies, child day care), no GOS moved; states sum to US output exactly in every industry.
Counties (parent = state): 16,229 cells, $9.4bn moved; $3.9bn of county GOS in other-crop
farming (111900) could not be placed and is dropped (county farm GOS exceeds the state's
farm output there). Replaces João's GDP-proxy and zero-intermediate fallbacks, which had
left states at 101.2% of US output. Remaining zero-intermediate state cells are private
households and state/local transit (zero intermediate nationally in the SUT) and five $1k
Wyoming chemical cells. Diagnostics: `process\SUT\va_output_diag_{state,county}.csv`.
Everything downstream (state Use/A, João A, RPC results) regenerated.
State Use matrices in levels ($ thousand, commodity x industry): results\Usematrices\recon2024_Use_total_SS000.csv.gz (productivity-adjusted, total) and recon2024_Use_domestic_SS000.csv.gz (net of international imports).

**2026-09-21, 2024 government and tax inputs.**
- State & local columns of `government.csv`: Census Annual Survey of State & Local Government
  Finances time series 2017-2024 (`raw\Government\GOVSLOCALFINTIMESERIES.GS00LOCALFIN_...zip`),
  2024, via `sut` step `process_government_census` (items LF0094 current operations, LF0107
  education, LF0129 hospitals, LF0132 health; 2022 values reproduce João's table within 0.5%).
- Federal columns: FY2024 federal **contracts** (award types A-D) by **place of performance**,
  DOD = defense, other agencies = nondefense, pulled from the USAspending API through the
  browser (`raw\Government\usaspending_contracts_pop_state_FY2024.csv`, step
  `process_government_usaspending`): $415bn / $281bn. João's 2022 columns were all award types
  by recipient location ($4.0T nondefense, incl. Medicare Advantage, loans and insurance booked
  to insurer HQ states - MN, KY, IN, ND).
- rpc tax inputs (step `process_rpc_tax_inputs`, writes into `process\RPC\`):
  `state_local_revenue.csv` from the same Census file, 2024, the fifteen 22slsstab1 lines mapped
  to item codes (2022 within 1.5%); `business_income_tax.csv` from the IRS Data Book FY2024
  state collections (`raw\Government\irs_state_collections_FY2024.csv`);
  `total_income_tax.csv` stays at IRS SOI tax year 2022 (latest on disk; used as shares).
  The rpc steps process_state_local_revenue / process_business_income_tax /
  process_total_income_tax must therefore not be run (they would restore 2022) - run the rpc
  stage with `--steps process_area process_vectors process_rpc_regression process_earnings export_results`.
- Individual income tax (households' federal tax rate in the rpc stage): IRS SOI Historic
  Table 2, tax year 2023 (latest published; pulled from irs.gov through the browser,
  `raw\Government\irs_soi_historic_table2_TY2022_TY2023_state_totals.csv`), A06500 income tax
  after credits (reproduces 22in01stateshares col W within 1.2% for 2022), carried to 2024 by
  each state's personal income growth 2023-24 (SAINC4 line 10) - i.e. the TY2023 state
  tax/income rate on 2024 income. US $2,234bn. Written by `process_rpc_tax_inputs`.
