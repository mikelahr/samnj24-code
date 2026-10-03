# Upstream run order: how New Jersey's state IO inputs were produced

`run_samnj24.py` starts from New Jersey's industry vectors. Those vectors came out
of a national RECON2024 run, because every state's figures are held to national
totals and the RPC model is estimated on all states. This file records that run,
in order, as it was executed for the version of the accounts in the paper
(last full rerun 2026-09-27/28). The paths are relative to the `RPC_explorations`
tree, and the code is in `code/recon2024_package` and `code/scripts`.

Running this part needs the full national inputs (about 18 GB under `raw/`). The
package carries only New Jersey's rows of them (`data/nj`), plus the national SUTs
and every bridge (`data/national_sut_421`, `data/bridges`). The step-by-step history,
including every design decision, is in `code/docs/README_RECON2024_port_v2.md`.

The device that ran this kills processes after 3 minutes, so every step was run on
its own:

```
cd RECON2024/code
export PYTHONPATH=$PWD/src          # Windows: set PYTHONPATH=%CD%\src
python -m recon2024.main --stages <stage> --steps <step> [<step> ...]
```

## 0. Inputs built outside RECON2024 (each has its own package or notes)

| input | built by | NJ extract in this package |
|---|---|---|
| 421-sector national SUT, 2024 (construction-revised) | `Construction_IO_2024` project | `data/national_sut_421/*_revised_NAICS421_2024.csv` |
| QCEW 2024 filled panel (suppressed cells filled, v6 / stage A) | QCEW paper package (`RECON_replication_package_2026-08-13_2002Z.zip`) | `data/nj/qcew/qcew_final_2024_NJ.csv.gz` (after bridging) |
| SAGDP compensation/GOS with suppressed cells filled | `raw/SAGDP/filled` (`code/docs/README_SAGDP_filled.md`) | `data/nj/bea/sagdp_comp_gos_filled_2022_2024_NJ.csv` |
| BEA jobs continuation, release 2026-09-27 | `BEA_EMP/release_20260927` (its README) | `data/nj/jobs/` |
| State international trade, ESR common-rate series | "What passes through" package, `Statewise International Trade/ESR_package_20260927_1626Z` | `data/nj/trade/account_NJ_hs6_2024.csv.gz` |
| FAF6 2024 RPC anchors | `anchors_2024_FAF6/build_anchors_2024.py` (copied to `code/scripts`) | `data/nj/rpc/rpc_anchor_state_sctg_2024_FAF6_NJ.csv` |

## 1. Pre-builds (scripts in `code/scripts`)

| # | command | writes |
|---|---|---|
| 1.1 | `python raw/Trade/code/build_state_trade_2024_ESR.py` (`build_state_trade_2024_ESR.py`, `bridge.py`) | `raw/Trade/out_2024_ESR/`: retained imports and netted exports by state x BEA commodity, the gateway margin by state, and exports with gateway (scenario `cr`, mu = 0.2529) |
| 1.2 | `python raw/Proprietors/release_20260927/build_assemble_inputs.py` (`build_jobs_assembly_inputs_release20260927.py`) | proprietors by county x sector, and uncovered wage-and-salary jobs by state x sector, cut to the release |
| 1.3 | `python RECON2024/code/scripts/build_travel_exports_state.py` | `process/SUT/travel_exports_state_commodity_2024.csv` (NTTO visitor and IIE student spending) |
| 1.4 | `python RECON2024/process/RPC/predict_2024/build_county_faf6_zone.py` | `county_to_faf6_zone.csv` |

## 2. Stage `qcew`: jobs, wages and establishments on the 421 SUT axis

`process_codes` → `import_filled` → `process_qcew_bridge` → `process_naics_bridge` →
`retarget_sut_axis` → `test_bridge` → `apply_bridge`

The bridges are `data/bridges/qcew_bridge_NAICS_to_SUT421.xlsx` and
`construction_BEA12_to_NAICS31_2024.csv`.

## 3. Stage `bea`: value added, earnings, jobs and consumption by state and SUT industry

`process_structure` → `process_codes` → `process_data` → `process_qcew` →
`process_proxies` → `import_gdp_filled` → `import_other_filled` →
`assemble_employment` → `rebuild_pce_413` → `process_pce_income` → `apply_bridge`

- `assemble_employment` holds each state's private nonfarm jobs by sector to the
  release (`state_sector_release` in `bea/pipeline.yaml`). The SAINC4 rescaling is
  skipped when the release is present.
- The bridges are `data/bridges/bea_bridge_lines_to_SUT421.xlsx`, and the PCE bridge
  files `pce_*.txt`.

## 4. Stage `sut`: national SUT, margins, state vectors and state A matrices

`extract_summary_year` → `process_suts` → `create_bridge` → `process_bea` →
`create_target` → `process_margins` → `apply_margins` → `create_symmetric` →
`process_government_census` → `process_government_usaspending` →
`process_rpc_tax_inputs` → `create_vectors` → `export_A_matrices` →
`export_state_use_A` → `export_domestic_demand`

- **`create_vectors`**
  - Calls `operations.apply_state_trade`, which reads goods trade from
    `raw/Trade/out_2024_ESR`. It places the transport part of the gateway margin
    (21.4%) in commodities 481000–486000. The wholesale part (78.6%) is already in
    the exports-with-gateway file.
  - Allocates services imports with `operations.services_import_shares`.
  - Calls `operations.reconcile_va_output`, which makes states sum to the US in
    every industry.
- **Government proxy.** It is `process/SUT/government_2024.csv`: Census 2024 for
  state and local; USAspending FY2024 contracts by place of performance for federal.
  The same file was also written to `government.csv` on 2026-09-28.
- **`export_state_use_A`.** This produces `results/Amatrices_use/recon2024_Amatrix_SS000.csv`
  and the state Use matrices. For New Jersey it is re-implemented, and verified
  cell for cell, in Stage A of `run_samnj24.py`.

## 5. RPCs

```
python RECON2024/code/scripts/export_supply_demand.py <RPC_explorations> 2024   # supply, domestic demand, regressors
cd RECON2024/process/RPC && python predict_2024/predict_rpc_2024.py            # fractional-probit predictions (rpc_model.pickle)
cd RECON2024/process/RPC/predict_2024 && python rake_rpc_2024.py              # rake to FAF6 2024 state x SCTG and zone x SCTG
python -m recon2024.main --stages rpc --steps process_area process_vectors process_rpc_regression process_earnings export_results
```

`export_results` writes `results/states/recon2024_SS000.csv`, which carries the raked
RPCs. New Jersey's file is `data/nj/state_io/recon2024_34000.csv`, the input to
Stages B–D. Do not run the rpc steps `process_state_local_revenue`,
`process_business_income_tax` or `process_total_income_tax`: they would restore the
2022 tax inputs. Their 2024 replacements come from the sut step
`process_rpc_tax_inputs`.

## 6. SAMNJ24

`python run_samnj24.py --check` (package root). This replaces
`code/legacy_samnj24_builds/build_samnj24.py` and `build_table6.py`, which it
reproduces exactly.
