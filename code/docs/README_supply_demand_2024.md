# 2024 supply and domestic demand for re-estimating the RPC equation

Written 2026-09-21 by the RECON2024 port session. **Version 3 (2026-09-21 18:58 UTC): supersedes versions 1 and 2.** Version 3 changes from version 2:
- State taxes on products now follow each state's own SAGDP taxes-less-subsidies (states without a sales tax get little); this moves some output through the value-added/output reconciliation.
- County `imports` are split by use share.
- Effect on supply/demand vs version 2: demand-weighted 0.5%; 228 of 21,408 state cells move >5%; the state × SCTG panel median is 0.19%, with 196 cells >1%.

The version 2 changes (still in effect): Version 2 adds items 6 and 7:
- State PCE by commodity is rebuilt from 2024 SAPCE4 and the national 2024 SUT PCE (`rebuild_pce_413`).
- Nonresident travel and education spending ($223.6bn) moves from PCE to exports, by state where spent: NTTO overseas visitors, NAFSA students, and a Canada/Mexico residual placed by hotel output. Goods bought by visitors are exports of the producing states.
- `raw/RPC/regression_data_FAF6_domdem.xlsx` has been rebuilt from these files.

**Files** (all in RECON2024/process/SUT, $ thousand, 2024)
- `supply_demand_2024.csv`: one row per region × industry (421-sector axis; US `00000`, states `SS000`, counties). Columns:
  - `level`: 0 US, 1 state, 2 county.
  - `sctg`: from raw/RPC/industry_sctg_conversion_421.csv (0 = services/non-freight).
  - `output`, `exports`, `imports`, `pce`: `exports` and `imports` are international only; `exports` includes nonresident travel and education spending.
  - `supply` = output − exports (international).
  - `demand_domestic` = use net of international imports.
  - `supplydemand` = supply / demand_domestic.
  - QCEW employment, establishments and wages.
- `supply_demand_state_sctg_2024.csv`: the same summed to state × SCTG, the unit of the FAF regression panel.
- `domestic_demand_2024.csv` and `domestic_demand_phi_2024.csv`: the raw step output and its calibration factors.

**Terms.** Imports and exports are between nations; inflows and outflows are between states or counties. The FAF6 dependent variable is rpc = m / (m + inflows). Domestic demand is the matching denominator concept: local supply to local users plus inflows.

**How domestic demand is built** (sut step `export_domestic_demand`):
1. Use at purchasers' prices is V_r = B · diag(r_s) · x_r + PCE + investment + inventories + government + trade/transport margin use.
   - B: national Use coefficients.
   - r_s: the state's productivity ratio, bounded to [0.25, 4]. Counties use their state's input recipes.
2. The use is moved to the industry basis with the market shares D′.
3. It is calibrated per industry by φ so that, for the US, domestic demand equals output − exports exactly.
4. International imports are subtracted. States use their own imports; counties get their state's imports in proportion to their share of the state's use. The imports taken out are capped at 95% of use.

**Caveats**
- 341 cells with negative use are set to 0.
- Size regressors in the current panel (raw/RPC/regression_data_FAF6_domdem.xlsx) are 2024. The dependent variable and the other regressors are 2022.

**RPC prediction regressors:** all 2024, except us_s_wv_ratio (FAF6 2022, kept by decision) and land share (land area does not change).
