# FAF6-based 2024 RPC anchors (built 2026-09-21)

Files
- rpc_anchor_state_sctg_2024_FAF6.csv: 51 states x SCTG 1-41 (2,091 cells)
- rpc_anchor_zone_sctg_2024_FAF6.csv: 134 FAF6 zones x SCTG 1-41 (5,494 cells)
- build_anchors_2024.py: the script that builds both files. Inputs are FAF6.0.csv, faf5_od_2022_2024.csv.gz and cfs2022_cull_state_sctg_mode.csv.

Method
1. Start from FAF6.0 2022 flows, value in 2022$ plus tons, split by component (intra, out_dom, in_dom, exp_for, imp_for) and domestic mode.
2. Carry each cell to 2024 with the FAF5.7.1 current-dollar growth ratio for that cell and component.
   - FAF5 zone 190 supplies growth to FAF6 zones 191 and 199.
   - FAF5 zone 399 supplies growth to FAF6 zones 395 and 399.
   - Where a cell has no FAF5 flow, use national SCTG x component growth.
3. Cull nonproducer short-haul intra shipments.
   - For each origin state x SCTG x mode, take the share of CFS 2022 intrastate value that the Paper-1 distance bands remove.
   - Where CFS value in the cell is under $5M, use the national SCTG x mode share instead.
   - Zones use the share for their state.
   - SCTG 1, 2, 3 and 41 are exempt (first handlers).
   - The 2022 shares are held fixed for 2024.
4. rpc = intra_c / (intra_c + in_dom + imp_for), with supply = intra_c + out_dom + exp_for.
5. Assign a status to each cell:
   - structural_zero: supply = 0, so RPC is set to 0.
   - no_demand: demand = 0.
   - thin: demand under $10M.
   - observed: everything else.
6. cred_weight = demand / (demand + $50M) is for shrinking toward the model later: rpc* = w*obs + (1-w)*model. When rpc_2024 is used as a raking target, apply the shrinkage to thin cells.
7. Tons are 2022 only, because the FAF5.7.1 OD extract has no tons. They are included for W/V checks: rpc_tons_2022, wv_intra_raw and wv_intra_c (tons per $M).
