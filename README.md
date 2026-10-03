# samnj24-code

Build code for **SAMNJ24**, a social accounting matrix for New Jersey, 2024.

This repository holds the code only. The data it reads and the matrices it
writes are deposited separately (see [Data](#data)). Together they reproduce
the accounts reported in:

> Álvarez-Martínez, M. T., Lahr, M. L., & Rodrigues, J. F. D. Social accounting
> matrix for New Jersey 2024 with methods and sources for other U.S. states.
> *Scientific Data* (submitted).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23126216.svg)](https://doi.org/10.5281/zenodo.23126216)

## What it builds

`run_samnj24.py` is the pipeline. It runs five stages, each documented where it
is defined in the file:

| Stage | Output |
|---|---|
| A | New Jersey's industry-by-industry direct-requirements matrix `A_NJ` and the state Use tables, total and net of international imports |
| B | The 65-account core SAM: 57 productive sectors, labour, capital, households, corporations, government, capital account, rest of the United States, rest of the world ($ millions, purchasers' prices) |
| C | The 19-account institutional SAM: government split into federal, state and local, with tax and transfer accounts (Table 7 of the paper) |
| D | Control totals and the accounting checks (Table 8) |
| E | The 75-account core SAM: Stage B's sector, factor and external blocks, government split by level, taxes routed through five tax accounts, and subsidies paid through a subsidy account so that no cell is negative |

The chain is deterministic and needs no manual adjustment. Every account
balances to within $0.1 million against $5.3 trillion in total flows.

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

`run_samnj24.py` itself needs only numpy, pandas and openpyxl. The remaining
requirements are for the upstream RECON2024 code in `code/`.

## Run it

Download the data deposit (below), unpack it so that `data/` and `product/` sit
beside `run_samnj24.py`, then:

```bash
python run_samnj24.py            # writes output/ and output/run_log.txt
python run_samnj24.py --check    # also compares every product against product/
```

`--check` is the reproduction test: it rebuilds all three matrices and reports
the maximum absolute difference against the published files, which should be
$0.000000 million.

## Data

The inputs and the published matrices are archived at Zenodo:

> https://doi.org/10.5281/zenodo.XXXXXXX

That deposit contains `data/national_sut_421/` (the 2024 national supply-use
tables), `data/bridges/` (every concordance used), `data/nj/` (New Jersey's rows
of every regional input, including RECON2024's New Jersey state IO accounts),
`data/DATA_DICTIONARY.md`, and `product/` (the 65-, 19- and 75-account matrices,
the account codes, and the control totals).

New Jersey's industry vectors — output, value added by component, final demand,
international trade and regional purchase coefficients — are **not** recomputed
here. They are products of the national RECON2024 run, because each state's
figures are constrained to national totals and the RPC model is estimated across
all states. They enter as `data/nj/state_io/recon2024_34000.csv` and
`vectors_34000.csv`.

## Repository layout

```
run_samnj24.py                  the pipeline (Stages A-E)
code/
  RUN_ORDER_upstream.md         the order in which the upstream code was run
  extract_nj_inputs.py          rebuilds data/ from the full working tree
  recon2024_package/            the RECON2024 package as run (qcew, bea, sut, rpc stages)
  scripts/                      stand-alone builders: state trade, jobs assembly,
                                FAF6 anchors, travel exports, RPC prediction and raking
  docs/                         RECON2024 porting log and methods note, the SAGDP
                                fill README, the supply/demand handoff README
  legacy_samnj24_builds/        the two scripts that first built the SAM;
                                run_samnj24.py supersedes them
```

## Upstream code

`code/recon2024_package` is RECON2024, which produces the state input-output
accounts for all states; SAMNJ24 uses its New Jersey rows. It runs as staged
pipelines:

```bash
python -m recon2024.main --stages sut --steps export_domestic_demand
```

`code/RUN_ORDER_upstream.md` records the order used for the 2024 vintage, and
`code/docs/README_RECON2024_port_v2.md` documents the port from the 2022 vintage.

## Citing

Cite the paper for the accounts and this repository for the code. `CITATION.cff`
carries machine-readable metadata; GitHub's "Cite this repository" button and
Zenodo both read it.

## License

MIT — see [LICENSE](LICENSE).
