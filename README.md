# Federated Intrusion Detection under Delayed Labels: A Time-Ordered Study on ToN_IoT

Code, configuration files and result logs accompanying the paper of the same
title, accepted at ICSIC 2026.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
<!-- Replace XXXXXXX with the Zenodo version DOI after the first release. -->

## Overview

The study evaluates federated intrusion detection on the time-ordered ToN_IoT
stream under explicit label-release delays. Models are evaluated with a
test-then-train protocol across simulated gateways, over ten random seeds, with
paired statistical tests, confidence intervals, ablations and controlled drift
experiments. See the paper for the full method and results.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Simulation and federated-training code |
| `scripts/` | Entry points that run the experiments and regenerate tables and figures |
| `results/` | Per-seed result logs and summary tables reported in the paper |
| `figures/` | Figures reported in the paper |
| `data/` | Instructions for obtaining ToN_IoT (the dataset is not redistributed) |
| `docs/` | Release (DOI) instructions and the code-availability text for the paper |

## Requirements

TODO: Python version and package versions (`requirements.txt`).

## Data

Download ToN_IoT from its official source and follow `data/README.md`.

## Reproducing the results

TODO: exact commands, for example:

```bash
pip install -r requirements.txt
python scripts/<run_experiments>.py --data data/<file>.csv --seeds 0-9
python scripts/<make_tables_and_figures>.py --results results/
```

Mapping of paper items to files:

| Paper item | Command / file |
|---|---|
| Main results table | TODO |
| Ablations | TODO |
| Controlled drift experiments | TODO |
| Communication cost | TODO |
| Statistical tests | TODO |

## How to cite

Please cite the paper and this archive.

```bibtex
@misc{fedids_delayed_labels_code,
  author       = {Abu Munshar, Hussam Hussein},
  title        = {Federated Intrusion Detection under Delayed Labels: A Time-Ordered Study on ToN_IoT (code and results)},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  howpublished = {\url{https://doi.org/10.5281/zenodo.XXXXXXX}}
}
```

The paper's bibliographic entry will be added after publication.

## License

- Code: MIT (`LICENSE`).
- Results and figures: CC BY 4.0 (`LICENSE-RESULTS.md`).
- ToN_IoT: not included; subject to its owners' terms (`data/README.md`).
