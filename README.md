# Federated Intrusion Detection under Delayed Labels: A Time-Ordered Study on ToN_IoT

Code, configuration and result records for the paper of the same title,
accepted at ICSIC 2026. All numbers and figures in the paper come from the
scripts and records in this repository.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23774652.svg)](https://doi.org/10.5281/zenodo.23774652)

Archived on Zenodo. Version 1.0.0: [10.5281/zenodo.23774652](https://doi.org/10.5281/zenodo.23774652).
All versions: [10.5281/zenodo.23774651](https://doi.org/10.5281/zenodo.23774651).

## Overview

The study evaluates federated intrusion detection on the time-ordered ToN_IoT
network stream under explicit label-release delays. The stream is split into
rounds of 8,000 records and assigned to eight simulated gateways. Labels of
round *r* are released at the end of round *r* + *d*, with *d* ∈ {0, 1, 5}.
Every configuration is evaluated test-then-train over ten seeds, with
seed-paired Wilcoxon tests and bootstrap confidence intervals, ablations,
communication-cost accounting and controlled drift checks.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Preprocessing, simulation, experiment runner, analysis and figure scripts |
| `results/` | Run records (`*.jsonl`), tables, check results and per-round logs |
| `figs/` | Figures as produced by `scripts/figs.py` |
| `data/` | How to obtain ToN_IoT, file checksums and stream manifests (no data files) |
| `docs/` | Release (DOI) steps and the code-availability text for the paper |

Main scripts:

- `scripts/sim.py`: stream loading, model, drift monitors, simulation, metrics and the byte model.
- `scripts/sim_v1_prespecified.py`: the simulator before the personalised variant was added; it behaves the same for all earlier configurations. It is kept so the pre-specified version can be inspected (see the paper's limitations section).
- `scripts/run_all.py`: runs every configuration × label delay × seed and appends one JSON line per run. It skips finished runs, so it can resume.
- `scripts/analyze.py`, `scripts/figs.py`: build the tables and figures from the run records.
- `scripts/control.py`, `scripts/join.py`, `scripts/make_staggered.py`: controlled monitor check, cold-start (client join) check and the staggered-drift stream.
- `scripts/quick.py`: a short smoke test of the simulator.

## Requirements

Python 3.13, numpy, pandas, scikit-learn, scipy, matplotlib and river 0.26.1:

```bash
pip install -r requirements.txt
```

CPU only. One run takes about 10 to 30 s on a 2-core machine.

## Data

ToN_IoT is not redistributed here; see `data/README.md`. The input is the 23
processed network files `Network_dataset_1.csv` … `Network_dataset_23.csv`.
`data/ton_iot_network_file_profile.csv` lists the SHA-256 of each file so you
can confirm you have the same files. The processed stream (1,276,810 records)
is not included; step 1 below rebuilds it, and
`data/stream_full.pkl.manifest.json` records what that step produced.

## Reproducing the results

Full pipeline (`RAW_DIR` is the folder that contains the 23 CSV files):

```bash
python scripts/preprocess.py RAW_DIR data/stream_full.pkl --cap 60000                    # 1. stream
python scripts/run_all.py data/stream_full.pkl results/tune.jsonl    --stage tune          # 2. step size, ADWIN delta
python scripts/run_all.py data/stream_full.pkl results/main.jsonl    --stage main    --best results/best.json --pred-dir results/pred
python scripts/run_all.py data/stream_full.pkl results/tune2.jsonl   --stage tune2   --best results/best.json   # lambda, W
python scripts/run_all.py data/stream_full.pkl results/variant.jsonl --stage variant --best results/best.json --pred-dir results/pred
python scripts/make_staggered.py data/stream_full.pkl data/stream_stagger.pkl
python scripts/run_all.py data/stream_stagger.pkl results/stagger.jsonl --stage stagger --best results/best.json
python scripts/control.py data/stream_full.pkl results/best.json results/control.json
python scripts/join.py    data/stream_full.pkl results/best.json results/join.json
python scripts/analyze.py pfl && python scripts/figs.py
```

`results/best.json` holds the values chosen on the tuning segment (learning
rate, ADWIN delta, matched period, lambda, W). The main stage reads
`matched_period` only after the first pass, so run it once more to add the
matched baselines, as was done for the paper.

Tables and figures only, from the included run records (no data needed):

```bash
python scripts/analyze.py pfl
python scripts/analyze.py proposed
python scripts/analyze.py local_adwin_replay
python scripts/figs.py
```

These commands reproduce the included tables byte for byte and the included
figures pixel for pixel.

## Configuration names

| Name in code | Name in the paper |
|---|---|
| `frozen` | Frozen |
| `fedavg_p1`, `fedavg_p10` | FedAvg, every round / every 10 rounds |
| `periodic_matched_replay`, `periodic_matched_noreplay` | Periodic, matched (with / without replay) |
| `proposed` | Shared model, triggered |
| `proposed_sched` | Shared model, triggered, W = 5 |
| `local_adwin_replay` | Local only |
| `pfl` | Personalised (ours) |
| `pfl_no_replay`, `pfl_pooled_adwin`, `abl_*` | Ablations |

`_d0`, `_d1`, `_d5` in file names give the label delay in rounds; `_s0` is the seed.

## Where each result comes from

| Paper content | File(s) |
|---|---|
| Main comparison (macro-F1, attack recall, FPR, PR-AUC, worst-client F1, bytes, updates, aggregations) | `results/summary_mean_std.csv`, `results/paper_numbers.txt`; per run: `results/all_runs_eval.csv` |
| Per-attack-type recall | `results/per_type_recall.csv` |
| Seed-paired tests and bootstrap CIs | `results/paired_tests_vs_pfl.csv`, `results/paired_tests_vs_proposed.csv`, `results/paired_tests_vs_local_adwin_replay.csv` |
| Ablations | `abl_*`, `pfl_no_replay` and `pfl_pooled_adwin` rows of the tables above |
| Tuning on the tuning segment | `results/tune.jsonl`, `results/tune2.jsonl`, `results/best.json` |
| Personalised variant | `results/variant.jsonl` |
| Staggered drift | `results/stagger.jsonl`, `data/stream_stagger.pkl.info.json` |
| Controlled monitor check | `results/control.json` |
| Cold-start (client join) check | `results/join.json` |
| Per-round timeline figure | `figs/timeline.pdf`, from `results/pred/*_d1_s0_rounds.csv` |
| Accuracy–communication trade-off (Fig. 3) | `figs/tradeoff.pdf` |

`results/pred/` holds per-round metrics and alarm logs for seed 0. Per-record
scores are not included; `--pred-dir` regenerates them.

In the camera-ready Fig. 3, the "Periodic, matched" label was moved upward by
hand to avoid overlapping another label. `scripts/figs.py` still places it at
the original position; the data are the same.

## How to cite

Please cite the paper and this archive.

```bibtex
@misc{fedids_delayed_labels_code,
  author       = {Abu Munshar, Hussam Hussein},
  title        = {Federated Intrusion Detection under Delayed Labels: A Time-Ordered Study on ToN_IoT (code and results)},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23774652},
  howpublished = {\url{https://doi.org/10.5281/zenodo.23774652}}
}
```

The paper's bibliographic entry will be added after publication.

## License

- Code: MIT (`LICENSE`).
- Results and figures: CC BY 4.0 (`LICENSE-RESULTS.md`).
- ToN_IoT: not included; subject to its owners' terms (`data/README.md`).
