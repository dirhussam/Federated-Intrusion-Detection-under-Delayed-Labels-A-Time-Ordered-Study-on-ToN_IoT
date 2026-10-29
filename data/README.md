# Data: ToN_IoT

The ToN_IoT dataset is **not redistributed** in this repository. Obtain it from
the official page maintained by UNSW Canberra:

https://research.unsw.edu.au/projects/toniot-datasets

Data files placed in this folder are excluded from version control by
`.gitignore`.

| Item | Value |
|---|---|
| Files used | All 23 files `Network_dataset_1.csv` … `Network_dataset_23.csv` in `TON_IoT datasets/Processed_datasets/Processed_Network_dataset` (22,339,021 records) |
| Ordering | Stable sort by `ts` (Unix seconds, UTC); ties keep file and row order |
| Subsampling | Constant type-wise rate, cap 60,000 records per attack type, all normal records kept, seed 0 (`scripts/preprocess.py`); 1,276,810 records out |
| SHA-256 of each file | `ton_iot_network_file_profile.csv` and `ton_iot_dataset_manifest.md` |

Files in this folder:

- `ton_iot_dataset_manifest.md`, `ton_iot_network_file_profile.csv`: rows, time span, traffic types and SHA-256 of each raw file.
- `stream_full.pkl.manifest.json`: what `preprocess.py` kept from each file, the rates and the output size.
- `stream_stagger.pkl.info.json`: the client orders used by `make_staggered.py` for the staggered-drift stream.

## Terms of use (as stated by the dataset owners)

According to the official page, free use for academic research is granted in
perpetuity, commercial use requires permission from the dataset author
(Dr Nour Moustafa), and users of the dataset are asked to cite the eight papers
listed below.

## Papers the dataset owners ask users to cite

Listed as they appear on the official page (checked 2026-10-29).

1. N. Moustafa, "A new distributed architecture for evaluating AI-based security
   systems at the edge: Network TON_IoT datasets," *Sustainable Cities and
   Society*, 2021, Art. no. 102994.
2. T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa, and F. T. H. den Hartog,
   "ToN_IoT: The role of heterogeneity and the need for standardization of
   features and attack types in IoT network intrusion datasets," *IEEE Internet
   of Things Journal*, 2021.
3. A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar, "TON_IoT telemetry
   dataset: A new generation dataset of IoT and IIoT for data-driven intrusion
   detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020,
   doi: 10.1109/ACCESS.2020.3022862.
4. N. Moustafa, M. Keshk, E. Debie, and H. Janicke, "Federated TON_IoT Windows
   datasets for evaluating AI-based security applications," in *Proc. IEEE 19th
   Int. Conf. Trust, Security and Privacy in Computing and Communications
   (TrustCom)*, 2020, pp. 848–855, doi: 10.1109/TrustCom50675.2020.00114.
5. N. Moustafa, M. Ahmed, and S. Ahmed, "Data analytics-enabled intrusion
   detection: Evaluations of ToN_IoT Linux datasets," in *Proc. IEEE 19th Int.
   Conf. Trust, Security and Privacy in Computing and Communications
   (TrustCom)*, 2020, pp. 727–735, doi: 10.1109/TrustCom50675.2020.00100.
6. N. Moustafa, "New generations of Internet of Things datasets for
   cybersecurity applications based machine learning: TON_IoT datasets," in
   *Proc. eResearch Australasia Conf.*, Brisbane, Australia, 2019.
7. N. Moustafa, "A systemic IoT-Fog-Cloud architecture for big-data analytics
   and cyber security systems: A review of fog computing," arXiv:1906.01055,
   2019.
8. J. Ashraf, M. Keshk, N. Moustafa, M. Abdel-Basset, H. Khurshid,
   A. D. Bakhshi, and R. R. Mostafa, "IoTBoT-IDS: A novel statistical
   learning-enabled botnet detection framework for protecting networks of smart
   cities," *Sustainable Cities and Society*, 2021, Art. no. 103041.
