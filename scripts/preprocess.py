#!/usr/bin/env python3
"""Build a time-ordered, class-wise subsampled stream from ToN_IoT processed network files.

Usage: python preprocess.py RAW_DIR OUT_PICKLE [--cap 60000] [--normal-rate 1.0]

Pass 1 counts records per traffic type over all files. Pass 2 keeps each record independently
with a constant type-wise rate min(1, cap / count_type) (normal traffic: --normal-rate), seed 0.
Rates do not change over time, so the temporal order and the timing of each attack type are
kept, while very large attack types (scanning, DDoS) no longer swamp rare ones (MITM). Identifier and free-text
columns are kept only for client assignment and auditing; they are never used as features.
"""
import argparse, glob, os, re, zlib, json, hashlib
import numpy as np, pandas as pd

KEEP = ['ts', 'src_ip', 'dst_ip', 'proto', 'service', 'duration', 'src_bytes', 'dst_bytes',
        'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
        'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected',
        'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established', 'http_trans_depth',
        'http_method', 'http_version', 'http_request_body_len', 'http_response_body_len',
        'http_status_code', 'http_orig_mime_types', 'http_resp_mime_types', 'weird_name',
        'weird_notice', 'label', 'type']


def file_index(p):
    return int(re.search(r'_(\d+)\.csv$', p).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir'); ap.add_argument('out')
    ap.add_argument('--cap', type=int, default=60000)
    ap.add_argument('--normal-rate', type=float, default=1.0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.raw_dir, '**', 'Network_dataset_*.csv'), recursive=True),
                   key=file_index)
    counts = {}
    for p in files:
        vc = pd.read_csv(p, usecols=['type'], dtype=str)['type'].value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    rates = {k: (a.normal_rate if k == 'normal' else min(1.0, a.cap / v)) for k, v in counts.items()}
    print('type counts', counts); print('rates', rates, flush=True)
    rng = np.random.default_rng(0)
    parts, manifest = [], []
    for p in files:
        df = pd.read_csv(p, usecols=lambda c: c in KEEP, low_memory=False, dtype=str)
        n0 = len(df)
        lab = df['label'].astype(int).values
        u = rng.random(n0)
        keep = u < df['type'].map(rates).values
        df = df[keep].copy()
        df['src_file'] = file_index(p)
        df['row_in_file'] = np.flatnonzero(keep)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        manifest.append({'file': os.path.basename(p), 'rows': n0, 'kept': int(keep.sum()), 'sha256': h,
                         'kept_types': {k: int(v) for k, v in df['type'].value_counts().items()}})
        parts.append(df)
        print(os.path.basename(p), n0, int(keep.sum()), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df['ts'] = df['ts'].astype(np.int64)
    df['label'] = df['label'].astype(np.int8)
    # stable sort by event time; ties keep file/row order
    df = df.sort_values(['ts', 'src_file', 'row_in_file'], kind='mergesort').reset_index(drop=True)
    df.to_pickle(a.out)
    json.dump({'files': manifest, 'cap': a.cap, 'type_counts_full': counts, 'rates': rates, 'normal_rate': a.normal_rate,
               'seed': 0, 'rows_out': len(df),
               'type_counts': df['type'].value_counts().to_dict()},
              open(a.out + '.manifest.json', 'w'), indent=2)
    print(df['type'].value_counts())


if __name__ == '__main__':
    main()
