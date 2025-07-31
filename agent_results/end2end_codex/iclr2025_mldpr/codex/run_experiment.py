#!/usr/bin/env python3
"""
Automated experiment for Adaptive Deprecation Scoring
"""
import os
import csv
import math
import datetime
import logging

import openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
DATASETS = [61, 554]  # OpenML dataset IDs: iris, mnist_784
ALPHA = 1.0
TAU_WARN = 0.9
TAU_DEP = 0.95

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def compute_citation_signal(upload_date, now, max_age_days):
    age_days = (now - upload_date).days
    # normalize age: older => higher risk
    s = min(1.0, age_days / max_age_days)
    return s, age_days

def compute_update_signal(version, max_version):
    try:
        v = float(version)
    except:
        v = 0.0
    s = min(1.0, max(0.0, (max_version - v) / max_version)) if max_version > 0 else 0.0
    return s

def compute_reproducibility_signal(X, y):
    # train simple logistic regression, low accuracy means failure
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    # failure if acc < 0.6
    return float(acc < 0.6), acc

def main():
    os.makedirs('codex', exist_ok=True)
    log_path = os.path.join('codex', 'log.txt')
    setup_logging(log_path)
    logging.info('Starting experiment')
    now = datetime.datetime.utcnow()

    records = []
    # compute max_age_days over datasets
    upload_dates = []
    for did in DATASETS:
        ds = openml.datasets.get_dataset(did)
        dt = datetime.datetime.fromisoformat(ds.upload_date)
        upload_dates.append(dt)
    ages = [(now - d).days for d in upload_dates]
    max_age = max(ages) or 1
    # version signal not used in this minimal experiment: skip
    for did in DATASETS:
        logging.info(f'Processing dataset {did}')
        ds = openml.datasets.get_dataset(did)
        name = ds.name

        # Citation age signal
        upload_dt = datetime.datetime.fromisoformat(ds.upload_date)
        S_cite, age_days = compute_citation_signal(upload_dt, now, max_age)
        # Update frequency (proxy): skip, set zero
        S_upd = 0.0
        # Community issues: not implemented
        S_iss = 0.0
        # FAIR drift: not implemented
        S_fair = 0.0

        # Reproducibility signal
        X, y, *_ = ds.get_data(target=ds.default_target_attribute)
        # drop non-numeric columns
        X = pd.DataFrame(X)
        X = X.select_dtypes(include=[np.number]).fillna(0)
        try:
            S_rep, acc = compute_reproducibility_signal(X, y)
        except Exception as e:
            logging.warning(f'Failed reproducibility test on {did}: {e}')
            S_rep, acc = 1.0, 0.0

        # Combine signals with equal weights
        weights = {'cite': 1.0}
        D = weights['cite'] * S_cite

        logging.info(f'{name}: age_days={age_days}, S_cite={S_cite:.3f}, S_rep={S_rep}, D={D:.3f}')
        records.append({
            'did': did,
            'name': name,
            'age_days': age_days,
            'S_cite': S_cite,
            'S_upd': S_upd,
            'S_iss': S_iss,
            'S_rep': S_rep,
            'S_fair': S_fair,
            'D': D
        })

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join('codex', 'results.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f'Results saved to {csv_path}')

    # Plot deprecation scores
    plt.figure()
    plt.bar(df['name'], df['D'], color='orange')
    plt.ylabel('Deprecation Score D')
    plt.title('Deprecation Scores for Datasets')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    for i, v in enumerate(df['D']):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    fig_path = os.path.join('codex', 'deprecation_scores.png')
    plt.savefig(fig_path)
    logging.info(f'Figure saved to {fig_path}')
    logging.info('Experiment completed')

if __name__ == '__main__':
    main()
