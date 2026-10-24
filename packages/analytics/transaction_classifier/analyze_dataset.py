"""Classify the complete participant dataset and report coverage, not accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import TransactionClassifier, clean
from .dataset import load_participant_dataset, classify_participant_transactions


def analyze_dataset(dataset_path, output_path, *, threshold=.8):
    root, output = Path(dataset_path), Path(output_path)
    operations, companies = load_participant_dataset(root)
    if operations.empty:
        raise ValueError('Dataset must contain at least one operation')
    classified = classify_participant_transactions(operations, companies, TransactionClassifier(threshold=threshold))
    amounts = pd.to_numeric(operations['amount'], errors='coerce')
    known = classified['category'].ne('unknown')
    by_company = []
    for company_id, rows in classified.groupby('company_id', sort=True):
        by_company.append({'company_id': company_id, 'rows': len(rows),
                           'classified_rows': int(rows['category'].ne('unknown').sum()),
                           'coverage': float(rows['category'].ne('unknown').mean())})
    digest = hashlib.sha256()
    with (root / 'all_operations.csv').open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    dates = pd.to_datetime(operations['operation_date'], errors='coerce')
    summary = {
        'data_source': 'Organizer-provided synthetic participant bank statements; not real companies',
        'source_file': 'all_operations.csv', 'source_sha256': digest.hexdigest(),
        'rows': len(operations), 'companies': int(operations['company_id'].nunique()),
        'date_min': str(dates.min().date()), 'date_max': str(dates.max().date()),
        'threshold': threshold,
        'empty_purpose_rows': int(operations['payment_purpose'].map(clean).eq('').sum()),
        'empty_counterparty_inn_rows': int(operations['counterparty_inn'].map(clean).eq('').sum()),
        'invalid_amount_rows': int((~np.isfinite(amounts) | amounts.le(0)).sum()),
        'invalid_date_rows': int(dates.isna().sum()),
        'duplicate_rows': int(operations.duplicated().sum()),
        'duplicate_operation_ids': int(operations.duplicated(['company_id', 'operation_id']).sum()),
        'duplicate_payload_rows': int(operations.drop(columns='operation_id').duplicated().sum()),
        'category_counts': {str(k): int(v) for k, v in classified['category'].value_counts().items()},
        'method_counts': {str(k): int(v) for k, v in classified['classification_method'].value_counts().items()},
        'classified_rows': int(known.sum()), 'unknown_rows': int((~known).sum()),
        'coverage': float(known.mean()), 'unknown_fraction': float((~known).mean()),
        'reference_labels_present': False, 'quality_metrics': None,
        'quality_status': 'No reference category column: accuracy/F1/precision/recall/confusion matrix are not measurable. Rules are not reference labels.',
        'ml_training': 'Not performed: no independent reference labels are available.',
        'confidence_scope': 'Expert rule scores; not calibrated probabilities or measured precision',
        'balance_scope': 'balance_eod is end-of-day, unchanged; no per-operation balance reinterpretation',
        'per_company': by_company,
    }
    output.mkdir(parents=True, exist_ok=True)
    classified.to_csv(output / 'classified_operations.csv', sep=';', encoding='utf-8-sig', index=False)
    unknown = classified.loc[~known].drop_duplicates(['direction', 'payment_purpose', 'counterparty_name'])
    unknown.head(200).to_csv(output / 'unknown_review_sample.csv', sep=';', encoding='utf-8-sig', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/classifier_dataset'))
    parser.add_argument('--threshold', type=float, default=.8)
    args = parser.parse_args()
    summary = analyze_dataset(args.dataset, args.output, threshold=args.threshold)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_company'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
