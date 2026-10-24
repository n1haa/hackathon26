"""Adapter for the organizers' UTF-8-BOM, semicolon-separated statements."""
from dataclasses import replace
from pathlib import Path

import pandas as pd
from pandas.api.types import is_scalar

from .core import TransactionClassifier, clean


PARTICIPANT_FIELDS = {'date': 'operation_date', 'purpose': 'payment_purpose'}
REQUIRED_COLUMNS = {'company_id', 'operation_id', 'operation_date', 'amount', 'direction',
                    'counterparty_inn', 'counterparty_name', 'payment_purpose', 'balance_eod'}
OUTPUT_COLUMNS = ('category', 'category_confidence', 'classification_method')


def load_participant_dataset(path):
    """Read all_operations only; statements/ contains the same rows, not extra data."""
    root = Path(path)
    operations = pd.read_csv(root / 'all_operations.csv', sep=';', encoding='utf-8-sig',
                             dtype=str, keep_default_na=False)
    companies = pd.read_csv(root / 'companies.csv', sep=';', encoding='utf-8-sig',
                            dtype=str, keep_default_na=False)
    if not REQUIRED_COLUMNS.issubset(operations):
        raise ValueError(f'Missing statement columns: {sorted(REQUIRED_COLUMNS - set(operations))}')
    _company_inns(operations, companies)
    return operations, companies


def _company_inns(operations, companies):
    if 'company_id' not in operations or not {'company_id', 'company_inn'}.issubset(companies):
        raise ValueError('Statements need company_id; companies need company_id and company_inn')
    if companies['company_id'].map(clean).eq('').any() or companies['company_id'].duplicated().any():
        raise ValueError('Company identifiers must be unique and nonempty')
    if operations['company_id'].map(clean).eq('').any():
        raise ValueError('Operation company_id must not be empty')
    lookup = dict(zip(companies['company_id'], companies['company_inn']))
    if not operations['company_id'].isin(lookup).all():
        raise ValueError('Some operation company_id values are missing in companies')
    return lookup


def classify_participant_transactions(transactions, companies, classifier=None):
    """Preserve input columns/order/index and verify ownership per business.

    balance_eod is preserved and never renamed to per-transaction closing_balance.
    Supply INNs as strings: already lost leading zeros cannot be reconstructed.
    """
    frame, companies = pd.DataFrame(transactions).copy(), pd.DataFrame(companies).copy()
    if not frame.columns.is_unique or not companies.columns.is_unique:
        raise ValueError('DataFrame columns must be unique')
    if not REQUIRED_COLUMNS.issubset(frame):
        raise ValueError(f'Missing statement columns: {sorted(REQUIRED_COLUMNS - set(frame))}')
    for data, column in ((frame, 'counterparty_inn'), (companies, 'company_inn')):
        if column in data and not data[column].map(
                lambda value: isinstance(value, str) or (is_scalar(value) and pd.isna(value))).all():
            raise ValueError(f'{column} must be read as strings to preserve leading zeros')
    lookup = _company_inns(frame, companies)
    base = classifier or TransactionClassifier()
    result = frame.copy()
    result['category'] = 'unknown'
    result['category_confidence'] = 0.
    result['classification_method'] = 'unknown'
    positioned = frame.reset_index(drop=True)
    for company_id, rows in positioned.groupby('company_id', sort=False):
        own_inn = clean(lookup[company_id])
        current = replace(base, fields={**PARTICIPANT_FIELDS, **base.fields},
                          own_inns={own_inn} if own_inn else set())
        classified = current.classify_transactions(rows)
        for column in OUTPUT_COLUMNS:
            result.iloc[rows.index, result.columns.get_loc(column)] = classified[column].to_numpy()
    return result
