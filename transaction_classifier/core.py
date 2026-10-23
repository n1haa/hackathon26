from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping
import re
import pandas as pd
from .rules import RULES, INTERNAL

CATEGORIES = ('revenue', 'other_income', 'purchases', 'salary', 'rent', 'taxes', 'bank_fees', 'other_expense', 'internal_transfer')
ALLOWED = {'incoming': {'revenue', 'other_income', 'internal_transfer'}, 'outgoing': set(CATEGORIES) - {'revenue', 'other_income'}}

def clean(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ''
    return str(value).lower().replace('ё', 'е').strip()

@dataclass
class TransactionClassifier:
    threshold: float = .8
    fields: dict = field(default_factory=dict)
    own_inns: set = field(default_factory=set)
    model: object = None

    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        self.own_inns = {clean(x) for x in self.own_inns if clean(x)}

    def value(self, row, name):
        return row.get(self.fields.get(name, name))

    def normalize(self, row):
        direction = clean(self.value(row, 'direction'))
        direction = {'поступление': 'incoming', 'списание': 'outgoing'}.get(direction, direction)
        try:
            raw = self.value(row, 'amount')
            if isinstance(raw, bool):
                raise ValueError()
            amount = Decimal(str(raw))
            valid = amount.is_finite() and amount > 0
        except (InvalidOperation, ValueError, TypeError):
            valid = False
        purpose = clean(self.value(row, 'purpose'))
        name = clean(self.value(row, 'counterparty_name'))
        inn = clean(self.value(row, 'counterparty_inn'))
        return direction, purpose, name, inn, valid and direction in ALLOWED

    def classify_transaction(self, transaction):
        if not isinstance(transaction, Mapping):
            raise TypeError('transaction must be a mapping')
        result = dict(transaction)
        result.update(category='unknown', category_confidence=0., classification_method='unknown')
        direction, purpose, name, inn, valid = self.normalize(transaction)
        if not valid:
            return result
        # Text alone is insufficient to establish ownership of both accounts.
        internal_hint = bool(re.search(INTERNAL, purpose))
        own = inn in self.own_inns if inn else False
        hits = [(cat, conf) for cat, d, pattern, conf in RULES
                if direction == d and re.search(pattern, purpose)]
        if own and internal_hint and not hits:
            hits = [('internal_transfer', .99)]
        elif internal_hint or own:
            return result
        if len({cat for cat, _ in hits}) > 1:
            return result
        if hits:
            category, confidence = hits[0]
            if confidence >= self.threshold:
                result.update(category=category, category_confidence=confidence, classification_method='rule')
                return result
            result['category_confidence'] = confidence
        if self.model is not None:
            probabilities = self.model.probabilities([transaction], self)[0]
            category = max(probabilities, key=probabilities.get)
            confidence = probabilities[category]
            result['category_confidence'] = confidence
            if category in ALLOWED[direction] and category != 'internal_transfer' and confidence >= self.threshold:
                result.update(category=category, classification_method='ml')
        return result

    def classify_transactions(self, transactions):
        if isinstance(transactions, pd.DataFrame):
            records = [self.classify_transaction(row) for row in transactions.to_dict('records')]
            result = transactions.copy()
            for column in ('category', 'category_confidence', 'classification_method'):
                result[column] = [row[column] for row in records]
            return result
        return [self.classify_transaction(row) for row in transactions]


def classify_transaction(transaction, **kwargs):
    return TransactionClassifier(**kwargs).classify_transaction(transaction)


def classify_transactions(transactions, **kwargs):
    return TransactionClassifier(**kwargs).classify_transactions(transactions)
