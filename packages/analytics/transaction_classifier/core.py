"""Conservative rules followed by optional, confidence-gated ML fallback."""
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from collections.abc import Mapping
from math import isfinite, isclose
import re
import pandas as pd
from pandas.api.types import is_bool, is_scalar
from .rules import RULES, COUNTERPARTY_RULES, INTERNAL, PERSONAL_TRANSFER, TAX_BOILERPLATE

CATEGORIES = ('revenue', 'other_income', 'purchases', 'salary', 'rent', 'taxes', 'bank_fees', 'other_expense', 'internal_transfer')
ALLOWED = {'incoming': {'revenue', 'other_income', 'internal_transfer'}, 'outgoing': set(CATEGORIES) - {'revenue', 'other_income'}}

def clean(value):
    if not is_scalar(value) or value is None or pd.isna(value):
        return ''
    return ' '.join(str(value).lower().replace('ё', 'е').split())

@dataclass
class TransactionClassifier:
    threshold: float = .8
    fields: dict = field(default_factory=dict)
    own_inns: set = field(default_factory=set)
    model: object = None

    def __post_init__(self):
        try:
            if is_bool(self.threshold):
                raise ValueError()
            self.threshold = float(self.threshold)
            if not isfinite(self.threshold) or not 0 <= self.threshold <= 1:
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise ValueError('threshold must be a finite number between 0 and 1') from exc
        self.fields = dict(self.fields or getattr(self.model, 'fields', {}))
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

    def _classify_rules(self, transaction):
        """Return the result and whether an unambiguous ML fallback is allowed."""
        if not isinstance(transaction, Mapping):
            raise TypeError('transaction must be a mapping')
        result = dict(transaction)
        result.update(category='unknown', category_confidence=0., classification_method='unknown')
        direction, purpose, name, inn, valid = self.normalize(transaction)
        if not valid:
            return result, False
        if re.search(PERSONAL_TRANSFER, purpose):
            return result, False
        # "Без налога (НДС)" is a VAT note, not evidence of a tax payment.
        rule_purpose = re.sub(TAX_BOILERPLATE, '', purpose)
        # Text alone is insufficient to establish ownership of both accounts.
        internal_hint = bool(re.search(INTERNAL, purpose))
        own = inn in self.own_inns if inn else False
        hits = [(cat, conf) for cat, d, pattern, conf in RULES
                if direction == d and re.search(pattern, rule_purpose)]
        hits.extend((cat, conf) for cat, d, pattern, conf in COUNTERPARTY_RULES
                    if direction == d and re.search(pattern, name))
        if own and internal_hint and not hits:
            hits = [('internal_transfer', .99)]
        elif internal_hint or own:
            return result, False
        if len({cat for cat, _ in hits}) > 1:
            return result, False
        if hits:
            category, confidence = max(hits, key=lambda hit: hit[1])
            if confidence >= self.threshold:
                result.update(category=category, category_confidence=confidence, classification_method='rule')
                return result, False
            result['category_confidence'] = confidence
        # An unconfirmed ownership hint must never become a learned own transfer.
        return result, not (internal_hint or own)

    def _apply_model(self, records, results, pending):
        if self.model is None or not pending:
            return
        probabilities = self.model.probabilities([records[i] for i in pending], self)
        if len(probabilities) != len(pending):
            raise ValueError('Model must return one probability mapping per transaction')
        for index, scores in zip(pending, probabilities):
            if not isinstance(scores, Mapping) or not scores or not set(scores).issubset(CATEGORIES):
                raise ValueError('Model returned unsupported categories or empty probabilities')
            try:
                scores = {category: float(value) for category, value in scores.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError('Model probabilities must be finite numbers between 0 and 1') from exc
            if any(not isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
                raise ValueError('Model probabilities must be finite numbers between 0 and 1')
            confidence = max(scores.values())
            candidates = [category for category, value in scores.items()
                          if isclose(value, confidence, rel_tol=0, abs_tol=1e-12)]
            result = results[index]
            result['category_confidence'] = confidence
            if len(candidates) != 1:
                continue
            category = candidates[0]
            direction = self.normalize(records[index])[0]
            if category in ALLOWED[direction] and category != 'internal_transfer' and confidence >= self.threshold:
                result.update(category=category, classification_method='ml')

    def classify_transaction(self, transaction):
        return self.classify_transactions([transaction])[0]

    def classify_transactions(self, transactions):
        is_frame = isinstance(transactions, pd.DataFrame)
        if isinstance(transactions, (Mapping, str, bytes)):
            raise TypeError('transactions must be a sequence of mappings or a DataFrame')
        if is_frame and not transactions.columns.is_unique:
            raise ValueError('DataFrame columns must be unique')
        records = transactions.to_dict('records') if is_frame else list(transactions)
        results, pending = [], []
        for index, record in enumerate(records):
            result, eligible = self._classify_rules(record)
            results.append(result)
            if eligible:
                pending.append(index)
        self._apply_model(records, results, pending)
        if is_frame:
            result = transactions.copy()
            for column in ('category', 'category_confidence', 'classification_method'):
                result[column] = [row[column] for row in results]
            return result
        return results


def classify_transaction(transaction, **kwargs):
    return TransactionClassifier(**kwargs).classify_transaction(transaction)


def classify_transactions(transactions, **kwargs):
    return TransactionClassifier(**kwargs).classify_transactions(transactions)
