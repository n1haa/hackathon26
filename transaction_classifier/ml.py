"""Supervised training only; no synthetic labels or implicit test training."""
import hashlib
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from .core import TransactionClassifier, CATEGORIES, ALLOWED


def features(records, classifier):
    rows = []
    for row in records:
        direction, purpose, name, inn, valid = classifier.normalize(row)
        if not valid:
            raise ValueError('Training/evaluation requires positive finite amounts and known directions')
        rows.append({'text': f'purpose {purpose} counterparty {name} inn {inn}', 'direction': direction})
    return pd.DataFrame(rows, columns=['text', 'direction'])


class MLClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('text', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), 'text'),
                ('direction', OneHotEncoder(handle_unknown='ignore'), ['direction'])])),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])

    def probabilities(self, records, classifier=None):
        classifier = classifier or TransactionClassifier()
        scores = self.pipeline.predict_proba(features(records, classifier))
        return [dict(zip(self.pipeline.classes_, map(float, row))) for row in scores]

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        # joblib/pickle is executable: only load artifacts from trusted sources.
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError('Unexpected model artifact')
        return model


def train_model(data, label_column='category', company_column='company_id',
                classifier=None, test_size=.25, random_state=42):
    """Company-disjoint split; duplicate feature groups also cannot cross split."""
    classifier = classifier or TransactionClassifier()
    frame = pd.DataFrame(data).reset_index(drop=True)
    if label_column not in frame or frame.empty:
        raise ValueError('Nonempty explicitly labeled data required')
    labels = frame[label_column]
    if not labels.isin(CATEGORIES).all() or labels.nunique() < 2:
        raise ValueError('At least two supported labeled categories required')
    records = frame.to_dict('records')
    x = features(records, classifier)
    for label, direction in zip(labels, x.direction):
        if label not in ALLOWED[direction]:
            raise ValueError('Label conflicts with direction')
    # Fingerprint intentionally ignores date, amount and balance: recurring near-identical
    # text must not inflate held-out quality, even on different dates.
    fingerprints = [hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                    for row in x.to_dict('records')]
    parent = list(range(len(frame)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[root(i)] = root(j)
    for values in ([fingerprints] + ([frame[company_column].tolist()] if company_column in frame else [])):
        seen = {}
        for i, value in enumerate(values):
            if pd.isna(value) or str(value).strip() == '':
                raise ValueError('Missing company identifier; supply reliable grouping')
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    groups = [root(i) for i in range(len(frame))]
    if len(set(groups)) < 2:
        raise ValueError('Not enough independent groups for leakage-free evaluation')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                      random_state=random_state).split(x, labels, groups))
    if set(labels.iloc[train]) != set(labels):
        raise ValueError('Training split misses categories; supply more independent groups')
    model = MLClassifier()
    model.pipeline.fit(x.iloc[train], labels.iloc[train])
    predicted = model.pipeline.predict(x.iloc[test])
    actual = labels.iloc[test]
    order = list(CATEGORIES)
    metrics = {
        'accuracy': float(accuracy_score(actual, predicted)),
        'macro_f1': float(f1_score(actual, predicted, labels=order, average='macro', zero_division=0)),
        'precision': float(precision_score(actual, predicted, labels=order, average='macro', zero_division=0)),
        'recall': float(recall_score(actual, predicted, labels=order, average='macro', zero_division=0)),
        'labels': order, 'confusion_matrix': confusion_matrix(actual, predicted, labels=order).tolist(),
        'train_rows': len(train), 'test_rows': len(test),
        'train_indices': train.tolist(), 'test_indices': test.tolist(),
        'test_missing_categories': sorted(set(labels) - set(actual)),
        'evaluation_scope': 'ML argmax before abstention; macro metrics over all supported categories'}
    return model, metrics
