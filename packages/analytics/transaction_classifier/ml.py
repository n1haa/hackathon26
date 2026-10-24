"""Supervised training only; no synthetic labels or implicit test training."""
import hashlib
import json
from collections import Counter
from dataclasses import replace
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted
from .core import TransactionClassifier, CATEGORIES, ALLOWED
from .evaluation import classification_metrics, evaluate_classifier


def features(records, classifier):
    rows = []
    for row in records:
        direction, purpose, name, inn, valid = classifier.normalize(row)
        if not valid:
            raise ValueError('Training/evaluation requires positive finite amounts and known directions')
        rows.append({'text': f'purpose {purpose} counterparty {name} inn {inn}', 'direction': direction})
    return pd.DataFrame(rows, columns=['text', 'direction'])


class MLClassifier:
    def __init__(self, fields=None):
        self.fields = dict(fields or {})
        self.pipeline = Pipeline([
            ('features', ColumnTransformer([
                ('text', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), 'text'),
                ('direction', OneHotEncoder(handle_unknown='ignore'), ['direction'])])),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])

    def probabilities(self, records, classifier=None):
        classifier = classifier or TransactionClassifier(fields=getattr(self, 'fields', {}))
        records = records.to_dict('records') if isinstance(records, pd.DataFrame) else list(records)
        if not records:
            return []
        check_is_fitted(self.pipeline)
        scores = self.pipeline.predict_proba(features(records, classifier))
        return [dict(zip(self.pipeline.classes_, map(float, row))) for row in scores]

    def save(self, path):
        check_is_fitted(self.pipeline)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        # joblib/pickle is executable: only load artifacts from trusted sources.
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError('Unexpected model artifact')
        check_is_fitted(model.pipeline)
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
    model = MLClassifier(fields=classifier.fields)
    # Fingerprint the actual TF-IDF token counts, not raw strings. Whitespace,
    # punctuation and token-equivalent repeated operations must stay together.
    # Dates, amounts and balances are intentionally excluded.
    analyzer = model.pipeline.named_steps['features'].transformers[0][1].build_analyzer()
    fingerprints = [hashlib.sha256(json.dumps({'direction': row['direction'],
                    'tokens': sorted(Counter(analyzer(row['text'])).items())},
                    sort_keys=True).encode()).hexdigest() for row in x.to_dict('records')]
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
    model.pipeline.fit(x.iloc[train], labels.iloc[train])
    predicted = model.pipeline.predict(x.iloc[test])
    actual = labels.iloc[test]
    metrics = classification_metrics(actual, predicted, include_unknown=False)
    metrics.update({
        'train_rows': len(train), 'test_rows': len(test),
        'train_indices': train.tolist(), 'test_indices': test.tolist(),
        'test_missing_categories': sorted(set(labels) - set(actual)),
        'evaluation_scope': 'ML argmax before abstention; macro metrics over all supported categories',
        'grouping_scope': 'company and feature-equivalent duplicates' if company_column in frame else 'feature-equivalent duplicates only; company isolation is not guaranteed',
        'train_groups': len({groups[i] for i in train}), 'test_groups': len({groups[i] for i in test}),
        'hybrid': evaluate_classifier(frame.iloc[test], replace(classifier, model=model), label_column=label_column),
    })
    if company_column in frame:
        metrics['train_companies'] = sorted(map(str, frame.iloc[train][company_column].unique()))
        metrics['test_companies'] = sorted(map(str, frame.iloc[test][company_column].unique()))
    return model, metrics
