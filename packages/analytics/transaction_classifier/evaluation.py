"""Metrics require explicit reference labels; abstentions count as errors."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from .core import CATEGORIES, TransactionClassifier


def classification_metrics(actual, predicted, *, include_unknown=True):
    order = list(CATEGORIES) + (['unknown'] if include_unknown else [])
    actual, predicted = list(actual), list(predicted)
    return {
        'accuracy': float(accuracy_score(actual, predicted)),
        'macro_f1': float(f1_score(actual, predicted, labels=list(CATEGORIES), average='macro', zero_division=0)),
        'precision': float(precision_score(actual, predicted, labels=list(CATEGORIES), average='macro', zero_division=0)),
        'recall': float(recall_score(actual, predicted, labels=list(CATEGORIES), average='macro', zero_division=0)),
        'labels': order,
        'confusion_matrix': confusion_matrix(actual, predicted, labels=order).tolist(),
        'support': {label: actual.count(label) for label in CATEGORIES},
    }


def evaluate_classifier(data, classifier=None, *, label_column='category'):
    """Evaluate the complete rules/ML/abstention path on supplied held-out data.

    The caller must keep this data out of training and threshold selection.
    Macro metrics average all nine supported categories, including absent ones.
    """
    frame = pd.DataFrame(data).copy()
    if frame.empty or label_column not in frame:
        raise ValueError('Nonempty explicitly labeled evaluation data required')
    labels = frame[label_column].copy()
    if not labels.isin(CATEGORIES).all():
        raise ValueError('Evaluation labels must be supported categories')
    result = (classifier or TransactionClassifier()).classify_transactions(frame)
    metrics = classification_metrics(labels, result['category'])
    metrics.update(rows=len(frame), coverage=float(result['category'].ne('unknown').mean()),
                   unknown_rows=int(result['category'].eq('unknown').sum()),
                   methods={str(k): int(v) for k, v in result['classification_method'].value_counts().items()},
                   evaluation_scope='Full classifier including abstentions; macro over all nine supported categories')
    return metrics
