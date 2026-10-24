"""Explicit small fixtures test behavior; they do not estimate dataset accuracy."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from transaction_classifier import (CATEGORIES, MLClassifier, TransactionClassifier,
                                    classify_transaction, classify_transactions,
                                    evaluate_classifier, train_model)


def operation(purpose='', direction='outgoing', **extra):
    return dict(date='2026-10-24', amount='100.00', direction=direction,
                purpose=purpose, counterparty_name='', counterparty_inn='',
                balance_eod='1000.00', **extra)


@pytest.mark.parametrize('purpose,direction,category', [
    ('Оплата за услуги', 'incoming', 'revenue'),
    ('Зачисление по операциям оплаты через СБП (QR-код)', 'incoming', 'revenue'),
    ('Оплата за перевозку груза согласно счету №1', 'incoming', 'revenue'),
    ('Возврат займа по договору', 'incoming', 'other_income'),
    ('Закупка сырья', 'outgoing', 'purchases'),
    ('Оплата по счету за канцтовары и хозтовары. Без налога (НДС)', 'outgoing', 'purchases'),
    ('Перечисление заработной платы (аванс)', 'outgoing', 'salary'),
    ('Аренда помещения', 'outgoing', 'rent'),
    ('Единый налоговый платёж', 'outgoing', 'taxes'),
    ('ЕНП', 'outgoing', 'taxes'),
    ('Комиссия за ведение счета за сентябрь. Без НДС', 'outgoing', 'bank_fees'),
    ('Благотворительный взнос', 'outgoing', 'other_expense'),
    ('Оплата по счету за бухгалтерские услуги. Без налога (НДС)', 'outgoing', 'other_expense'),
])
def test_rules_cover_supported_categories(purpose, direction, category):
    result = classify_transaction(operation(purpose, direction))
    assert result['category'] == category
    assert result['classification_method'] == 'rule'
    assert .8 <= result['category_confidence'] <= 1


@pytest.mark.parametrize('purpose', ['', 'Оплата по договору №42', 'Без налога (НДС)', 'Оплата по УПД №2. Без налога (НДС)',
                                    'Выплата дивидендов за 1 квартал. НДФЛ удержан. НДС не облагается'])
def test_insufficient_purpose_and_vat_notes_are_unknown(purpose):
    result = classify_transaction(operation(purpose))
    assert result['category'] == result['classification_method'] == 'unknown'
    assert result['purpose'] == purpose


@pytest.mark.parametrize('value', [None, pd.NA, np.nan, [], {}, np.array([1, 2])])
def test_missing_or_malformed_text_does_not_crash(value):
    row = operation()
    row.update(purpose=value, counterparty_name=value, counterparty_inn=value)
    assert classify_transaction(row)['category'] == 'unknown'


def test_missing_counterparty_and_unknown_inn_do_not_block_clear_purpose():
    row = operation('Аренда помещения')
    row.pop('counterparty_name')
    row['counterparty_inn'] = 'unknown-anonymized-inn'
    assert classify_transaction(row)['category'] == 'rent'


def test_explicit_tax_authority_name_and_conflict():
    row = operation('По реквизитам')
    row['counterparty_name'] = 'Казначейство России (ФНС России)'
    assert classify_transaction(row)['category'] == 'taxes'
    row['purpose'] = 'Аренда помещения'
    assert classify_transaction(row)['category'] == 'unknown'
    row['counterparty_name'] = 'ООО Налоговый консультант'
    row['purpose'] = 'По договору'
    assert classify_transaction(row)['category'] == 'unknown'


@pytest.mark.parametrize('amount', [0, -1, True, np.bool_(True), None, '', 'bad', 'NaN', 'Infinity', np.inf])
def test_invalid_amounts_are_unknown(amount):
    row = operation('Аренда помещения')
    row['amount'] = amount
    assert classify_transaction(row)['category'] == 'unknown'


@pytest.mark.parametrize('direction', ['sideways', None, '', pd.NA])
def test_invalid_direction_is_unknown(direction):
    assert classify_transaction(operation('Аренда помещения', direction))['category'] == 'unknown'


def test_russian_directions_and_preservation():
    row = operation('  Оплата за услуги Ёж  ', 'Поступление')
    before = copy.deepcopy(row)
    result = classify_transaction(row)
    assert result['category'] == 'revenue'
    assert row == before
    assert all(result[key] == value for key, value in before.items())


def test_own_transfer_requires_inn_and_purpose_and_never_amount_only():
    classifier = TransactionClassifier(own_inns={'001234'})
    row = operation('Перевод собственных средств на расчетный счет в другом банке')
    assert classifier.classify_transaction(row)['category'] == 'unknown'
    row['counterparty_inn'] = '001234'
    assert classifier.classify_transaction(row)['category'] == 'internal_transfer'
    row['purpose'] = ''
    assert classifier.classify_transaction(row)['category'] == 'unknown'
    row['counterparty_inn'] = 'different'
    assert classifier.classify_transaction(row)['category'] == 'unknown'


@pytest.mark.parametrize('purpose', ['Перевод собственных средств на личную карту',
                                    'Перевод собственных средств ИП на счет физического лица',
                                    'Перевод на личный счет предпринимателя'])
def test_personal_owner_withdrawal_is_not_business_internal_transfer(purpose):
    row = operation(purpose)
    row['counterparty_inn'] = '001234'
    assert TransactionClassifier(own_inns={'001234'}).classify_transaction(row)['category'] == 'unknown'


def test_threshold_abstains_but_retains_candidate_score():
    result = classify_transaction(operation('Аренда'), threshold=.95)
    assert result['category'] == result['classification_method'] == 'unknown'
    assert result['category_confidence'] == .94


@pytest.mark.parametrize('threshold', [-.1, 1.1, None, 'bad', np.nan, True, np.bool_(True)])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match='threshold'):
        TransactionClassifier(threshold=threshold)


class FixedModel:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def probabilities(self, records, classifier):
        self.calls.append(records)
        return [self.scores.copy() for _ in records]


def test_ml_fallback_is_batched_and_skips_ambiguous_invalid_and_rule_rows():
    model = FixedModel({'rent': .95, 'purchases': .05})
    result = classify_transactions([operation(), operation('По договору'), operation('Зарплата и аренда'),
                                    operation('Аренда'), {**operation(), 'amount': -1}], model=model)
    assert [r['classification_method'] for r in result] == ['ml', 'ml', 'unknown', 'rule', 'unknown']
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 2


@pytest.mark.parametrize('scores,threshold', [({'rent': .5, 'salary': .5}, .5),
                                            ({'internal_transfer': .99, 'rent': .01}, .8),
                                            ({'revenue': .99, 'rent': .01}, .8),
                                            ({'rent': .7, 'salary': .3}, .8)])
def test_ml_ties_direction_constraints_internal_and_threshold_abstain(scores, threshold):
    result = classify_transaction(operation(), model=FixedModel(scores), threshold=threshold)
    assert result['category'] == result['classification_method'] == 'unknown'
    assert result['category_confidence'] == max(scores.values())


@pytest.mark.parametrize('scores', [{}, {'bad': .9}, {'rent': np.nan}, {'rent': -1}, {'rent': 1.1}])
def test_invalid_model_probabilities_are_rejected(scores):
    with pytest.raises(ValueError, match='probabilit|categories'):
        classify_transaction(operation(), model=FixedModel(scores))


def test_dataframe_index_order_input_and_original_fields_are_preserved():
    frame = pd.DataFrame([operation('Аренда'), operation('Зарплата')], index=[8, 8])
    before = frame.copy(deep=True)
    result = classify_transactions(frame)
    assert result.index.tolist() == [8, 8]
    assert result['category'].tolist() == ['rent', 'salary']
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result[before.columns], before)
    assert classify_transactions([]) == []
    assert list(classify_transactions(frame.iloc[:0]).columns[-3:]) == ['category', 'category_confidence', 'classification_method']


def test_field_adapter():
    row = {'Сумма': '42', 'Направление': 'Списание', 'Назначение': 'Аренда', 'ИНН': '0001'}
    result = classify_transaction(row, fields={'amount': 'Сумма', 'direction': 'Направление', 'purpose': 'Назначение', 'counterparty_inn': 'ИНН'})
    assert result['category'] == 'rent'
    assert result['ИНН'] == '0001'


def labeled_fixture():
    records = []
    # Explicitly specified test labels, never promoted to a training dataset.
    for company in range(12):
        for category, purpose in [('rent', 'Аренда офиса'), ('salary', 'Заработная плата')]:
            row = operation(purpose)
            row.update(company_id=f'company-{company}', counterparty_inn=f'00{company:04}',
                       counterparty_name=f'Контрагент companytoken{company}', category=category)
            records.extend([row.copy(), {**row, 'date': '2026-10-25', 'amount': '200'}])
    return pd.DataFrame(records)


def test_ml_training_group_isolation_metrics_and_vocabulary_use_train_only(tmp_path):
    frame = labeled_fixture()
    model, metrics = train_model(frame)
    assert set(metrics['train_indices']).isdisjoint(metrics['test_indices'])
    assert set(metrics['train_companies']).isdisjoint(metrics['test_companies'])
    vocabulary = model.pipeline.named_steps['features'].named_transformers_['text'].vocabulary_
    for company in metrics['test_companies']:
        assert 'companytoken' + company.split('-')[-1] not in vocabulary
    assert len(metrics['labels']) == len(CATEGORIES)
    assert np.asarray(metrics['confusion_matrix']).sum() == metrics['test_rows']
    assert len(metrics['hybrid']['labels']) == len(CATEGORIES) + 1
    assert set(['accuracy', 'macro_f1', 'precision', 'recall']).issubset(metrics)
    json.dumps(metrics, allow_nan=False)
    before = model.probabilities(frame.iloc[:2])
    artifact = tmp_path / 'model.joblib'
    model.save(artifact)
    loaded = MLClassifier.load(artifact)
    assert loaded.probabilities(frame.iloc[:2]) == before
    assert loaded.probabilities([]) == []


def test_token_equivalent_duplicates_link_companies_transitively():
    frame = labeled_fixture()
    for index, purpose in [(0, 'Оплата аренды офиса'), (4, 'ОПЛАТА,   АРЕНДЫ ОФИСА!!!')]:
        frame.loc[index, ['purpose', 'counterparty_inn', 'counterparty_name']] = [purpose, '00123', 'ООО Альфа']
    _, metrics = train_model(frame)
    train = set(metrics['train_indices'])
    assert (0 in train) == (4 in train)
    assert ('company-0' in metrics['train_companies']) == ('company-1' in metrics['train_companies'])


def test_ml_field_mapping_survives_save_load(tmp_path):
    frame = labeled_fixture().rename(columns={'purpose': 'Описание'})
    model, _ = train_model(frame, classifier=TransactionClassifier(fields={'purpose': 'Описание'}))
    artifact = tmp_path / 'mapped.joblib'
    model.save(artifact)
    classifier = TransactionClassifier(model=MLClassifier.load(artifact))
    assert classifier.fields == {'purpose': 'Описание'}
    assert classifier.classify_transaction(frame.iloc[0].to_dict())['category'] == 'rent'


@pytest.mark.parametrize('mode', ['no_labels', 'invalid_label', 'invalid_amount', 'direction_conflict', 'missing_company', 'single_group'])
def test_invalid_or_unlabeled_training_is_rejected(mode):
    frame = labeled_fixture()
    if mode == 'no_labels':
        frame = frame.drop(columns='category')
    elif mode == 'invalid_label':
        frame.loc[0, 'category'] = 'unknown'
    elif mode == 'invalid_amount':
        frame.loc[0, 'amount'] = -1
    elif mode == 'direction_conflict':
        frame.loc[0, 'direction'] = 'incoming'
    elif mode == 'missing_company':
        frame.loc[0, 'company_id'] = None
    else:
        frame['company_id'] = 'same'
    with pytest.raises(ValueError):
        train_model(frame)


def test_fully_unfitted_model_cannot_be_used_or_saved(tmp_path):
    model = MLClassifier()
    with pytest.raises(NotFittedError):
        model.probabilities([operation()])
    with pytest.raises(NotFittedError):
        model.save(tmp_path / 'invalid.joblib')


def test_full_classifier_evaluation_includes_abstentions_as_errors():
    frame = pd.DataFrame([{**operation('Аренда'), 'category': 'rent'},
                          {**operation('Неизвестно'), 'category': 'salary'}])
    metric = evaluate_classifier(frame)
    assert metric['accuracy'] == .5
    assert metric['coverage'] == .5
    assert metric['unknown_rows'] == 1
    assert np.asarray(metric['confusion_matrix']).sum() == 2
    with pytest.raises(ValueError, match='labeled'):
        evaluate_classifier(frame.drop(columns='category'))
