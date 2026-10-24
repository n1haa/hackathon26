import json

import pandas as pd
import pytest

from transaction_classifier import TransactionClassifier, load_participant_dataset, classify_participant_transactions
from transaction_classifier.analyze_dataset import analyze_dataset


def dataset():
    companies = pd.DataFrame([{'company_id': 'a', 'company_inn': '00123'},
                              {'company_id': 'b', 'company_inn': '00456'}])
    base = dict(operation_date='2026-01-01', amount='10', direction='Списание',
                counterparty_inn='00123', counterparty_name='Компания', balance_eod='100.00')
    operations = pd.DataFrame([
        dict(base, company_id='a', operation_id='a1', payment_purpose='Перевод собственных средств на расчетный счет'),
        dict(base, company_id='b', operation_id='b1', payment_purpose='Перевод собственных средств на расчетный счет'),
        dict(base, company_id='a', operation_id='a2', counterparty_inn='00999', counterparty_name='ПАО Банк', payment_purpose='Комиссия за ведение счета'),
    ], index=[4, 4, 3])
    return operations, companies


def test_participant_adapter_uses_each_company_inn_and_preserves_eod():
    operations, companies = dataset()
    before = operations.copy(deep=True)
    result = classify_participant_transactions(operations, companies)
    assert result['category'].tolist() == ['internal_transfer', 'unknown', 'bank_fees']
    assert 'closing_balance' not in result
    assert result['balance_eod'].tolist() == ['100.00'] * 3
    pd.testing.assert_frame_equal(operations, before)
    pd.testing.assert_frame_equal(result[before.columns], before)


def test_read_and_full_report_preserve_leading_zeros_and_do_not_invent_quality(tmp_path):
    operations, companies = dataset()
    root, output = tmp_path / 'input', tmp_path / 'output'
    root.mkdir()
    operations.to_csv(root / 'all_operations.csv', sep=';', encoding='utf-8-sig', index=False)
    companies.to_csv(root / 'companies.csv', sep=';', encoding='utf-8-sig', index=False)
    loaded, metadata = load_participant_dataset(root)
    assert loaded.iloc[0]['counterparty_inn'] == metadata.iloc[0]['company_inn'] == '00123'
    summary = analyze_dataset(root, output)
    assert summary['rows'] == 3
    assert summary['coverage'] == 2 / 3
    assert summary['quality_metrics'] is None
    assert summary['reference_labels_present'] is False
    assert json.loads((output / 'summary.json').read_text(encoding='utf-8')) == summary
    assert (output / 'classified_operations.csv').is_file()
    assert (output / 'unknown_review_sample.csv').is_file()


def test_unknown_company_and_numeric_inn_are_rejected():
    operations, companies = dataset()
    with pytest.raises(ValueError, match='missing in companies'):
        classify_participant_transactions(operations, companies.iloc[:1])
    companies['company_inn'] = [123, 456]
    with pytest.raises(ValueError, match='strings'):
        classify_participant_transactions(operations, companies)


def test_shared_classifier_inns_do_not_leak_ownership_between_companies():
    operations, companies = dataset()
    classifier = TransactionClassifier(own_inns={'00123', '00456'})
    result = classify_participant_transactions(operations, companies, classifier)
    assert result['category'].tolist() == ['internal_transfer', 'unknown', 'bank_fees']
    assert classifier.own_inns == {'00123', '00456'}


@pytest.mark.parametrize('value', [[1, 2], {'inn': '00123'}])
def test_nonscalar_inns_are_rejected_clearly(value):
    operations, companies = dataset()
    operations.iat[2, operations.columns.get_loc('counterparty_inn')] = value
    with pytest.raises(ValueError, match='strings'):
        classify_participant_transactions(operations, companies)


def test_empty_dataset_analysis_fails_before_writing_output(tmp_path):
    operations, companies = dataset()
    operations.iloc[:0].to_csv(tmp_path / 'all_operations.csv', sep=';', encoding='utf-8-sig', index=False)
    companies.to_csv(tmp_path / 'companies.csv', sep=';', encoding='utf-8-sig', index=False)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='at least one operation'):
        analyze_dataset(tmp_path, output)
    assert not output.exists()
