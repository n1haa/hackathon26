from forecasting import forecast_cashflow
from transaction_classifier import TransactionClassifier, MLClassifier, train_model


def test_existing_classifier_output_can_feed_forecast():
    classifier = TransactionClassifier()
    records = classifier.classify_transactions([
        {"date": "2026-01-01", "direction": "incoming", "amount": 100, "purpose": "Оплата за услуги"},
        {"date": "2026-01-01", "direction": "outgoing", "amount": 40, "purpose": "Аренда помещения"}])
    assert [r["category"] for r in records] == ["revenue", "rent"]
    result = forecast_cashflow(records, 1000)
    assert result["forecast"][0]["predicted_balance"] == 1060
    assert callable(train_model)
    assert isinstance(MLClassifier(), MLClassifier)


def test_existing_classifier_verifies_own_transfer_upstream():
    classifier = TransactionClassifier(own_inns={"own"})
    record = classifier.classify_transaction({"date": "2026-01-01", "direction": "outgoing", "amount": 50,
                                              "counterparty_inn": "own", "purpose": "Между своими счетами"})
    assert record["category"] == "internal_transfer"
    assert forecast_cashflow([record], 1000)["forecast"][0]["predicted_balance"] == 1000
