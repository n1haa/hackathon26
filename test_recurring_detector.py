"""Smoke tests for recurring_detector.py; run with: python -m unittest -v"""
import unittest
import pandas as pd
from recurring_detector import detect_recurring


class RecurringDetectorTests(unittest.TestCase):
    def test_monthly_recurring_payment(self):
        df = pd.DataFrame({
            "company_id": ["C001"] * 4,
            "operation_date": ["2026-01-05", "2026-02-05", "2026-03-05", "2026-04-06"],
            "amount": [50000, 50000, 52000, 50000],
            "direction": ["Списание"] * 4,
            "counterparty_inn": ["0012345678"] * 4,
            "counterparty_name": ["ООО Арендодатель"] * 4,
            "payment_purpose": ["Арендная плата за помещение"] * 4,
        })
        result = detect_recurring(df)
        self.assertTrue(result)
        self.assertEqual(result[0]["direction"], "Списание")
        self.assertEqual(result[0]["frequency"], "monthly")
        self.assertEqual(result[0]["occurrences_count"], 4)
        self.assertTrue(0 <= result[0]["confidence"] <= 1)
        self.assertNotIn("probability", result[0])

    def test_same_amount_different_counterparty_not_merged(self):
        df = pd.DataFrame({
            "operation_date": ["2026-01-01", "2026-02-01", "2026-03-01",
                               "2026-01-01", "2026-02-01", "2026-03-01"],
            "amount": [1000] * 6,
            "direction": ["Списание"] * 6,
            "counterparty_inn": ["111"] * 3 + ["222"] * 3,
            "counterparty_name": ["A"] * 3 + ["B"] * 3,
            "payment_purpose": ["Услуги"] * 6,
        })
        result = detect_recurring(df)
        self.assertGreaterEqual(len(result), 2)
        self.assertEqual({x["counterparty"] for x in result}, {"A", "B"})

    def test_empty_input(self):
        self.assertEqual(detect_recurring(pd.DataFrame()), [])

    def test_direction_not_mixed(self):
        df = pd.DataFrame({
            "operation_date": ["2026-01-01", "2026-02-01", "2026-03-01",
                               "2026-01-01", "2026-02-01", "2026-03-01"],
            "amount": [1000] * 6,
            "direction": ["Поступление"] * 3 + ["Списание"] * 3,
            "counterparty_inn": ["111"] * 6,
            "counterparty_name": ["A"] * 6,
            "payment_purpose": ["Оплата"] * 6,
        })
        result = detect_recurring(df)
        self.assertEqual({x["direction"] for x in result}, {"Поступление", "Списание"})


if __name__ == "__main__":
    unittest.main()
