# Подключение задач №2 и №4 к forecasting

Команды выполняются из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q -rx
.\.venv\Scripts\python.exe -m cashflow_integrated.main
.\.venv\Scripts\python.exe -m cashflow_integrated.audit_data
```

`main` использует явно синтетический прогноз. Рабочий вход пакета — функции ниже.
Старый запуск `python cashflow_integrated/main.py` также поддерживается.
Корневой `main.py` — исходная заглушка, он не запускает pipeline.

## Загрузка и подготовка до прогнозирования

```python
from cashflow_integrated import load_transactions, prepare_forecast_inputs

transactions = load_transactions("all_operations.csv", company_id="C002")
payload = prepare_forecast_inputs(
    transactions, start_date="2026-10-01", horizon_days=30,
)
expected_events = payload["events"]
```

Дата в примере следует за концом предоставленной истории, это не текущая дата.
При отсутствии `start_date` берётся день после последней операции.
История должна полностью предшествовать началу прогноза. При backtesting сначала
отделите прошлое; передача будущих операций вызывает `ValueError`.

Загрузчик читает CSV UTF-8/BOM с `;`. Идентификаторы и ИНН остаются строками.
Для общего CSV требуется `company_id`. Невалидные обязательные поля и повторные
`operation_id` вызывают ошибку; одинаковые платежи с разными ID сохраняются.
Отрицательная сумма не меняет направление автоматически: `amount > 0`, знак
задаёт `direction`. Необязательные ИНН и назначение могут отсутствовать.
DataFrame и список словарей также принимаются напрямую; алиасы ТЗ `date`,
`inn`, `counterparty`, `description`, `category`, `income/expense` поддерживаются.
XLSX и категоризация относятся к вышестоящему модулю загрузки/классификации.

## Контракт события

Каждый элемент `events` содержит:

| Поле | Значение |
|---|---|
| `event_id` | Идентификатор внутри текущего payload, не глобальный ключ дедупликации |
| `company_id` | Компания, либо `None`, если история не содержит этого поля |
| `date` | Дата ожидаемого события, ISO `YYYY-MM-DD` |
| `amount` | Неотрицательная сумма, округлённая до копеек |
| `signed_amount` | Положительное поступление / отрицательное списание |
| `direction` | `Поступление` / `Списание` |
| `counterparty`, `category` | Значения детектора; категория отсутствует в исходных CSV |
| `frequency`, `interval_days` | Частота и типичный интервал |
| `confidence` | Эвристическая оценка регулярности, не вероятность оплаты |
| `occurrences_count` | Число различных дат исторических операций в группе |
| `source` | `recurring_detector` |

Передаётся только **следующее событие каждого шаблона**, если оно попадает в окно.
Недельный шаблон не разворачивается автоматически в четыре платежа. Старые даты
не переносятся вперёд, исключения перечислены в `warnings`. Для полного календаря
на 30 дней forecasting должен отдельно согласовать правила разворачивания.

## Вызов реальной модели и оценка риска

Ниже функция подключения; `predict` — переданная вызывающей стороной реальная
функция forecasting с указанными параметрами, а не реализация в этом репозитории.

```python
from cashflow_integrated import run_cashflow_pipeline

def analyze_with_predictor(transactions, predict):
    def builder(*, transactions, recurring_events, base_forecast,
                start_date, horizon_days):
        return predict(
            transactions=transactions,
            expected_events=recurring_events,
            start_date=start_date,
            horizon_days=horizon_days,
        )

    return run_cashflow_pipeline(
        transactions,
        forecast_builder=builder,
        horizon_days=30,
    )
```

`predict` должен вернуть по одной строке на каждый день окна, например
`{"forecast": [{"date": "2026-10-01", "balance": 100000}, ...]}`.
Поддерживаются также DataFrame/список строк и алиасы остатка из `risk_engine`.
Для относительной оценки дефицита передайте одинаковый `current_balance` в строках.
Необходимо передавать уже рассчитанные остатки, а не только `income/expense`.

Forecasting отвечает за включение событий в модель потоков и **однократный учёт**.
Если baseline обучен на всей истории, регулярная часть уже может входить в него:
нельзя слепо добавить ожидаемые события поверх того же компонента. Нужно моделировать
остаточный нерегулярный поток либо заменить регулярную часть baseline.
Интеграция не пересчитывает остаток и не прибавляет события к готовому прогнозу.

Можно передать готовый `forecast` без builder. Тогда анализ выполнится с предупреждением,
что интеграционный слой не встраивал события. Сам факт вызова builder также не доказывает,
что внешний код использовал события: это проверяется интеграционными тестами forecasting.

## Проверки результата

- Горизонт по умолчанию всегда 30 дней. Короткий прогноз не сокращает его автоматически.
- Явный `company_id` в строках или обёртке прогноза должен совпадать с историей.
- Даты вне выбранного окна отклоняются. Разные компании не объединяются.
- `forecast_coverage` сообщает `complete`, число валидных дней и отсутствующие даты,
  включая начало/конец окна. Неполный положительный прогноз получает `incomplete`.
- При отрицательных остатках риск сохраняется, даже если покрытие неполное:
  проверяйте `forecast_coverage.complete` и `warnings` вместе со `status`.
- Повторные даты: `unknown`; пустой/непригодный прогноз: `unknown`.
- `probability=None`, `probability_status="not_estimated"`. Вероятностной модели нет.

Возвращаются `recurring_events`, `forecast_inputs`, `forecast_window`,
`forecast_coverage`, `cash_gap` и удобные верхнеуровневые поля риска.
Копии входов защищают данные вызывающей стороны от изменений внутри builder.

Пример реального payload: [forecast_inputs_C002.json](../artifacts/forecast_inputs_C002.json).
Он полностью сериализуется в JSON; это гипотезы по истории, не гарантированные обязательства.
