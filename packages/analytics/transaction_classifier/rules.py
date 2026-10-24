"""Auditable rules: (category, direction, regex, heuristic confidence)."""
RULES = (
    ('salary', 'outgoing', r'\b(заработн\w* плат\w*|зарплат\w*|salary|payroll)\b', .96),
    ('rent', 'outgoing', r'\b(аренд\w*|rent)\b', .94),
    ('taxes', 'outgoing', r'\b(налог\w*|ндфл|усн|енп|tax)\b', .94),
    ('bank_fees', 'outgoing', r'\b(комисси\w* банк\w*|банковск\w* комисси\w*|комисси\w* за (ведение|обслуживание) счета|рко|bank fee)\b', .96),
    ('purchases', 'outgoing', r'\b(закуп\w*|поставк\w* товар\w*|оплата за товар\w*|за (канцтовар\w*|хозтовар\w*|сырь\w*|запчаст\w*|продукци\w*|товар\w*)|сырь\w*|purchase)\b', .90),
    ('revenue', 'incoming', r'\b(выручк\w*|оплата за товар\w*|оплата за услуг\w*|эквайринг\w*|revenue)\b', .90),
    ('other_income', 'incoming', r'\b(возврат займ\w*|процент\w* по депозит\w*)\b', .88),
    ('other_expense', 'outgoing', r'\b(благотворительн\w* взнос\w*|пожертвован\w*)\b', .90),
    # Explicit sales receipts from the participant dataset; generic invoice,
    # contract and UPD references intentionally do not imply revenue.
    ('revenue', 'incoming', r'\bзачисление по операциям оплаты через сбп\b', .95),
    ('revenue', 'incoming', r'\b(оплат\w*|предоплат\w*).{0,180}\bза (товар\w*|продукци\w*|поставку товар\w*|перевозку груза|транспортно-экспедиционные услуги)\b', .90),
    ('other_expense', 'outgoing', r'\b(за бухгалтерские услуги|за услуги уборки|услуг связи|доступа в интернет|пополнение рекламного кабинета|услуг по договору гпх)\b', .90),
)
INTERNAL = r'\b(между (своими|собственными) счетами|перевод собственных средств|перевод на свой счет)\b'

# A transfer to an owner's personal account is not proof of a transfer between
# business accounts. Keep it unknown, even if the tax identifier matches.
PERSONAL_TRANSFER = r'\b(на личн\w* (карт\w*|счет\w*)|на счет физического лица|личн\w* нужд\w*)\b'
TAX_BOILERPLATE = r'\bбез налога\b(?:\s*\(ндс\))?|\bндфл удержан\b'

# Only a tax authority's explicit name is used as a counterparty rule. A bank,
# employer or supplier name alone does not identify the purpose of a payment.
COUNTERPARTY_RULES = (
    ('taxes', 'outgoing', r'^(фнс|уфнс|ифнс|мифнс)\b|\bфедеральная налоговая служба\b|^казначейство россии\s*\(фнс россии\)', .90),
)
