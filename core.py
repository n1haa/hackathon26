"""Auditable rules: (category, direction, regex, heuristic confidence)."""
RULES = (
    ('salary', 'outgoing', r'\b(заработн\w* плат\w*|зарплат\w*|salary|payroll)\b', .96),
    ('rent', 'outgoing', r'\b(аренд\w*|rent)\b', .94),
    ('taxes', 'outgoing', r'\b(налог\w*|ндфл|усн|енп|tax)\b', .94),
    ('bank_fees', 'outgoing', r'\b(комисси\w* банк\w*|банковск\w* комисси\w*|рко|bank fee)\b', .96),
    ('purchases', 'outgoing', r'\b(закуп\w*|поставк\w* товар\w*|оплата за товар\w*|сырь\w*|purchase)\b', .90),
    ('revenue', 'incoming', r'\b(выручк\w*|оплата за товар\w*|оплата за услуг\w*|эквайринг\w*|revenue)\b', .90),
    ('other_income', 'incoming', r'\b(возврат займ\w*|процент\w* по депозит\w*)\b', .88),
    ('other_expense', 'outgoing', r'\b(благотворительн\w* взнос\w*|пожертвован\w*)\b', .90),
)
INTERNAL = r'\b(между (своими|собственными) счетами|перевод собственных средств|перевод на свой счет)\b'
