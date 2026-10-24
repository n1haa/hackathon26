from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cashflow import data_access
from cashflow.styles import CSS
from cashflow.upload import render_upload as render_statement_upload

st.set_page_config(page_title="CashFlow AI · Финансовый обзор", page_icon="◈", layout="wide", initial_sidebar_state="auto")
st.markdown(CSS, unsafe_allow_html=True)

NAV_ITEMS = ["Обзор", "Загрузка данных", "Операции", "Регулярные платежи", "Финансовый прогноз", "Финансовые риски"]
NAV_ICONS = ["◫", "↑", "↔", "⟳", "⌁", "◇"]


def navigate(page: str):
    st.session_state.page = page


def clear_upload():
    for key in ["has_upload", "upload_name", "upload_preview", "upload_error", "upload_digest", "upload_frame", "analysis_input", "analysis_results"]:
        st.session_state.pop(key, None)
    st.session_state.upload_widget_version = st.session_state.get("upload_widget_version", 0) + 1


def rub(value: int | float) -> str:
    return f"{value:,.0f} ₽".replace(",", " ")


def safe_get(getter, label: str):
    """Keep the page usable when a future data provider is unavailable."""
    try:
        return getter()
    except Exception as exc:
        st.error(f"Не удалось загрузить {label}: {exc}")
        return None


def insufficient_data(message: str = "Пока недостаточно данных для этого раздела."):
    st.markdown(f'<div class="empty"><div style="font-size:27px;color:#a7b3ad;margin-bottom:10px">◌</div><div style="font:700 15px Manrope;color:#29352f">Данные пока недоступны</div><div style="font-size:11px;margin-top:7px">{message}</div></div>', unsafe_allow_html=True)


def transaction_table(df: pd.DataFrame, include_balance: bool = True) -> pd.DataFrame:
    shown = df.copy()
    shown = shown.rename(columns={"Операция": "Назначение платежа", "Тип": "Направление"})
    shown["Сумма"] = shown["Сумма"].map(lambda value: ("+" if value > 0 else "") + rub(value))
    if include_balance and "Остаток на конец дня" not in shown.columns:
        shown["Остаток на конец дня"] = "—"
    return shown


def styled_transactions(df: pd.DataFrame):
    return df.style.map(
        lambda value: "color:#20815d;font-weight:700" if str(value).startswith("+") else "color:#a86a12;font-weight:700" if str(value).startswith("-") else "",
        subset=["Сумма"],
    )


def header(title: str, subtitle: str, demo: bool = True):
    left, right = st.columns([4, 1.25], vertical_alignment="center")
    with left:
        st.markdown(f'<div class="eyebrow">CASHFLOW AI&nbsp;&nbsp; / &nbsp;&nbsp; ФИНАНСЫ</div><h1 class="page-title">{title}</h1><div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with right:
        if demo:
            st.markdown('<div style="text-align:right;padding-top:19px"><span class="demo-pill"><span class="demo-dot"></span>ДЕМО-ДАННЫЕ</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:right;padding-top:19px"><span class="demo-pill" style="background:#f1f3f2;color:#748079"><span class="demo-dot" style="background:#9ca9a2"></span>ВАШИ ДАННЫЕ</span></div>', unsafe_allow_html=True)
    st.write("")


def section(title: str, note: str | None = None):
    st.markdown(f'<div class="section-title">{title}</div>{f"<div class=section-note>{note}</div>" if note else ""}', unsafe_allow_html=True)


def make_chart(days: int, forecast_df: pd.DataFrame | None, risks: list[dict] | None = None):
    df = forecast_df
    if df is None or df.empty or not {"date", "balance", "kind"}.issubset(df.columns):
        return None
    hist = df[df.kind == "Факт"]
    future = df[df.kind != "Факт"]
    fig = go.Figure()
    if len(hist):
        fig.add_trace(go.Scatter(x=hist.date, y=hist.balance, mode="lines", name="Фактический остаток", line=dict(color="#19805d", width=2.5), hovertemplate="%{x|%d %b}<br><b>%{y:,.0f} ₽</b><extra>Факт</extra>"))
    if len(hist) and len(future):
        bridge = pd.concat([hist.tail(1), future.head(1)])
        fig.add_trace(go.Scatter(x=bridge.date, y=bridge.balance, mode="lines", showlegend=False, line=dict(color="#a9c8b8", width=2, dash="dot"), hoverinfo="skip"))
    if len(future):
        trace_name = "Демо-прогноз" if future.kind.astype(str).str.contains("Демо", case=False).any() else "Прогноз"
        fig.add_trace(go.Scatter(x=future.date, y=future.balance, mode="lines", name=trace_name, line=dict(color="#91b8a4", width=2.5, dash="dash"), fill="tozeroy", fillcolor="rgba(145,184,164,.10)", hovertemplate="%{x|%d %b}<br><b>%{y:,.0f} ₽</b><extra>%{fullData.name}</extra>"))
        reserve_levels = [risk.get("reserve_level") for risk in (risks or []) if risk.get("reserve_level") is not None]
        if reserve_levels:
            reserve = reserve_levels[0]
            fig.add_hline(y=reserve, line_width=1, line_dash="dot", line_color="#d9a451", annotation_text="Резерв", annotation_position="bottom right", annotation_font_size=9, annotation_font_color="#a86a12")
            below = future[future.balance < reserve]
            if not below.empty:
                fig.add_trace(go.Scatter(x=below.date, y=below.balance, mode="markers", name="Ниже резерва", marker=dict(color="#c86d63", size=7), hovertemplate="%{x|%d %b}<br><b>%{y:,.0f} ₽</b><extra>Ниже резерва</extra>"))
    if len(hist):
        fig.add_vline(x=hist.date.iloc[-1], line_width=1, line_dash="dot", line_color="#c7d1cc")
        fig.add_annotation(x=hist.date.iloc[-1], y=1.03, yref="paper", text="КОНЕЦ ИСТОРИИ", showarrow=False, font=dict(size=9, color="#84928b"), xanchor="right")
    fig.update_layout(height=305, margin=dict(l=3, r=8, t=30, b=4), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#829087", size=10), legend=dict(orientation="h", y=1.15, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"), hovermode="x unified")
    fig.update_xaxes(showgrid=False, tickformat="%d %b", linecolor="#edf1ee", ticks="outside", ticklen=5)
    fig.update_yaxes(showgrid=True, gridcolor="#eff3f0", zeroline=False, tickprefix="₽ ", tickformat="~s", ticks="outside", ticklen=4)
    return fig


def category_flow_chart(transactions: pd.DataFrame):
    rows = transactions.assign(
        Поступления=transactions["Сумма"].clip(lower=0),
        Списания=-transactions["Сумма"].clip(upper=0),
    ).groupby("Категория", as_index=False)[["Поступления", "Списания"]].sum()
    fig = go.Figure()
    fig.add_bar(y=rows["Категория"], x=rows["Поступления"], name="Поступления", orientation="h", marker_color="#26815f", hovertemplate="%{y}<br>%{x:,.0f} ₽<extra>Поступления</extra>")
    fig.add_bar(y=rows["Категория"], x=rows["Списания"], name="Списания", orientation="h", marker_color="#d3a151", hovertemplate="%{y}<br>%{x:,.0f} ₽<extra>Списания</extra>")
    fig.update_layout(barmode="group", height=max(250, 34 * len(rows)), margin=dict(l=5, r=10, t=15, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#829087", size=10), legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)), xaxis=dict(showgrid=True, gridcolor="#eff3f0", tickprefix="₽ ", tickformat="~s"), yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def render_dashboard():
    if st.session_state.get("has_upload") and not data_access.is_ready():
        header("Финансовый обзор", "Выписка ожидает успешного анализа.", demo=False)
        insufficient_data("Укажите параметры и выполните расчёт на странице загрузки.")
        st.info("Проверьте компанию, остаток и полноту истории в разделе «Загрузка данных».", icon="ℹ️")
        st.button("Открыть загруженную выписку", key="open_uploaded", on_click=navigate, args=("Загрузка данных",))
        return

    header("Финансовый обзор", "Контролируйте денежные потоки и прогнозируйте финансовое состояние бизнеса.", demo=not data_access.is_ready())
    col_period, col_cta = st.columns([4, 1.35], vertical_alignment="center")
    with col_period:
        st.caption("Период выбранной выписки" if data_access.is_ready() else "Демонстрационный пример: октябрь 2026")
    with col_cta:
        st.button("＋  Загрузить выписку", type="primary", width="stretch", key="dash_upload", on_click=navigate, args=("Загрузка данных",))
    st.write("")
    summary = safe_get(data_access.get_financial_summary, "финансовые показатели")
    if summary is None:
        insufficient_data("Показатели временно не загрузились. Попробуйте обновить страницу.")
        return
    forecast_df = safe_get(lambda: data_access.get_forecast(30), "прогноз")
    risks = safe_get(data_access.get_risks, "предупреждения")
    transactions = safe_get(data_access.get_transactions, "операции")
    recurring = safe_get(data_access.get_recurring_payments, "регулярные платежи")
    has_forecast_fields = forecast_df is not None and {"kind", "balance"}.issubset(forecast_df.columns)
    future_values = forecast_df.loc[forecast_df.kind != "Факт", "balance"] if has_forecast_fields and not forecast_df.empty else pd.Series(dtype=float)
    minimum_balance = int(future_values.min()) if not future_values.empty else None
    forecast_balance = summary.get("forecast_balance")
    labels = [
        ("Текущий остаток", summary.get("balance"), "Доступные денежные средства", "◉", None),
        ("Остаток через 30 дней", forecast_balance, "Значение из прогноза", "⌁", None),
        ("Минимальный остаток", minimum_balance, "За 30 дней · из прогноза", "↓", None),
        ("Поступления", summary.get("income"), "За текущий период", "↙", None),
        ("Расходы", summary.get("expenses"), "За текущий период", "↗", None),
    ]
    kcols = st.columns(5)
    for col, (label, amount, foot, icon, delta) in zip(kcols, labels):
        with col:
            value = rub(amount) if amount is not None and pd.notna(amount) else "—"
            st.markdown(f'<div class="card"><div class="kpi-label"><span class="kpi-icon">{icon}</span>{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)
    st.write("")
    chart_col, side_col = st.columns([2.2, 1], gap="large")
    with chart_col:
        top_a, top_b = st.columns([2, 1], vertical_alignment="center")
        with top_a:
            section("Движение денежных средств", "Фактический остаток и ориентир на будущее")
        with top_b:
            period = st.radio("Период графика", ["7 дней", "30 дней", "90 дней"], horizontal=True, index=1, label_visibility="collapsed", key="dashboard_period")
        period_days = {"7 дней": 7, "30 дней": 30, "90 дней": 90}[period]
        st.markdown('<div class="card" style="padding:15px 16px 8px;margin-top:13px">', unsafe_allow_html=True)
        chart_data = safe_get(lambda: data_access.get_forecast(period_days), "прогноз")
        chart = make_chart(period_days, forecast_df=chart_data, risks=risks or [])
        if chart is None:
            insufficient_data("Модуль пока не передал временной ряд остатков.")
        else:
            st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with side_col:
        section("Финансовые риски", "События на горизонте 30 дней")
        st.write("")
        if risks is None:
            st.warning("Предупреждения не удалось загрузить.")
        elif not risks:
            st.markdown('<div class="risk-card safe"><div class="risk-title">Кассовый разрыв не обнаружен</div><div class="risk-detail">Модуль риска не передал предупреждений для выбранного периода.</div></div>', unsafe_allow_html=True)
        else:
            for risk in risks[:2]:
                st.markdown(f'<div class="risk-card" style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;gap:8px"><span class="risk-title">{escape(str(risk.get("title", "Предупреждение")))}</span><span class="risk-date">{escape(str(risk.get("date", "Дата не предоставлена")))}</span></div><div class="risk-amount" style="margin-top:9px">{escape(str(risk.get("amount", "Сумма не предоставлена")))}</div><div class="risk-detail">{escape(str(risk.get("detail", "Подробности не предоставлены.")))}</div></div>', unsafe_allow_html=True)
        st.button("Все предупреждения →", key="all_risks", on_click=navigate, args=("Финансовые риски",))
    st.write("")
    recent_col, spend_col = st.columns([1.65, 1], gap="large")
    with recent_col:
        section("Последние операции", "Движение по счёту за последние дни")
        if transactions is None or transactions.empty:
            insufficient_data("Список операций пока пуст.")
        else:
            shown = transaction_table(transactions.head(5), include_balance=False)
            st.dataframe(styled_transactions(shown[["Дата", "Назначение платежа", "Категория", "Сумма"]]), hide_index=True, width="stretch", height=235, column_config={"Дата": st.column_config.DateColumn(format="DD MMM")})
        st.button("Все операции →", key="all_operations", on_click=navigate, args=("Операции",))
    with spend_col:
        section("Структура расходов", "По загруженным операциям")
        if transactions is None or transactions.empty:
            insufficient_data("Категории расходов пока не переданы.")
        else:
            expense_rows = transactions.loc[transactions["Сумма"] < 0].assign(Расходы=lambda rows: -rows["Сумма"]).groupby("Категория", as_index=False)["Расходы"].sum()
            if expense_rows.empty:
                insufficient_data("В текущей выборке нет расходов.")
            else:
                palette = ["#187a5b", "#66a789", "#91b8a4", "#b4cfbf", "#d4e4db", "#e8efeb"]
                colors = [palette[index % len(palette)] for index in range(len(expense_rows))]
                fig = go.Figure(go.Pie(labels=expense_rows["Категория"], values=expense_rows["Расходы"], hole=.72, sort=False, marker=dict(colors=colors, line=dict(color="#fff", width=3)), textinfo="none", hovertemplate="%{label}<br>%{value:,.0f} ₽ (%{percent})<extra></extra>"))
                fig.update_layout(height=215, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=True, legend=dict(orientation="h", y=-.08, x=.02, font=dict(size=9, color="#75817c")))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if recurring is not None and not recurring.empty:
            st.markdown(f'<div class="small-label">Регулярные платежи &nbsp; · &nbsp; {len(recurring)} в источнике</div><div class="small-value">{rub(recurring["Сумма"].sum())} <span style="font:400 10px DM Sans;color:#8b9691">сумма ближайших операций</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-label">Регулярные платежи</div><div class="small-value">Не предоставлены</div>', unsafe_allow_html=True)


def render_upload():
    render_statement_upload(header, section, clear_upload)


def render_transactions():
    demo = not st.session_state.get("has_upload")
    header("Операции", "Просматривайте движения по счёту и быстро находите нужные платежи.", demo)
    if not demo and not data_access.is_ready():
        insufficient_data("Выполните анализ на странице «Загрузка данных»; при ошибке проверьте формат колонок.")
        return
    df = safe_get(data_access.get_transactions, "операции")
    if df is None:
        insufficient_data("Операции появятся здесь, когда их передаст источник данных.")
        return
    a, b, c = st.columns([2.1, 1, 1])
    query = a.text_input("Поиск", placeholder="Поиск по операции или контрагенту", label_visibility="collapsed", key="tx_search")
    category = b.selectbox("Категория", ["Все категории"] + sorted(df["Категория"].unique()), label_visibility="collapsed", key="tx_category")
    kind = c.selectbox("Тип операции", ["Все операции", "Поступление", "Списание"], label_visibility="collapsed", key="tx_kind")
    if query:
        df = df[df.apply(lambda row: query.lower() in " ".join(map(str, row.values)).lower(), axis=1)]
    if category != "Все категории":
        df = df[df["Категория"] == category]
    if kind != "Все операции":
        df = df[df["Тип"] == kind]
    if df.empty:
        st.info("По выбранным фильтрам операций не найдено.")
        return
    inflow = int(df.loc[df["Сумма"] > 0, "Сумма"].sum())
    outflow = int(-df.loc[df["Сумма"] < 0, "Сумма"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Операций", len(df))
    m2.metric("Поступления", rub(inflow))
    m3.metric("Списания", rub(outflow))
    st.write("")
    section("Поступления и списания по категориям", "Суммы в текущей выборке")
    st.plotly_chart(category_flow_chart(df), width="stretch", config={"displayModeBar": False})
    section("История операций", f"{len(df)} операций в выборке")
    if df.empty:
        st.info("По выбранным фильтрам операций не найдено.")
    else:
        shown = transaction_table(df)
        column_order = ["Дата", "Сумма", "Направление", "Контрагент", "Назначение платежа", "Категория", "Остаток на конец дня", "Статус"]
        st.dataframe(styled_transactions(shown[column_order]), hide_index=True, width="stretch", height=500, column_config={"Дата": st.column_config.DateColumn(format="DD MMM YYYY"), "Остаток на конец дня": st.column_config.TextColumn(help="Не предоставляется текущим источником данных.")})
        st.caption("Остаток на конец дня отображается только при наличии balance_eod в выписке.")


def render_recurring():
    demo = not st.session_state.get("has_upload")
    header("Регулярные платежи", "Планируйте повторяющиеся расходы и держите обязательства под контролем.", demo)
    if not demo and not data_access.is_ready():
        insufficient_data("Выполните анализ на странице «Загрузка данных» для поиска регулярных платежей.")
        return
    df = safe_get(data_access.get_recurring_payments, "регулярные платежи")
    if df is None or df.empty:
        insufficient_data("Регулярные платежи пока не переданы источником данных.")
        return
    a, b = st.columns([1, 1])
    a.metric("Обнаруженных шаблонов", len(df))
    b.metric("Сумма ближайших операций", rub(df["Сумма"].sum()))
    st.write("")
    section("Регулярные операции", "Ближайшая дата каждого шаблона; просроченные даты не переносятся")
    st.write("")
    shown = df.copy()
    shown["Сумма"] = shown["Сумма"].map(rub)
    st.dataframe(shown, hide_index=True, width="stretch", height=360)
    st.caption("Оценка регулярности — эвристика, не вероятность оплаты. В таблице могут быть просроченные ожидания.")


def render_forecast():
    demo = not st.session_state.get("has_upload")
    header("Финансовый прогноз", "Наглядный ориентир для планирования денежных средств на ближайший период.", demo)
    if not demo and not data_access.is_ready():
        insufficient_data("Выполните анализ на странице «Загрузка данных» для построения прогноза.")
        return
    left, right = st.columns([2.3, 1], vertical_alignment="center")
    with left:
        section("Остаток денежных средств", "Исторические значения и прогноз")
    with right:
        period = st.radio("Горизонт прогноза", ["7 дней", "30 дней", "90 дней"], horizontal=True, index=1, label_visibility="collapsed", key="forecast_period")
    days = {"7 дней": 7, "30 дней": 30, "90 дней": 90}[period]
    st.write("")
    summary = safe_get(data_access.get_financial_summary, "финансовые показатели")
    forecast = safe_get(lambda: data_access.get_forecast(days), "прогноз")
    if forecast is None or forecast.empty or not {"date", "balance", "kind"}.issubset(forecast.columns):
        insufficient_data("Для прогноза нужны дневные остатки из финансового модуля.")
        return
    future = forecast[forecast.kind != "Факт"]
    if future.empty:
        insufficient_data("Модуль не передал прогнозные значения.")
        return
    current_value = summary.get("balance") if summary else None
    ending = future.balance.iloc[-1]
    minimum = future.balance.min()
    a, b, c = st.columns(3)
    a.metric("Текущий остаток", rub(current_value) if current_value is not None else "—")
    b.metric("Минимальный остаток", rub(minimum))
    c.metric(f"Остаток через {days} дней", rub(ending))
    if not demo:
        st.dataframe(pd.DataFrame(data_access.result(days)["forecast"]["forecast"]), hide_index=True, width="stretch")
    st.write("")
    st.markdown('<div class="card" style="padding:15px 16px 8px">', unsafe_allow_html=True)
    risks = safe_get(data_access.get_risks, "предупреждения")
    chart = make_chart(days, forecast_df=forecast, risks=risks or [])
    if chart is not None:
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Синтетический демонстрационный сценарий." if demo else f"Модель: {data_access.result(days)['forecast']['model_name']}; горизонт {days} дней.")
    st.write("")
    section("Минимальный остаток", "Наименьшее значение в выбранном прогнозе")
    st.metric(f"Минимум за {days} дней", rub(minimum))


def render_risks():
    demo = not st.session_state.get("has_upload")
    header("Финансовые риски", "Кассовые разрывы в прогнозе на 30 дней.", demo)
    if not demo and not data_access.is_ready():
        insufficient_data("Выполните анализ на странице «Загрузка данных» для поиска кассовых разрывов.")
        return
    risks = safe_get(data_access.get_risks, "предупреждения")
    if risks is None:
        insufficient_data("Сервис рисков не ответил. Попробуйте обновить страницу.")
        return
    if not risks:
        st.markdown('<div class="risk-card safe"><div class="risk-title">Кассовых разрывов не обнаружено</div><div class="risk-detail">Источник данных завершил проверку и не передал предупреждений для выбранного периода.</div></div>', unsafe_allow_html=True)
        return
    gap_values = [risk.get("gap_amount") for risk in risks if risk.get("gap_amount") is not None]
    c1, c2, c3 = st.columns(3)
    c1.metric("Предупреждений", len(risks))
    c2.metric("Максимальный кассовый разрыв", rub(max(gap_values)) if gap_values else "Не предоставлен")
    c3.metric("Ближайшая дата риска", risks[0].get("date") or "Не предоставлена")
    st.write("")
    section("Потенциальные кассовые разрывы", "События из результата модуля риска")
    st.write("")
    for risk in risks:
        gap_amount = risk.get("gap_amount")
        amount_label = "Размер кассового разрыва" if gap_amount is not None else "Сумма ожидаемого списания"
        amount_value = rub(gap_amount) if gap_amount is not None else risk.get("amount", "Не предоставлена")
        probability = risk.get("probability")
        probability_label = escape(str(probability)) if probability is not None else "Не предоставлена модулем"
        date_label = escape(str(risk.get("date") or "Дата не предоставлена"))
        minimum_label = escape(str(risk.get("minimum") or "Не предоставлен"))
        risk_level = escape(str(risk.get("level") or "Уровень не предоставлен"))
        duration = escape(str(risk.get("duration") or "Длительность не предоставлена"))
        title = escape(str(risk.get("title", "Финансовое предупреждение")))
        detail = escape(str(risk.get("detail", "Подробности не предоставлены источником данных.")))
        amount_label = escape(amount_label)
        amount_value = escape(str(amount_value))
        st.markdown(f'<div class="risk-card" style="margin-bottom:12px;padding:20px 22px"><div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px"><div><div class="risk-date">{date_label} &nbsp; · &nbsp; {duration}</div><div class="risk-title" style="font-size:14px;margin-top:7px">{title}</div></div><span class="demo-pill" style="background:#fbf3e7;color:#a86a12">{risk_level}</span></div><div style="display:flex;gap:45px;flex-wrap:wrap;margin-top:17px"><div><div class="small-label">{amount_label}</div><div class="risk-amount" style="margin-top:5px">{amount_value}</div></div><div><div class="small-label">Минимальный остаток</div><div class="small-value" style="font-size:14px">{minimum_label}</div></div><div><div class="small-label">Вероятность</div><div class="small-value" style="font:600 12px DM Sans">{probability_label}</div></div></div><div class="risk-detail" style="margin-top:13px">{detail}</div></div>', unsafe_allow_html=True)
    st.caption("Модуль риска оценивает тяжесть кассового разрыва, а не его вероятность.")


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">C</div><div><div class="brand-name">cashflow<span style="color:#19805d">.ai</span></div><div class="brand-caption">FINANCIAL WORKSPACE</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Рабочее пространство</div>', unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = "Обзор"
    page = st.radio("Навигация", NAV_ITEMS, format_func=lambda x: f"{NAV_ICONS[NAV_ITEMS.index(x)]}    {x}", label_visibility="collapsed", key="page")
    st.markdown('<div class="nav-label" style="margin-top:22px">Компания</div>', unsafe_allow_html=True)
    if data_access.is_ready():
        st.write("Компания:", data_access.result()["company_id"] or "Из выписки")
    else:
        st.caption("Демонстрационная компания" if not st.session_state.get("has_upload") else "Выписка ожидает анализа")

if page == "Обзор":
    render_dashboard()
elif page == "Загрузка данных":
    render_upload()
elif page == "Операции":
    render_transactions()
elif page == "Регулярные платежи":
    render_recurring()
elif page == "Финансовый прогноз":
    render_forecast()
else:
    render_risks()

if data_access.is_ready():
    with st.expander("Качество данных и ограничения расчёта"):
        for warning in data_access.result()["warnings"]:
            if not warning.startswith("forecast_builder не передан:"):
                st.warning(warning)
