"""Statement upload workflow shared by the preserved Streamlit pages."""
import hashlib

import pandas as pd
import streamlit as st

from cashflow_app import analyze_statement, read_statement


def render_upload(header, section, clear_upload):
    header("Загрузка данных", "Загрузите выписку и рассчитайте прогноз и кассовые риски.",
           demo=not st.session_state.get("has_upload", False))
    uploaded = st.file_uploader("Банковская выписка CSV / XLSX, до 20 МБ", type=["csv", "xlsx"],
                               key=f"statement_upload_{st.session_state.get('upload_widget_version', 0)}")
    st.caption("Обязательные поля: operation_date (или date), amount, direction. "
               "Сумма положительная; направление — Поступление / Списание или incoming / outgoing. "
               "company_id нужен для выбора компании. ИНН и идентификаторы в XLSX храните как текст.")
    if uploaded is not None:
        contents = uploaded.getvalue()
        digest = hashlib.sha256(contents).hexdigest()
        if digest != st.session_state.get("upload_digest"):
            for key in ("analysis_input", "analysis_results", "upload_frame", "upload_error"):
                st.session_state.pop(key, None)
            st.session_state.has_upload = True
            st.session_state.upload_digest = digest
            st.session_state.upload_name = uploaded.name
            try:
                st.session_state.upload_frame = read_statement(contents, filename=uploaded.name)
            except (ValueError, OSError, UnicodeError) as exc:
                st.session_state.upload_error = str(exc)
    if st.session_state.get("upload_error"):
        st.error(st.session_state.upload_error)
    frame = st.session_state.get("upload_frame")
    if frame is not None:
        section("Предпросмотр", "Первые 100 строк исходного файла")
        st.dataframe(frame.head(100), hide_index=True, width="stretch")
        company = None
        if "company_id" in frame:
            companies = sorted(frame.company_id.dropna().astype(str).str.strip().unique())
            if not companies:
                st.error("В файле нет company_id")
                return
            company = st.selectbox("Компания для анализа", companies)
        selected = frame if company is None else frame[frame.company_id.astype(str).str.strip().eq(company)]
        date_col = next((c for c in ("operation_date", "date", "transaction_date") if c in selected), None)
        balance_col = next((c for c in ("balance_eod", "balance") if c in selected), None)
        default_balance = 0.0
        if balance_col and date_col:
            ordered = selected.assign(_date=pd.to_datetime(selected[date_col], errors="coerce")).sort_values("_date")
            number = pd.to_numeric(ordered[balance_col].iloc[-1], errors="coerce")
            if pd.notna(number):
                default_balance = float(number)
        if date_col:
            dates = pd.to_datetime(selected[date_col], errors="coerce")
            st.caption(f"Период выбранной истории: {dates.min()} — {dates.max()}. Прогноз начинается на следующий день.")
        form_key = f"analysis_{st.session_state.get('upload_digest', '')}_{company}"
        with st.form(form_key):
            balance = st.number_input("Остаток на конец последнего дня выписки, ₽", value=default_balance, step=1000.0, format="%.2f")
            own_inn = st.text_input("ИНН вашей компании (необязательно)")
            complete = st.checkbox("Подтверждаю: история полная между первой и последней датой; в пропущенные дни операций не было")
            submitted = st.form_submit_button("Рассчитать прогноз и риски", type="primary")
        if submitted:
            # Invalidate old analysis before validation, so a failed upload never shows stale results.
            st.session_state.pop("analysis_input", None)
            st.session_state.pop("analysis_results", None)
            options = {"company_id": company, "current_balance": balance,
                       "complete_history": complete, "own_inn": own_inn or None}
            try:
                with st.spinner("Классификация, прогнозирование и оценка риска…"):
                    result = analyze_statement(frame, **options, horizon=30)
                st.session_state.analysis_input = {"frame": frame, "options": options}
                st.session_state.analysis_results = {30: result}
            except ValueError as exc:
                st.error(f"Анализ не выполнен: {exc}")
        if "analysis_input" in st.session_state:
            result = st.session_state.analysis_results[30]
            st.success(f"Обработано {len(result['transactions'])} операций компании {result['company_id'] or 'из выписки'}. "
                       "Результаты доступны во всех разделах.")
    if st.session_state.get("has_upload"):
        st.button("Удалить выбранный файл", on_click=clear_upload)
