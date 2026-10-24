CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#17211e; --muted:#75817c; --line:#e8eeeb; --green:#187a5b; --green-soft:#eaf5ef; --canvas:#f7f9f8; --amber:#a86a12; --red:#b34b4b; }
html, body, [class*="css"] { font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:var(--ink); }
.stApp { background:var(--canvas); }
[data-testid="stHeader"] { background:transparent; height:2.5rem; }
[data-testid="stToolbar"] { right:1rem; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding:1.1rem .95rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom:.25rem; }
[data-testid="stMain"] .block-container { padding:2.2rem 3rem 3.5rem; max-width:1500px; }
.brand { display:flex; align-items:center; gap:11px; padding:4px 7px 24px; }
.brand-mark { width:36px;height:36px;border-radius:12px;background:#187a5b;color:#fff;display:grid;place-items:center;font-weight:800;font-family:Manrope;font-size:18px;box-shadow:0 5px 12px #187a5b24; }
.brand-name { font:800 16px Manrope,sans-serif; letter-spacing:-.5px; color:#17211e; }
.brand-caption { color:#9aa49f;font-size:10px;margin-top:1px;letter-spacing:.3px; }
.nav-label { color:#a0aaa5;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;padding:7px 9px 8px; }
.profile { border-top:1px solid var(--line); margin-top:24px;padding:18px 8px 4px;display:flex;gap:10px;align-items:center; }
.avatar { width:34px;height:34px;border-radius:50%;background:#e6f2ed;color:#187a5b;font-weight:700;display:grid;place-items:center; }
.profile-name { font-size:12px;font-weight:700; }.profile-company { font-size:10px;color:#8a9690;margin-top:2px; }
.eyebrow { color:#19805d;font-weight:700;font-size:11px;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:8px; }
.page-title { font:800 31px Manrope,sans-serif; letter-spacing:-1.2px; margin:0; color:#18231f; }
.page-subtitle { font-size:13px;color:#7b8781;margin-top:7px; }
.topline { display:flex;align-items:center;gap:8px;color:#75817c;font-size:11px; }
.demo-pill { display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:99px;background:#eef6f1;color:#39785e;font-size:10px;font-weight:700;letter-spacing:.3px; }
.demo-dot { width:6px;height:6px;border-radius:50%;background:#36a273; }
.section-title { font:700 16px Manrope,sans-serif; letter-spacing:-.3px; color:#1c2823; }
.section-note { color:#8b9691;font-size:11px;margin-top:4px; }
.card { background:#fff;border:1px solid var(--line);border-radius:16px;padding:19px 20px;box-shadow:0 2px 8px #18352b05; }
.kpi-label { color:#7d8983;font-size:11px;font-weight:600;display:flex;align-items:center;gap:7px; }
.kpi-icon { width:28px;height:28px;border-radius:9px;background:#edf6f1;color:#21815e;display:grid;place-items:center;font-size:14px; }
.kpi-value { font:800 25px Manrope,sans-serif;letter-spacing:-1px;color:#18231f;margin-top:15px;white-space:nowrap; }
.kpi-foot { color:#8b9691;font-size:10px;margin-top:6px; }
.positive { color:#20815d!important; }.negative { color:#a86a12!important; }
.risk-card { background:#fff;border:1px solid var(--line);border-left:3px solid #d7a74c;border-radius:13px;padding:15px 17px; }
.risk-card.critical { border-left-color:#ca6868; }.risk-card.safe { border-left-color:#4caa7b; }
.risk-title { font-size:12px;font-weight:700;color:#27332d; }.risk-detail { color:#84908a;font-size:10px;line-height:1.5;margin-top:5px; }
.risk-date { color:#89958f;font-size:10px; }.risk-amount { font:700 14px Manrope;color:#a86a12; }
.small-label { color:#89958f;font-size:10px; }.small-value { font:700 17px Manrope;color:#1d2a24;margin-top:6px; }
.upload-box { background:#fff;border:1px dashed #b9ccc2;border-radius:18px;padding:42px 24px;text-align:center; }
.upload-icon { width:54px;height:54px;margin:0 auto 14px;border-radius:17px;background:#eaf5ef;display:grid;place-items:center;color:#187a5b;font-size:25px; }
.empty { background:#fff;border:1px solid var(--line);border-radius:16px;padding:36px;text-align:center;color:#7d8983; }
div[data-testid="stMetric"] { background:#fff;border:1px solid var(--line);padding:17px 18px;border-radius:15px; }
div[data-testid="stMetricLabel"] { color:#77847d;font-size:11px; }
div[data-testid="stMetricValue"] { color:#18231f;font:800 24px Manrope; }
div[data-testid="stMetricDelta"] { font-size:10px; }
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap:4px; }
div[data-testid="stRadio"] div[role="radiogroup"] label { border-radius:9px;padding:6px 10px;color:#66736c;font-size:11px; }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:#eaf5ef;color:#187a5b;font-weight:700; }
div[data-testid="stButton"] button { border-radius:10px;font-weight:600;font-size:12px;min-height:38px; }
div[data-testid="stButton"] button[kind="primary"] { background:#187a5b;border-color:#187a5b; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#126b4f;border-color:#126b4f; }
div[data-testid="stSelectbox"] [data-baseweb="select"] > div, div[data-testid="stTextInput"] input { border-radius:10px; }
div[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:12px;overflow:hidden; }
hr { border-color:var(--line); }
@media(max-width:1100px) {
 [data-testid="stMain"] .block-container { padding:1.4rem 1rem 2rem; }
 [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:12px; }
 [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:1 1 45% !important; min-width:min(100%, 270px) !important; }
 .page-title {font-size:25px;overflow-wrap:anywhere;}
 .kpi-value {font-size:21px;}
 div[data-testid="stRadio"] div[role="radiogroup"] { flex-wrap:wrap; }
}
</style>
"""
