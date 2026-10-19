import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Smart Irrigation",
    page_icon="🌱",
    layout="wide",
)

MODEL_PATH = Path("best_irrigation_model.joblib")
META_PATH = Path("model_metadata.json")

# Store these values in Streamlit Secrets, not in public code.
CHANNEL_ID = str(st.secrets["THINGSPEAK_CHANNEL_ID"])
READ_API_KEY = st.secrets.get("THINGSPEAK_READ_API_KEY", "")

FIELD_MAP = {
    "Temperature": "field1",
    "Humidity": "field2",
    "Soil Moisture Percentage": "field4",
    "RawLDR": "field5",
}

INK = "#12332c"
MUTED = "#5d7a72"

# ----------------------------------------------------------------------------
# Styling: light, fresh, farm-inspired
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

:root{
  --ink:#12332c; --muted:#5d7a72; --line:#dfe9e3; --card:#ffffff;
  --deep:#0f3d36; --brand:#0f5b46;
  --soil:#16a34a; --temp:#ea580c; --hum:#2563eb; --light:#d99a06;
}
html, body, [class*="css"], .stApp{font-family:'DM Sans',sans-serif;}
.stApp{
  background:
    radial-gradient(800px 420px at 100% 0%, rgba(125,200,240,.28), transparent 65%),
    radial-gradient(700px 420px at 0% 100%, rgba(150,220,170,.30), transparent 65%),
    #f3f7f1;
  color:var(--ink);
}
.block-container{max-width:1180px; padding-top:1.6rem; padding-bottom:3rem;}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stWidgetLabel"] p{color:var(--ink); font-weight:500;}

/* ---------- hero with a little farm scene ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:26px; padding:34px 36px; min-height:190px;
  background:linear-gradient(120deg,#d9eefa 0%, #e6f6ee 60%, #d9f0df 100%);
  border:1px solid #d3e6db; box-shadow:0 10px 30px rgba(18,51,44,.07);
}
.hero-text{position:relative; z-index:2; max-width:52%;}
.hero h1{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:2.5rem;
  line-height:1.08; margin:0; color:var(--ink); letter-spacing:-.02em;
}
.hero p{margin:.6rem 0 0; color:var(--muted); font-size:1.02rem; line-height:1.55;}
.scene{position:absolute; right:0; bottom:0; width:62%; height:100%; z-index:1;}
.rays{transform-origin:500px 58px; animation:spin 40s linear infinite;}
.cloud1{animation:drift 14s ease-in-out infinite alternate;}
.cloud2{animation:drift 19s ease-in-out infinite alternate-reverse;}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes drift{from{transform:translateX(0)}to{transform:translateX(46px)}}
@media (max-width:760px){
  .hero{padding:24px 22px;} .hero-text{max-width:100%;} .hero h1{font-size:1.8rem}
  .scene{opacity:.35; width:100%}
}

.live{display:inline-flex; align-items:center; gap:8px; font-size:.85rem; color:var(--ink);
  border:1px solid var(--line); border-radius:999px; padding:6px 14px; background:#fff;}
.dot{width:9px; height:9px; border-radius:50%; background:#22c55e; position:relative;}
.dot::after{content:""; position:absolute; inset:0; border-radius:50%; background:#22c55e;
  animation:ping 1.8s ease-out infinite;}
.dot.stale{background:#ef4444} .dot.stale::after{background:#ef4444}
@keyframes ping{0%{transform:scale(1); opacity:.7}100%{transform:scale(3.2); opacity:0}}

/* ---------- side by side layout ---------- */
.top{display:grid; grid-template-columns:1.1fr 1fr; gap:18px; margin:14px 0 18px; align-items:stretch;}
.mid{display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-bottom:6px;}
@media (max-width:900px){ .top, .mid{grid-template-columns:1fr;} }

/* ---------- prediction card: the one dark, bold element ---------- */
.predict{
  position:relative; overflow:hidden; display:grid; grid-template-columns:150px 1fr; gap:22px;
  align-items:center; padding:26px 28px; border-radius:26px; color:#fff;
  background:
    radial-gradient(380px 250px at 14% 50%, color-mix(in srgb, var(--accent) 32%, transparent), transparent 70%),
    linear-gradient(145deg,#0f3d36,#0a2c2a);
  box-shadow:0 18px 40px rgba(15,61,54,.28);
}
.predict .rd{
  position:absolute; top:-14px; width:3px; height:18px; border-radius:3px; opacity:0;
  background:linear-gradient(transparent, var(--accent)); animation:fall linear infinite;
}
@keyframes fall{0%{transform:translateY(0); opacity:0}15%{opacity:.75}100%{transform:translateY(300px); opacity:0}}
.drop-wrap{position:relative; z-index:1; display:flex; justify-content:center;}
.drop{width:140px; height:186px; filter:drop-shadow(0 8px 22px color-mix(in srgb, var(--accent) 55%, transparent));}
.wave1{animation:slide 5s linear infinite;}
.wave2{animation:slide 8s linear infinite reverse;}
@keyframes slide{from{transform:translateX(0)}to{transform:translateX(-60px)}}
.pred-text{position:relative; z-index:1;}
.pred-label{color:#a9cfc6; font-size:.92rem; margin:0 0 4px;}
.pred-value{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3.4rem; line-height:1;
  letter-spacing:-.03em; margin:0; color:var(--accent);
}
.pred-advice{margin:.9rem 0 0; color:#dcefe9; font-size:.98rem; line-height:1.55; max-width:40ch;}
@media (max-width:560px){
  .predict{grid-template-columns:1fr; text-align:center; padding:22px 18px;}
  .pred-advice{margin-left:auto; margin-right:auto}
}

/* ---------- metric cards ---------- */
.metrics{display:grid; grid-template-columns:1fr 1fr; gap:14px;}
.card{
  background:var(--card); border:1px solid var(--line); border-radius:20px; padding:16px 18px;
  box-shadow:0 6px 18px rgba(18,51,44,.06);
}
.card .k{display:flex; align-items:center; gap:10px; color:var(--muted); font-size:.9rem; font-weight:500;}
.chip{width:32px; height:32px; border-radius:10px; display:grid; place-items:center; color:var(--c);
  background:color-mix(in srgb, var(--c) 13%, #fff);}
.ico{width:18px; height:18px;}
.card .v{font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:2.2rem; margin:.45rem 0 .6rem; color:var(--ink);}
.card .v small{font-size:1rem; font-weight:500; color:var(--muted); margin-left:3px;}
.bar{height:8px; border-radius:8px; background:#e8f0ea; overflow:hidden;}
.bar i{display:block; height:100%; width:var(--w); border-radius:8px; background:var(--c);
  animation:fillbar 1.2s cubic-bezier(.2,.8,.2,1) both;}
@keyframes fillbar{from{width:0}}

.soil{display:flex; align-items:center; gap:12px; margin-top:.35rem}
.gauge{width:78px; height:78px; flex:none; transform:rotate(-90deg);}
.gauge .track{stroke:#e4efe7}
.gauge .prog{stroke-dasharray:339.3; animation:gauge 1.6s cubic-bezier(.2,.8,.2,1) both; stroke-linecap:round;}
@keyframes gauge{from{stroke-dashoffset:339.3}}
.soil .v{margin:0}

/* ---------- panels ---------- */
.panel{background:var(--card); border:1px solid var(--line); border-radius:22px; padding:20px 24px;
  box-shadow:0 6px 18px rgba(18,51,44,.06);}
.sec{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.2rem; margin:0 0 .8rem; color:var(--ink);}
.prob{display:grid; grid-template-columns:84px 1fr 56px; align-items:center; gap:12px; margin:11px 0; font-size:.94rem; color:var(--ink);}
.prob .bar{height:11px}
.prob b{text-align:right; font-weight:600}
.info{color:var(--muted); font-size:.93rem; line-height:1.75;}
.info strong{color:var(--ink); font-weight:600;}
.notice{border-radius:14px; padding:12px 16px; margin:0 0 14px; font-size:.92rem; line-height:1.5;
  border:1px solid #f1d27a; background:#fff7dc; color:#7a5800;}
.chartsec{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.3rem; margin:1.6rem 0 .6rem; color:var(--ink);}

/* streamlit widgets */
.stButton>button{
  border-radius:12px; border:0; font-weight:600; padding:.55rem 1.1rem;
  background:var(--brand); transition:transform .15s, background .2s;
}
.stButton>button, .stButton>button p{color:#fff;}
.stButton>button:hover{background:#0b4a39; transform:translateY(-1px);}
.stButton>button:focus-visible{outline:2px solid var(--hum); outline-offset:2px;}
button[data-baseweb="tab"]{font-weight:600}
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#fff; border:1px solid var(--line); border-radius:22px; box-shadow:0 6px 18px rgba(18,51,44,.06);
}

@media (prefers-reduced-motion:reduce){
  *{animation:none !important; transition:none !important;}
}
</style>
"""


def html(block: str):
    """Render raw HTML. Leading spaces / blank lines are removed because
    Markdown would otherwise treat indented lines as code blocks."""
    clean = "\n".join(line.strip() for line in block.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model + data
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected = metadata["feature_columns_in_required_order"]
    if expected != list(FIELD_MAP.keys()):
        raise ValueError(f"Model feature order mismatch. Metadata expects: {expected}")
    return model, metadata


@st.cache_data(ttl=15, show_spinner=False)
def get_latest_reading():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json"
    params = {"api_key": READ_API_KEY} if READ_API_KEY else {}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    values = {}
    for feature, field in FIELD_MAP.items():
        if data.get(field) in (None, ""):
            raise ValueError(f"ThingSpeak has no value for {field}")
        values[feature] = float(data[field])
    return data, values


@st.cache_data(ttl=60, show_spinner=False)
def get_history(n: int = 100) -> pd.DataFrame:
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json"
    params = {"results": n}
    if READ_API_KEY:
        params["api_key"] = READ_API_KEY
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    feeds = r.json().get("feeds", [])
    if not feeds:
        return pd.DataFrame()
    df = pd.DataFrame(feeds)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    out = pd.DataFrame({"Time": df["created_at"]})
    for name, field in FIELD_MAP.items():
        if field in df:
            out[name] = pd.to_numeric(df[field], errors="coerce")
    return out.set_index("Time")


# ----------------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------------
def level_theme(label: str):
    l = label.lower()
    if "high" in l:
        return dict(color="#38bdf8", fill=0.86, drops=14,
                    advice="Dry conditions. Check the soil by hand, then irrigate soon.")
    if "med" in l:
        return dict(color="#4ade80", fill=0.55, drops=8,
                    advice="Moderate watering is suggested. A normal irrigation cycle should be enough.")
    if "low" in l:
        return dict(color="#fbbf24", fill=0.28, drops=4,
                    advice="Only a little water is needed right now. Light irrigation or wait.")
    return dict(color="#9ab8c8", fill=0.08, drops=0,
                advice="No irrigation needed at the moment. The soil is fine.")


SCENE = """
<svg class="scene" viewBox="0 0 600 200" preserveAspectRatio="xMaxYMax slice" aria-hidden="true">
  <g class="rays" stroke="#ffc83d" stroke-width="3" stroke-linecap="round">
    <line x1="500" y1="14" x2="500" y2="24"/><line x1="500" y1="92" x2="500" y2="102"/>
    <line x1="456" y1="58" x2="466" y2="58"/><line x1="534" y1="58" x2="544" y2="58"/>
    <line x1="469" y1="27" x2="476" y2="34"/><line x1="524" y1="82" x2="531" y2="89"/>
    <line x1="531" y1="27" x2="524" y2="34"/><line x1="476" y1="82" x2="469" y2="89"/>
  </g>
  <circle cx="500" cy="58" r="26" fill="#ffd24d"/>
  <g class="cloud1" fill="#fff" opacity=".95">
    <ellipse cx="330" cy="46" rx="30" ry="12"/><ellipse cx="350" cy="38" rx="20" ry="12"/><ellipse cx="312" cy="40" rx="16" ry="9"/>
  </g>
  <g class="cloud2" fill="#fff" opacity=".8">
    <ellipse cx="420" cy="92" rx="24" ry="9"/><ellipse cx="436" cy="86" rx="15" ry="9"/>
  </g>
  <path d="M0 200V128Q140 82 300 122T600 100V200Z" fill="#b5e3bf"/>
  <path d="M0 200V150Q190 96 370 142T600 128V200Z" fill="#6fcb88"/>
  <path d="M0 200V172Q170 132 340 166T600 156V200Z" fill="#34a56a"/>
  <g stroke="#1f7d4d" stroke-width="2" fill="none" opacity=".45" stroke-linecap="round">
    <path d="M60 196Q190 160 330 184"/><path d="M180 198Q300 168 430 186"/><path d="M330 198Q450 172 580 184"/>
  </g>
  <g fill="#2f9e5f"><circle cx="395" cy="118" r="9"/><rect x="393" y="124" width="4" height="12" fill="#7a5233"/></g>
</svg>
"""


def droplet_svg(fill: float, color: str) -> str:
    y = 148 - fill * 140  # droplet spans y=8..148
    wave = "M0 8 Q15 0 30 8 T60 8 T90 8 T120 8 T150 8 T180 8 T210 8 T240 8 V170 H0 Z"
    return f"""
    <svg class="drop" viewBox="0 0 120 160" aria-hidden="true">
      <defs>
        <clipPath id="dclip"><path d="M60 6 C60 6 12 62 12 100 a48 48 0 0 0 96 0 C108 62 60 6 60 6 Z"/></clipPath>
      </defs>
      <path d="M60 6 C60 6 12 62 12 100 a48 48 0 0 0 96 0 C108 62 60 6 60 6 Z"
            fill="rgba(255,255,255,.08)" stroke="{color}" stroke-width="3"/>
      <g clip-path="url(#dclip)">
        <g transform="translate(0,{y - 5:.1f})"><path class="wave2" d="{wave}" fill="{color}" opacity=".4"/></g>
        <g transform="translate(0,{y:.1f})"><path class="wave1" d="{wave}" fill="{color}" opacity=".95"/></g>
      </g>
      <path d="M34 92 C34 80 40 70 48 62" stroke="rgba(255,255,255,.7)" stroke-width="4"
            fill="none" stroke-linecap="round"/>
    </svg>
    """


def rain_html(n: int) -> str:
    out = []
    for i in range(n):
        left = 4 + i * (92 / max(n, 1))
        delay = (i * 0.37) % 2.2
        dur = 1.5 + (i % 3) * 0.45
        out.append(f'<span class="rd" style="left:{left:.1f}%;animation-delay:{delay:.2f}s;animation-duration:{dur:.2f}s"></span>')
    return "".join(out)


def pct(v, lo, hi):
    return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))


def icon(path: str) -> str:
    return (f'<span class="chip"><svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{path}</svg></span>')


ICON_TEMP = icon('<path d="M14 14.8V5a2 2 0 1 0-4 0v9.8a4 4 0 1 0 4 0Z"/>')
ICON_HUM = icon('<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z"/>')
ICON_SOIL = icon('<path d="M12 20V10M12 10c0-3 2-5 6-5 0 3-2 5-6 5ZM12 13c0-2.5-1.7-4-5-4 0 2.5 1.7 4 5 4Z"/>')
ICON_LDR = icon('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>')


def trend_chart(df: pd.DataFrame, col: str, color: str):
    d = df[[col]].dropna().reset_index()
    d.columns = ["Time", "Value"]
    axis_kw = dict(labelColor=MUTED, labelFont="DM Sans", labelFontSize=11)
    chart = (
        alt.Chart(d)
        .mark_area(
            interpolate="monotone",
            line={"color": color, "strokeWidth": 2.6},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#ffffff", offset=0),
                       alt.GradientStop(color=color, offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            ),
            opacity=0.35,
        )
        .encode(
            x=alt.X("Time:T", title=None,
                    axis=alt.Axis(grid=False, domainColor="#d5e1da", tickColor="#d5e1da", **axis_kw)),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(zero=False),
                    axis=alt.Axis(gridColor="#e9f0eb", domain=False, ticks=False, **axis_kw)),
            tooltip=[alt.Tooltip("Time:T", format="%d %b, %H:%M"),
                     alt.Tooltip("Value:Q", format=".1f")],
        )
        .properties(height=280, background="#ffffff")
        .configure_view(strokeWidth=0)
    )
    return chart


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

html(f"""
<div class="hero">
  <div class="hero-text">
    <h1>Smart Irrigation Dashboard</h1>
    <p>Live ESP32 readings and a trained model that decides how much water your field needs.</p>
  </div>
  {SCENE}
</div>
""")

st.write("")
c1, c2, c3, _ = st.columns([1.5, 1.1, 1, 3])
with c1:
    if st.button("Refresh sensor readings", type="primary"):
        st.cache_data.clear()
with c2:
    auto = st.toggle("Auto refresh", value=True)
with c3:
    every = st.selectbox("Every", ["15s", "30s", "60s"], index=1,
                         label_visibility="collapsed", disabled=not auto)


def render():
    try:
        model, metadata = load_model()
        data, values = get_latest_reading()
        cols = metadata["feature_columns_in_required_order"]

        X_live = pd.DataFrame([[values[f] for f in cols]], columns=cols)
        prediction = str(model.predict(X_live)[0])
        theme = level_theme(prediction)

        # ---- freshness ----
        created = data.get("created_at")
        age_txt, stale = "time unknown", False
        if created:
            ts = pd.to_datetime(created, utc=True)
            mins = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
            stale = mins > 10
            if mins < 1:
                age_txt = "updated just now"
            elif mins < 60:
                age_txt = f"updated {int(mins)} min ago"
            else:
                age_txt = f"updated {mins/60:.1f} h ago"

        html(f"""
        <span class="live"><span class="dot {'stale' if stale else ''}"></span>
        {'Sensor offline?' if stale else 'Live'} &nbsp;|&nbsp; {age_txt}</span>
        """)

        t = values["Temperature"]
        h = values["Humidity"]
        s = values["Soil Moisture Percentage"]
        ldr = values["RawLDR"]
        offset = 339.3 * (1 - max(0.0, min(100.0, s)) / 100)

        # ---- ROW 1: prediction | 2x2 readings ----
        html(f"""
        <div class="top">
          <div class="predict" style="--accent:{theme['color']}">
            {rain_html(theme['drops'])}
            <div class="drop-wrap">{droplet_svg(theme['fill'], theme['color'])}</div>
            <div class="pred-text">
              <p class="pred-label">Predicted irrigation level</p>
              <p class="pred-value">{prediction.upper()}</p>
              <p class="pred-advice">{theme['advice']}</p>
            </div>
          </div>
          <div class="metrics">
            <div class="card" style="--c:var(--soil)">
              <div class="k">{ICON_SOIL} Soil moisture</div>
              <div class="soil">
                <svg class="gauge" viewBox="0 0 120 120" aria-hidden="true">
                  <circle class="track" cx="60" cy="60" r="54" fill="none" stroke-width="12"/>
                  <circle class="prog" cx="60" cy="60" r="54" fill="none" stroke-width="12"
                          stroke="#16a34a" style="stroke-dashoffset:{offset:.1f}"/>
                </svg>
                <div class="v">{s:.1f}<small>%</small></div>
              </div>
            </div>
            <div class="card" style="--c:var(--temp)">
              <div class="k">{ICON_TEMP} Temperature</div>
              <div class="v">{t:.1f}<small>°C</small></div>
              <div class="bar"><i style="--w:{pct(t,0,50):.0f}%"></i></div>
            </div>
            <div class="card" style="--c:var(--hum)">
              <div class="k">{ICON_HUM} Humidity</div>
              <div class="v">{h:.1f}<small>%</small></div>
              <div class="bar"><i style="--w:{pct(h,0,100):.0f}%"></i></div>
            </div>
            <div class="card" style="--c:var(--light)">
              <div class="k">{ICON_LDR} Light (raw LDR)</div>
              <div class="v">{ldr:.0f}</div>
              <div class="bar"><i style="--w:{pct(ldr,0,4095):.0f}%"></i></div>
            </div>
          </div>
        </div>
        """)

        if stale:
            html('<div class="notice">The last reading is more than 10 minutes old. Check that the ESP32 is powered and connected to Wi-Fi.</div>')
        if s <= 1:
            html('<div class="notice">Soil moisture reads 0%. If the field is not completely dry, check that the soil sensor is pushed into the soil and its wiring is secure.</div>')

        # ---- ROW 2: model confidence | reading info ----
        conf_html = '<div class="info">This model does not provide probability scores.</div>'
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_live)[0]
            conf_html = ""
            for cls, p in sorted(zip(model.classes_, probs), key=lambda x: -x[1]):
                col = level_theme(str(cls))["color"]
                conf_html += f"""
                <div class="prob">
                  <span>{str(cls).title()}</span>
                  <div class="bar" style="--c:{col}"><i style="--w:{p*100:.1f}%"></i></div>
                  <b>{p*100:.1f}%</b>
                </div>"""

        html(f"""
        <div class="mid">
          <div class="panel">
            <div class="sec">How sure is the model?</div>
            {conf_html}
          </div>
          <div class="panel">
            <div class="sec">About this reading</div>
            <div class="info">
              <strong>Reading time (UTC):</strong> {created or 'Unknown'}<br>
              <strong>Source:</strong> ESP32 via ThingSpeak<br>
              <strong>Note:</strong> This is a prediction only. Verify the soil condition
              before irrigating. This dashboard does not control a water pump.
            </div>
          </div>
        </div>
        """)

        # ---- ROW 3: graphs underneath everything ----
        try:
            hist = get_history(100)
        except Exception:
            hist = pd.DataFrame()
        if not hist.empty:
            html('<div class="chartsec">Recent trends</div>')
            with st.container(border=True):
                tabs = st.tabs(["Soil moisture", "Temperature", "Humidity", "Light"])
                spec = [
                    ("Soil Moisture Percentage", "#16a34a"),
                    ("Temperature", "#ea580c"),
                    ("Humidity", "#2563eb"),
                    ("RawLDR", "#d99a06"),
                ]
                for tab, (col, color) in zip(tabs, spec):
                    with tab:
                        if col in hist:
                            st.altair_chart(trend_chart(hist, col, color),
                                            use_container_width=True, theme=None)

    except Exception as e:
        st.error(f"Could not generate prediction: {e}")


if auto:
    st.fragment(run_every=every)(render)()
else:
    render()
