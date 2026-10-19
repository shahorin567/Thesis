import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Smart Irrigation",
    page_icon="🌱",
    layout="centered",
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

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

:root{
  --bg:#0c1a17; --bg2:#11251f; --card:rgba(255,255,255,.045);
  --line:rgba(255,255,255,.09); --text:#eaf4ef; --muted:#8fa8a0;
  --water:#4cc9f0; --leaf:#7bd389; --sand:#f2b84b; --clay:#e07a5f;
}
html, body, [class*="css"], .stApp{font-family:'DM Sans',sans-serif;}
.stApp{
  background:
    radial-gradient(900px 500px at 85% -10%, rgba(76,201,240,.13), transparent 60%),
    radial-gradient(800px 500px at -10% 10%, rgba(123,211,137,.10), transparent 60%),
    linear-gradient(180deg,var(--bg) 0%, #0a1512 100%);
  color:var(--text);
}
.block-container{max-width:920px; padding-top:2.2rem; padding-bottom:3rem;}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}

/* ---------- hero ---------- */
.hero{display:flex; align-items:center; gap:18px; margin-bottom:.4rem;}
.hero h1{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800;
  font-size:2.5rem; line-height:1.05; margin:0; color:var(--text); letter-spacing:-.02em;
}
.hero p{margin:.35rem 0 0; color:var(--muted); font-size:.98rem;}
.sprout{width:68px; height:68px; flex:none; overflow:visible;}
.sprout .stem{stroke-dasharray:60; stroke-dashoffset:60; animation:grow 1.4s ease-out forwards;}
.sprout .leaf{transform-box:fill-box; transform-origin:0% 100%; transform:scale(0); animation:pop .9s .9s cubic-bezier(.2,1.4,.4,1) forwards;}
.sprout .leaf.r{transform-origin:100% 100%; animation-delay:1.15s;}
.sprout .sway{transform-origin:34px 62px; animation:sway 5s 2.2s ease-in-out infinite;}
@keyframes grow{to{stroke-dashoffset:0}}
@keyframes pop{to{transform:scale(1)}}
@keyframes sway{0%,100%{transform:rotate(-2.5deg)}50%{transform:rotate(2.5deg)}}

.live{display:inline-flex; align-items:center; gap:8px; font-size:.82rem; color:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:5px 12px; background:var(--card);}
.dot{width:8px; height:8px; border-radius:50%; background:var(--leaf); position:relative;}
.dot::after{content:""; position:absolute; inset:0; border-radius:50%; background:var(--leaf);
  animation:ping 1.8s ease-out infinite;}
.dot.stale{background:var(--clay)} .dot.stale::after{background:var(--clay)}
@keyframes ping{0%{transform:scale(1); opacity:.7}100%{transform:scale(3.2); opacity:0}}

/* ---------- prediction card (the one bold moment) ---------- */
.predict{
  position:relative; overflow:hidden; display:grid; grid-template-columns:170px 1fr; gap:28px;
  align-items:center; padding:28px 32px; margin:18px 0 14px; border-radius:26px;
  border:1px solid color-mix(in srgb, var(--accent) 35%, transparent);
  background:
    radial-gradient(500px 260px at 15% 50%, color-mix(in srgb, var(--accent) 22%, transparent), transparent 70%),
    var(--bg2);
}
.predict .rd{
  position:absolute; top:-14px; width:3px; height:16px; border-radius:3px; opacity:0;
  background:linear-gradient(transparent, var(--accent)); animation:fall linear infinite;
}
@keyframes fall{0%{transform:translateY(0); opacity:0}15%{opacity:.55}100%{transform:translateY(260px); opacity:0}}
.drop-wrap{position:relative; z-index:1; display:flex; justify-content:center;}
.drop{width:150px; height:200px; filter:drop-shadow(0 10px 28px color-mix(in srgb, var(--accent) 45%, transparent));}
.wave1{animation:slide 5s linear infinite;}
.wave2{animation:slide 8s linear infinite reverse;}
@keyframes slide{from{transform:translateX(0)}to{transform:translateX(-60px)}}
.pred-text{position:relative; z-index:1;}
.pred-label{color:var(--muted); font-size:.9rem; margin:0 0 4px;}
.pred-value{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:3.6rem; line-height:1;
  letter-spacing:-.03em; margin:0; color:var(--accent);
}
.pred-advice{margin:.9rem 0 0; color:var(--text); font-size:1rem; line-height:1.55; max-width:44ch;}
@media (max-width:640px){
  .predict{grid-template-columns:1fr; text-align:center; padding:24px 20px;}
  .pred-advice{margin-left:auto; margin-right:auto}
  .hero h1{font-size:1.9rem}
}

/* ---------- metric cards ---------- */
.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:10px 0 6px;}
.card{background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px 20px;}
.card .k{display:flex; align-items:center; gap:8px; color:var(--muted); font-size:.88rem;}
.card .v{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:2.1rem; margin:.35rem 0 .7rem; color:var(--text);}
.card .v small{font-size:1rem; font-weight:500; color:var(--muted); margin-left:3px;}
.bar{height:6px; border-radius:6px; background:rgba(255,255,255,.08); overflow:hidden;}
.bar i{display:block; height:100%; width:var(--w); border-radius:6px; background:var(--c);
  animation:fillbar 1.2s cubic-bezier(.2,.8,.2,1) both;}
@keyframes fillbar{from{width:0}}
.ico{width:20px; height:20px; flex:none;}

/* soil gauge */
.soil{display:flex; align-items:center; gap:16px}
.gauge{width:92px; height:92px; flex:none; transform:rotate(-90deg);}
.gauge .track{stroke:rgba(255,255,255,.09)}
.gauge .prog{stroke-dasharray:339.3; animation:gauge 1.6s cubic-bezier(.2,.8,.2,1) both; stroke-linecap:round;}
@keyframes gauge{from{stroke-dashoffset:339.3}}
.soil .v{margin:.2rem 0 0}

/* ---------- probabilities ---------- */
.sec{font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.25rem; margin:1.8rem 0 .7rem;}
.prob{display:grid; grid-template-columns:90px 1fr 56px; align-items:center; gap:12px; margin:9px 0; font-size:.93rem;}
.prob .bar{height:10px}
.prob b{text-align:right; font-weight:600}

.notice{border-radius:14px; padding:12px 16px; margin:12px 0; font-size:.92rem; line-height:1.5;
  border:1px solid rgba(242,184,75,.35); background:rgba(242,184,75,.08); color:#f6d99a;}
.foot{margin-top:1.6rem; padding:14px 16px; border-radius:14px; border:1px dashed var(--line);
  color:var(--muted); font-size:.88rem; line-height:1.55;}

/* streamlit widgets */
.stButton>button{
  border-radius:12px; border:1px solid color-mix(in srgb, var(--water) 55%, transparent);
  background:rgba(76,201,240,.12); color:var(--text); font-weight:600; padding:.5rem 1.1rem;
  transition:background .2s, transform .15s;
}
.stButton>button:hover{background:rgba(76,201,240,.24); border-color:var(--water); color:#fff; transform:translateY(-1px);}
.stButton>button:focus-visible{outline:2px solid var(--water); outline-offset:2px;}
button[data-baseweb="tab"]{font-weight:600}

@media (prefers-reduced-motion:reduce){
  *{animation:none !important; transition:none !important;}
  .sprout .stem{stroke-dashoffset:0} .sprout .leaf{transform:scale(1)}
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
        return dict(color="#4cc9f0", fill=0.86, drops=14,
                    advice="Dry conditions. Check the soil by hand, then irrigate soon.")
    if "med" in l:
        return dict(color="#7bd389", fill=0.55, drops=8,
                    advice="Moderate watering is suggested. A normal irrigation cycle should be enough.")
    if "low" in l:
        return dict(color="#f2b84b", fill=0.28, drops=4,
                    advice="Only a little water is needed right now. Light irrigation or wait.")
    return dict(color="#8fa8a0", fill=0.08, drops=0,
                advice="No irrigation needed at the moment. The soil is fine.")


SPROUT = """
<svg class="sprout" viewBox="0 0 68 68" aria-hidden="true">
  <g class="sway">
    <path class="stem" d="M34 62 C34 46 34 36 34 22" stroke="#7bd389" stroke-width="3.5" fill="none" stroke-linecap="round"/>
    <path class="leaf" d="M34 34 C18 34 8 24 8 12 C22 12 34 20 34 34Z" fill="#7bd389"/>
    <path class="leaf r" d="M34 26 C34 12 44 4 60 4 C60 18 50 26 34 26Z" fill="#4fae66"/>
  </g>
  <ellipse cx="34" cy="63" rx="20" ry="4" fill="#5b3f2c"/>
</svg>
"""


def droplet_svg(fill: float, color: str) -> str:
    y = 148 - fill * 140  # droplet spans y=8..148
    wave = ("M0 8 Q15 0 30 8 T60 8 T90 8 T120 8 T150 8 T180 8 T210 8 T240 8 V170 H0 Z")
    return f"""
    <svg class="drop" viewBox="0 0 120 160" aria-hidden="true">
      <defs>
        <clipPath id="dclip"><path d="M60 6 C60 6 12 62 12 100 a48 48 0 0 0 96 0 C108 62 60 6 60 6 Z"/></clipPath>
      </defs>
      <path d="M60 6 C60 6 12 62 12 100 a48 48 0 0 0 96 0 C108 62 60 6 60 6 Z"
            fill="rgba(255,255,255,.05)" stroke="{color}" stroke-width="2.5"/>
      <g clip-path="url(#dclip)">
        <g transform="translate(0,{y - 5:.1f})"><path class="wave2" d="{wave}" fill="{color}" opacity=".35"/></g>
        <g transform="translate(0,{y:.1f})"><path class="wave1" d="{wave}" fill="{color}" opacity=".85"/></g>
      </g>
      <path d="M34 92 C34 80 40 70 48 62" stroke="rgba(255,255,255,.55)" stroke-width="4"
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


ICON_TEMP = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="#e07a5f" stroke-width="2" stroke-linecap="round"><path d="M14 14.8V5a2 2 0 1 0-4 0v9.8a4 4 0 1 0 4 0Z"/></svg>'
ICON_HUM = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="#4cc9f0" stroke-width="2" stroke-linecap="round"><path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z"/></svg>'
ICON_SOIL = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="#7bd389" stroke-width="2" stroke-linecap="round"><path d="M12 20V10M12 10c0-3 2-5 6-5 0 3-2 5-6 5ZM12 13c0-2.5-1.7-4-5-4 0 2.5 1.7 4 5 4Z"/></svg>'
ICON_LDR = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="#f2b84b" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

html(f"""
<div class="hero">
  {SPROUT}
  <div>
    <h1>Smart Irrigation Dashboard</h1>
    <p>Live ESP32 readings and a trained model that decides how much water your field needs.</p>
  </div>
</div>
""")

c1, c2, c3 = st.columns([1.3, 1.2, 1.5])
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
        if stale:
            html('<div class="notice">The last reading is more than 10 minutes old. Check that the ESP32 is powered and connected to Wi-Fi.</div>')

        # ---- prediction ----
        html(f"""
        <div class="predict" style="--accent:{theme['color']}">
          {rain_html(theme['drops'])}
          <div class="drop-wrap">{droplet_svg(theme['fill'], theme['color'])}</div>
          <div class="pred-text">
            <p class="pred-label">Predicted irrigation level</p>
            <p class="pred-value">{prediction.upper()}</p>
            <p class="pred-advice">{theme['advice']}</p>
          </div>
        </div>
        """)

        # ---- readings ----
        t = values["Temperature"]
        h = values["Humidity"]
        s = values["Soil Moisture Percentage"]
        ldr = values["RawLDR"]
        s_pct = max(0.0, min(100.0, s))
        offset = 339.3 * (1 - s_pct / 100)

        html(f"""
        <div class="grid">
          <div class="card" style="grid-column:1/-1">
            <div class="soil">
              <svg class="gauge" viewBox="0 0 120 120" aria-hidden="true">
                <circle class="track" cx="60" cy="60" r="54" fill="none" stroke-width="10"/>
                <circle class="prog" cx="60" cy="60" r="54" fill="none" stroke-width="10"
                        stroke="#7bd389" style="stroke-dashoffset:{offset:.1f}"/>
              </svg>
              <div>
                <div class="k">{ICON_SOIL} Soil moisture</div>
                <div class="v">{s:.1f}<small>%</small></div>
              </div>
            </div>
          </div>
          <div class="card">
            <div class="k">{ICON_TEMP} Temperature</div>
            <div class="v">{t:.1f}<small>°C</small></div>
            <div class="bar"><i style="--w:{pct(t,0,50):.0f}%;--c:#e07a5f"></i></div>
          </div>
          <div class="card">
            <div class="k">{ICON_HUM} Humidity</div>
            <div class="v">{h:.1f}<small>%</small></div>
            <div class="bar"><i style="--w:{pct(h,0,100):.0f}%;--c:#4cc9f0"></i></div>
          </div>
          <div class="card">
            <div class="k">{ICON_LDR} Light (raw LDR)</div>
            <div class="v">{ldr:.0f}</div>
            <div class="bar"><i style="--w:{pct(ldr,0,4095):.0f}%;--c:#f2b84b"></i></div>
          </div>
        </div>
        """)

        if s <= 1:
            html('<div class="notice">Soil moisture reads 0%. If the field is not completely dry, check that the soil sensor is pushed into the soil and its wiring is secure.</div>')

        # ---- model scores ----
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_live)[0]
            rows = ""
            for cls, p in sorted(zip(model.classes_, probs), key=lambda x: -x[1]):
                col = level_theme(str(cls))["color"]
                rows += f"""
                <div class="prob">
                  <span>{str(cls).title()}</span>
                  <div class="bar"><i style="--w:{p*100:.1f}%;--c:{col}"></i></div>
                  <b>{p*100:.1f}%</b>
                </div>"""
            html(f'<div class="sec">How sure is the model?</div>{rows}')

        # ---- history ----
        try:
            hist = get_history(100)
        except Exception:
            hist = pd.DataFrame()
        if not hist.empty:
            html('<div class="sec">Recent trends</div>')
            tabs = st.tabs(["Soil moisture", "Temperature", "Humidity", "Light"])
            spec = [
                ("Soil Moisture Percentage", "#7bd389"),
                ("Temperature", "#e07a5f"),
                ("Humidity", "#4cc9f0"),
                ("RawLDR", "#f2b84b"),
            ]
            for tab, (col, color) in zip(tabs, spec):
                with tab:
                    if col in hist:
                        st.area_chart(hist[[col]], color=color, height=240)

        html(f"""
        <div class="foot">
          Reading time (UTC): {created or 'Unknown'}<br>
          This is a prediction only. Verify the soil condition before irrigating.
          This dashboard does not control a water pump.
        </div>
        """)

    except Exception as e:
        st.error(f"Could not generate prediction: {e}")


if auto:
    st.fragment(run_every=every)(render)()
else:
    render()
