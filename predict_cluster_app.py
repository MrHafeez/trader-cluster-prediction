import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trader Profiling Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e0e0e; }
[data-testid="stSidebar"]          { background-color: #111111; }
html, body, [class*="st-"]         { color: #f0f0f0; font-family: 'Segoe UI', sans-serif; }
.stButton>button {
    background-color: #1e88e5; color: white;
    border: none; padding: 0.6em 1.4em;
    font-weight: bold; border-radius: 6px; width: 100%;
}
.stButton>button:hover { background-color: #1565c0; }
hr { border-color: #333; }
small { color: #999; }
</style>
""", unsafe_allow_html=True)

# ── Cluster metadata ──────────────────────────────────────────────────────────
CLUSTER_NAMES = {
    0: "Risk-Averse Trader",
    1: "High-Frequency Trader",
    2: "Momentum Trader",
    3: "Balanced Trader",
}
CLUSTER_DESC = {
    0: "Low activity, high win rate, conservative approach.",
    1: "Very high trade volume, tight margins, fast execution.",
    2: "Infrequent but large bets, high volatility tolerance.",
    3: "Moderate activity across all metrics, well-rounded.",
}
CLUSTER_COLORS = {0: "#4caf50", 1: "#2196f3", 2: "#f44336", 3: "#ff9800"}

# ── Model loader (builds from scratch if pkl missing/incompatible) ────────────
def build_models():
    np.random.seed(42)
    n = 500
    def mc(aq, ap, apnl, pv, nt, bsr, wr):
        return pd.DataFrame({
            "avg_quantity":   np.random.normal(aq,   aq*0.1,        n).clip(1),
            "avg_price":      np.random.normal(ap,   ap*0.1,        n).clip(0),
            "avg_pnl":        np.random.normal(apnl, abs(apnl)*0.3, n),
            "pnl_volatility": np.random.normal(pv,   pv*0.1,        n).clip(0),
            "num_trades":     np.random.normal(nt,   nt*0.1,        n).clip(1),
            "buy_sell_ratio": np.random.normal(bsr,  bsr*0.1,       n).clip(0),
            "win_rate":       np.random.normal(wr,   0.05,          n).clip(0, 1),
        })
    df = pd.concat([
        mc(20,  500,  100,  200,  10,  1.0, 0.75),
        mc(200, 100,  50,   500,  200, 1.5, 0.55),
        mc(50,  3000, 500,  3000, 15,  2.0, 0.40),
        mc(80,  1200, 250,  1200, 50,  1.2, 0.62),
    ], ignore_index=True)
    X = df.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(Xp)
    return scaler, pca, kmeans

@st.cache_resource
def load_models():
    try:
        s = joblib.load("scaler.pkl")
        p = joblib.load("pca.pkl")
        k = joblib.load("kmeans_model.pkl")
        s.transform(np.zeros((1, 7)))   # compatibility check
        return s, p, k
    except Exception:
        return build_models()

scaler, pca, kmeans = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Trader Profiling")
    st.markdown("Predict a trader's behavioral cluster using key metrics.")
    st.markdown("---")
    st.markdown("### Cluster Guide")
    for idx, name in CLUSTER_NAMES.items():
        color = CLUSTER_COLORS[idx]
        st.markdown(
            f"<div style='padding:6px 10px; margin:4px 0; border-left:4px solid {color};"
            f"background:#1a1a1a; border-radius:4px;'>"
            f"<b style='color:{color}'>Cluster {idx}</b><br>"
            f"<span style='font-size:13px'>{name}</span><br>"
            f"<span style='font-size:11px;color:#aaa'>{CLUSTER_DESC[idx]}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.caption("Model: KMeans (k=4) on PCA-reduced, StandardScaled features.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧠 Trader Cluster Prediction")
st.caption("Enter a trader's metrics below to identify their behavioral profile.")
st.markdown("---")

with st.form("trader_form"):
    st.subheader("📊 Trading Metrics")
    col1, col2 = st.columns(2)

    with col1:
        avg_quantity = st.number_input(
            "Average Quantity", min_value=1.0, max_value=100000.0, value=45.0,
            help="Average number of units traded per trade")
        avg_pnl = st.number_input(
            "Average PnL (₹)", value=200.0,
            help="Average profit or loss per trade")
        num_trades = st.number_input(
            "Number of Trades", min_value=1.0, max_value=10000.0, value=20.0,
            help="Total number of trades in the period")
        win_rate = st.slider(
            "Win Rate", min_value=0.0, max_value=1.0, value=0.7, step=0.01,
            help="Proportion of profitable trades (0 to 1)")

    with col2:
        avg_price = st.number_input(
            "Average Price (₹)", min_value=0.0, max_value=100000.0, value=1750.0,
            help="Average trade execution price")
        pnl_volatility = st.number_input(
            "PnL Volatility", value=1500.0,
            help="Standard deviation of PnL across trades")
        buy_sell_ratio = st.number_input(
            "Buy/Sell Ratio", min_value=0.0, value=1.5, step=0.1,
            help="Ratio of buy trades to sell trades")

    submitted = st.form_submit_button("🔍 Predict Cluster")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_data = np.array([[avg_quantity, avg_price, avg_pnl,
                            pnl_volatility, num_trades, buy_sell_ratio, win_rate]])
    scaled  = scaler.transform(input_data)
    reduced = pca.transform(scaled)
    cluster = int(kmeans.predict(reduced)[0])

    name  = CLUSTER_NAMES[cluster]
    desc  = CLUSTER_DESC[cluster]
    color = CLUSTER_COLORS[cluster]

    st.markdown("---")
    st.markdown(
        f"<div style='padding:20px; background:#1a1a1a; border-left:6px solid {color};"
        f"border-radius:8px; margin-bottom:20px;'>"
        f"<h2 style='color:{color}; margin:0'>Cluster {cluster} — {name}</h2>"
        f"<p style='color:#ccc; margin-top:8px'>{desc}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📐 PCA Coordinates")
        coords = pd.DataFrame(reduced, columns=["PCA 1", "PCA 2"])
        st.dataframe(coords.style.format("{:.4f}"), use_container_width=True)
        csv = coords.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download PCA Coordinates",
            data=csv, file_name="pca_coords.csv", mime="text/csv"
        )

    with col_b:
        st.markdown("#### 📈 Input Summary")
        summary = pd.DataFrame({
            "Metric": ["Avg Quantity","Avg Price","Avg PnL","PnL Volatility",
                       "Num Trades","Buy/Sell Ratio","Win Rate"],
            "Value":  [avg_quantity, avg_price, avg_pnl, pnl_volatility,
                       num_trades, buy_sell_ratio, win_rate],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # Chart
    st.markdown("#### 🗺️ PCA Space")
    chart = (
        alt.Chart(coords.reset_index())
        .mark_circle(size=200, color=color)
        .encode(x="PCA 1:Q", y="PCA 2:Q", tooltip=["PCA 1","PCA 2"])
        .properties(height=300)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<hr><div style='text-align:center;font-size:12px;color:#666'>"
    "Trader Profiling Dashboard · Developed by MrHafeez · 2025"
    "</div>",
    unsafe_allow_html=True,
)
