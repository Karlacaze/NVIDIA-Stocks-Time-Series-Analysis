import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler          # igual que notebook v1
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional,
    LayerNormalization, Input, Conv1D,
    Concatenate, GlobalAveragePooling1D               # igual que notebook v1
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NVDA Forecast", page_icon="📈", layout="wide")

# ── DESIGN SYSTEM ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:       #060810;
  --surface:  #0c0f1d;
  --card:     #111528;
  --border:   #1e2340;
  --accent:   #5e7bff;
  --green:    #00ffb3;
  --red:      #ff4466;
  --yellow:   #ffd060;
  --muted:    #4a5080;
  --text:     #e0e4ff;
}

html, body, [class*="css"] {
  font-family: 'Syne', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.top-bar {
  background: linear-gradient(90deg, #0c0f1d 0%, #0f1535 50%, #0c0f1d 100%);
  border-bottom: 1px solid var(--border);
  padding: 1.2rem 2rem;
  margin: -1rem -1rem 1.5rem -1rem;
  display: flex; align-items: center; gap: 1.2rem;
}
.top-bar .logo {
  font-family: 'Space Mono', monospace;
  font-size: 1.4rem; font-weight: 700;
  background: linear-gradient(135deg, #5e7bff, #00ffb3);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -1px;
}
.top-bar .sub { color: var(--muted); font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; }

.kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin: 1rem 0; }
.kpi {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  position: relative; overflow: hidden;
}
.kpi::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--green));
}
.kpi .label { color: var(--muted); font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.4rem; }
.kpi .val   { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--text); }
.kpi .delta { font-size: 0.78rem; margin-top: 0.25rem; }
.kpi .delta.up      { color: var(--green); }
.kpi .delta.down    { color: var(--red); }
.kpi .delta.neutral { color: var(--muted); }

.forecast-hero {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem 2rem;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1rem;
}
.forecast-hero .price { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; line-height: 1; }
.forecast-hero .badge {
  padding: 0.4rem 1rem; border-radius: 50px;
  font-size: 0.9rem; font-weight: 700; letter-spacing: 1px;
}
.badge-bull { background: rgba(0,255,179,0.12); color: var(--green); border: 1px solid rgba(0,255,179,0.3); }
.badge-bear { background: rgba(255,68,102,0.12); color: var(--red);   border: 1px solid rgba(255,68,102,0.3); }

.section-title {
  font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase;
  color: var(--accent); margin: 1.5rem 0 0.75rem 0; font-weight: 600;
}

.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.5rem 0 1rem 0; }
.stat {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.7rem 1rem; flex: 1; min-width: 100px;
}
.stat .s-label { color: var(--muted); font-size: 0.65rem; letter-spacing: 1.5px; text-transform: uppercase; }
.stat .s-val   { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: var(--text); margin-top: 0.2rem; }

.info-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
.info-table td { padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
.info-table td:first-child { color: var(--muted); width: 40%; }
.info-table td:last-child  { font-family: 'Space Mono', monospace; color: var(--text); }

.bar-outer { background: var(--border); border-radius: 4px; height: 6px; margin-top: 4px; }
.bar-inner { height: 6px; border-radius: 4px; background: linear-gradient(90deg, var(--accent), var(--green)); }

.tag {
  display: inline-block;
  background: rgba(94,123,255,0.1); border: 1px solid rgba(94,123,255,0.25);
  color: #8fa3ff; border-radius: 4px;
  padding: 0.2rem 0.5rem; font-size: 0.72rem;
  font-family: 'Space Mono', monospace; margin: 2px;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border: none !important;
  color: var(--muted) !important; font-family: 'Syne', sans-serif !important;
  font-size: 0.82rem !important; letter-spacing: 1px; text-transform: uppercase;
  padding: 0.6rem 1rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

.stSlider > div > div > div { background: var(--accent) !important; }
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
}
div[data-testid="stMetric"] {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; padding: 1rem !important;
}
div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: var(--text) !important; }
div[data-testid="stMetricDelta"] svg { display: none; }

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--card) !important; border-color: var(--border) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS — exactamente igual que notebook v1 ─────────────
FEATURE_COLS = [
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_20',
    'ma5_ratio', 'ma10_ratio', 'ma20_ratio', 'ma50_ratio',
    'vol_5', 'vol_10', 'vol_20',
    'rsi', 'macd_norm', 'macd_signal', 'bb_pos',
    'hl_pct', 'oc_pct', 'vol_ratio'
]
N_FEATURES  = len(FEATURE_COLS)   # 19
WINDOW_SIZE = 30                  # igual notebook v1


# ── HELPERS — código copiado 1:1 del notebook v1 ──────────────
def add_features(df):
    d = df.copy()
    d['ret_1']  = d['Adj Close'].pct_change(1)
    d['ret_3']  = d['Adj Close'].pct_change(3)
    d['ret_5']  = d['Adj Close'].pct_change(5)
    d['ret_10'] = d['Adj Close'].pct_change(10)
    d['ret_20'] = d['Adj Close'].pct_change(20)
    for w in [5, 10, 20, 50]:
        d[f'ma{w}_ratio'] = d['Adj Close'].rolling(w).mean() / d['Adj Close'] - 1
    d['vol_5']  = d['ret_1'].rolling(5).std()
    d['vol_10'] = d['ret_1'].rolling(10).std()
    d['vol_20'] = d['ret_1'].rolling(20).std()
    delta = d['Adj Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['rsi'] = (100 - (100 / (1 + gain / (loss + 1e-8)))) / 100
    ema12 = d['Adj Close'].ewm(span=12).mean()
    ema26 = d['Adj Close'].ewm(span=26).mean()
    macd  = ema12 - ema26
    d['macd_norm']   = macd / d['Adj Close']
    d['macd_signal'] = macd.ewm(span=9).mean() / d['Adj Close']
    bb_mid = d['Adj Close'].rolling(20).mean()
    bb_std = d['Adj Close'].rolling(20).std()
    d['bb_pos']    = (d['Adj Close'] - bb_mid) / (2 * bb_std + 1e-8)
    d['hl_pct']    = (d['High'] - d['Low']) / d['Adj Close']
    d['oc_pct']    = (d['Adj Close'] - d['Open']) / d['Open']
    d['vol_ratio'] = np.log1p(d['Volume']) / np.log1p(d['Volume'].rolling(20).mean() + 1e-8) - 1
    return d.dropna().reset_index(drop=True)


def create_sequences(features, prices, window):
    """Igual que notebook v1: target = retorno desde inicio de ventana."""
    Xs, ys, price_refs = [], [], []
    for i in range(window, len(features)):
        Xs.append(features[i-window:i])
        ref_price = prices[i - window]
        ys.append((prices[i] / ref_price) - 1)
        price_refs.append(ref_price)
    return np.array(Xs), np.array(ys), np.array(price_refs)


def build_model(window, n_features):
    """Arquitectura idéntica al notebook v1."""
    inputs = Input(shape=(window, n_features))
    # CNN branch
    c = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    c = Conv1D(32, kernel_size=3, activation='relu', padding='same')(c)
    c = GlobalAveragePooling1D()(c)
    # LSTM branch
    x = Bidirectional(LSTM(96, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(48, return_sequences=False))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.20)(x)
    # Fusion
    merged = Concatenate()([x, c])
    out = Dense(64, activation='relu')(merged)
    out = Dropout(0.15)(out)
    out = Dense(32, activation='relu')(out)
    out = Dense(1)(out)
    return Model(inputs, out)


@st.cache_resource(show_spinner=False)
def load_and_train(csv_bytes):
    import io
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    df_raw = df_raw.sort_values('Date').reset_index(drop=True)

    # Dataset info (antes del filtro)
    raw_shape    = df_raw.shape
    raw_nulls    = df_raw.isnull().sum().to_dict()
    raw_dupes    = int(df_raw.duplicated().sum())
    raw_date_min = df_raw['Date'].min()
    raw_date_max = df_raw['Date'].max()

    # Filtro 2020+ igual que notebook v1
    df_raw = df_raw[df_raw['Date'] >= '2020-01-01'].reset_index(drop=True)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])

    num_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    outlier_info = {}
    for col in num_cols:
        Q1, Q3 = df_raw[col].quantile(0.25), df_raw[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df_raw[col] < Q1 - 1.5*IQR) | (df_raw[col] > Q3 + 1.5*IQR)).sum())
        outlier_info[col] = {'count': n_out, 'pct': round(n_out / len(df_raw) * 100, 2)}

    describe = df_raw[num_cols].describe().round(4)

    df = add_features(df_raw)

    # MinMaxScaler(-1,1) igual que notebook v1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features_scaled = scaler.fit_transform(df[FEATURE_COLS])

    prices = df['Adj Close'].values
    X_all, y_all, refs_all = create_sequences(features_scaled, prices, WINDOW_SIZE)

    n         = len(X_all)
    train_end = int(n * 0.72)
    val_end   = int(n * 0.86)

    X_train, y_train = X_all[:train_end],        y_all[:train_end]
    X_val,   y_val   = X_all[train_end:val_end],  y_all[train_end:val_end]
    X_test,  y_test  = X_all[val_end:],           y_all[val_end:]
    refs_test        = refs_all[val_end:]

    # Compilación idéntica al notebook v1
    # ── Semilla fija para reproducibilidad ──────────────────────
    import random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_model(WINDOW_SIZE, N_FEATURES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss='huber',
        metrics=['mae']
    )
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', verbose=0),
        ReduceLROnPlateau(factor=0.4, patience=8, min_lr=5e-7, verbose=0)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=32,
        callbacks=callbacks, verbose=0
    )

    # Evaluación idéntica al notebook v1
    y_pred_norm    = model.predict(X_test, verbose=0).flatten()
    y_real_norm    = y_test.flatten()
    test_start_idx = val_end + WINDOW_SIZE
    prices_real_df = df['Adj Close'].iloc[test_start_idx:test_start_idx + len(y_pred_norm)].values
    prices_pred_arr = refs_test * (1 + y_pred_norm)

    mae     = mean_absolute_error(prices_real_df, prices_pred_arr)
    rmse    = np.sqrt(mean_squared_error(prices_real_df, prices_pred_arr))
    r2      = r2_score(prices_real_df, prices_pred_arr)
    mape    = np.mean(np.abs((prices_real_df - prices_pred_arr) / (prices_real_df + 1e-8))) * 100
    # Directional accuracy — fórmula exacta del notebook v1
    dir_acc = np.mean(np.sign(y_pred_norm) == np.sign(y_real_norm)) * 100

    test_dates = df['Date'].iloc[test_start_idx:test_start_idx + len(y_pred_norm)].values

    # Forecast 30 días — lógica idéntica al notebook v1
    last_features = features_scaled[-WINDOW_SIZE:]
    forecast_prices_list = []
    forecast_dates = pd.bdate_range(
        start=df['Date'].max() + pd.Timedelta(days=1), periods=30
    )
    current_window = last_features.copy()
    last_price = df['Adj Close'].iloc[-1]

    for step in range(30):
        ref_price = prices[-(WINDOW_SIZE - step)] if step < WINDOW_SIZE else last_price
        pred_norm = model.predict(current_window[np.newaxis, :, :], verbose=0)[0, 0]
        new_price = ref_price * (1 + pred_norm)
        forecast_prices_list.append(new_price)
        new_row    = current_window[-1].copy()
        new_row[0] = np.clip(pred_norm, -0.5, 0.5)
        current_window = np.vstack([current_window[1:], new_row])

    forecast_prices_arr = np.array(forecast_prices_list)

    metrics = {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'dir_acc': dir_acc,
        'train_size': len(X_train), 'val_size': len(X_val), 'test_size': len(X_test),
        'total_params': model.count_params(),
        'epochs_trained': len(history.history['loss']),
    }
    data_info = {
        'raw_shape': raw_shape, 'raw_nulls': raw_nulls,
        'raw_dupes': raw_dupes, 'raw_date_min': raw_date_min,
        'raw_date_max': raw_date_max, 'filtered_rows': len(df_raw),
        'after_fe_rows': len(df), 'outlier_info': outlier_info,
        'describe': describe,
    }
    return (df, model, history, metrics, test_dates, prices_real_df, prices_pred_arr,
            forecast_dates, forecast_prices_arr, features_scaled, prices, scaler, data_info)


# ── PLOT HELPERS ──────────────────────────────────────────────
DARK = dict(template='plotly_dark', paper_bgcolor='#060810', plot_bgcolor='#0c0f1d')

def styled_layout(fig, title="", height=400):
    fig.update_layout(
        **DARK, height=height,
        title=dict(text=title, x=0.5, font=dict(size=13, color='#e0e4ff')),
        margin=dict(t=45, b=40, l=55, r=20), hovermode='x unified',
        font=dict(family='Syne', color='#e0e4ff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='#1e2340'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='#1e2340'),
        legend=dict(bgcolor='rgba(0,0,0,0.4)', bordercolor='#1e2340', borderwidth=1)
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem 0'>
      <div style='font-family:"Space Mono",monospace; font-size:1.1rem; font-weight:700;
                  background:linear-gradient(135deg,#5e7bff,#00ffb3);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
        NVDA / FORECAST
      </div>
      <div style='color:#4a5080; font-size:0.7rem; letter-spacing:2px; margin-top:2px'>
        CNN + BILSTM · v1
      </div>
    </div>
    <hr style='border-color:#1e2340; margin:0.5rem 0 1rem 0'>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload NVDA.csv", type=["csv"])

    st.markdown("<hr style='border-color:#1e2340; margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#4a5080; font-size:0.65rem; letter-spacing:2px; margin-bottom:0.5rem'>FORECAST</div>", unsafe_allow_html=True)
    forecast_days = st.slider("Days ahead", 5, 30, 30, 5)

    st.markdown("<hr style='border-color:#1e2340; margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#4a5080; font-size:0.65rem; letter-spacing:2px; margin-bottom:0.5rem'>CHART</div>", unsafe_allow_html=True)
    show_history_days = st.slider("Historical context", 30, 180, 90, 10)

    st.markdown("""
    <hr style='border-color:#1e2340; margin:1rem 0'>
    <div style='color:#4a5080; font-size:0.65rem; letter-spacing:2px; margin-bottom:0.8rem'>MODEL INFO</div>
    <div style='font-size:0.75rem; line-height:2; color:#6070a0'>
      Window &nbsp;&nbsp;<span style='color:#e0e4ff; font-family:"Space Mono",monospace'>30 days</span><br>
      Features &nbsp;<span style='color:#e0e4ff; font-family:"Space Mono",monospace'>19</span><br>
      Params &nbsp;&nbsp;&nbsp;<span style='color:#e0e4ff; font-family:"Space Mono",monospace'>202,465</span><br>
      Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#e0e4ff; font-family:"Space Mono",monospace'>Huber</span><br>
      Scaler &nbsp;&nbsp;&nbsp;<span style='color:#e0e4ff; font-family:"Space Mono",monospace'>MinMax(-1,1)</span><br>
      Split &nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#e0e4ff; font-family:"Space Mono",monospace'>72/14/14</span>
    </div>
    """, unsafe_allow_html=True)


# ── TOP BAR ───────────────────────────────────────────────────
st.markdown("""
<div class='top-bar'>
  <div>
    <div class='logo'>NVDA · STOCK FORECAST</div>
    <div class='sub'>CNN + Bidirectional LSTM · NVIDIA Corporation</div>
  </div>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div style='background:#0c0f1d; border:1px dashed #1e2340; border-radius:14px;
                padding:3rem; text-align:center; margin-top:2rem'>
      <div style='font-size:2.5rem; margin-bottom:1rem'>📂</div>
      <div style='color:#e0e4ff; font-size:1.1rem; font-weight:600; margin-bottom:0.5rem'>
        Upload your NVDA.csv to begin
      </div>
      <div style='color:#4a5080; font-size:0.85rem'>
        Required columns: Date · Adj Close · Close · High · Low · Open · Volume
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Training CNN + BiLSTM model… (~1-2 min)"):
    (df, model, history, metrics, test_dates, prices_real_df, prices_pred_arr,
     forecast_dates, forecast_prices_arr, features_scaled, prices, scaler, data_info) = load_and_train(uploaded_file.getvalue())

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "FORECAST", "PERFORMANCE", "TRAINING", "ARCHITECTURE", "DATASET"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════
with tab1:
    last_price  = df['Adj Close'].iloc[-1]
    last_date   = df['Date'].iloc[-1].date()
    valid_dates = [d.date() for d in forecast_dates[:forecast_days]]

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("<div class='section-title'>Select date</div>", unsafe_allow_html=True)
        selected_date = st.selectbox(
            "", options=valid_dates,
            format_func=lambda d: d.strftime("%a %b %d, %Y"),
            label_visibility="collapsed"
        )

    sel_idx   = valid_dates.index(selected_date)
    sel_price = forecast_prices_arr[sel_idx]
    sel_delta = sel_price - last_price
    sel_pct   = sel_delta / last_price * 100
    is_bull   = sel_delta >= 0
    badge_cls = "badge-bull" if is_bull else "badge-bear"
    badge_txt = "▲ BULLISH" if is_bull else "▼ BEARISH"
    price_col = "#00ffb3" if is_bull else "#ff4466"

    with col_r:
        st.markdown(f"""
        <div class='forecast-hero'>
          <div>
            <div style='color:#4a5080; font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.4rem'>
              Day +{sel_idx+1} · {selected_date}
            </div>
            <div class='price' style='color:{price_col}'>${sel_price:.2f}</div>
            <div style='color:#4a5080; font-size:0.85rem; margin-top:0.4rem'>
              from ${last_price:.2f} on {last_date} &nbsp;·&nbsp;
              <span style='color:{price_col}'>{sel_pct:+.2f}%</span>
            </div>
          </div>
          <div class='badge {badge_cls}'>{badge_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    # KPIs — dir_acc viene de la fórmula exacta del notebook v1
    day_n_price = forecast_prices_arr[forecast_days - 1]
    day_n_pct   = (day_n_price - last_price) / last_price * 100
    max_p = forecast_prices_arr[:forecast_days].max()
    min_p = forecast_prices_arr[:forecast_days].min()
    max_d = forecast_dates[np.argmax(forecast_prices_arr[:forecast_days])].strftime("%b %d")
    min_d = forecast_dates[np.argmin(forecast_prices_arr[:forecast_days])].strftime("%b %d")

    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi'>
        <div class='label'>Current</div>
        <div class='val'>${last_price:.2f}</div>
        <div class='delta neutral'>last close</div>
      </div>
      <div class='kpi'>
        <div class='label'>Day +{forecast_days}</div>
        <div class='val'>${day_n_price:.2f}</div>
        <div class='delta {"up" if day_n_pct>=0 else "down"}'>{day_n_pct:+.2f}%</div>
      </div>
      <div class='kpi'>
        <div class='label'>Peak ({max_d})</div>
        <div class='val'>${max_p:.2f}</div>
        <div class='delta up'>{(max_p-last_price)/last_price*100:+.2f}%</div>
      </div>
      <div class='kpi'>
        <div class='label'>Trough ({min_d})</div>
        <div class='val'>${min_p:.2f}</div>
        <div class='delta down'>{(min_p-last_price)/last_price*100:+.2f}%</div>
      </div>
      <div class='kpi'>
        <div class='label'>Dir. Accuracy</div>
        <div class='val'>{metrics['dir_acc']:.1f}%</div>
        <div class='delta neutral'>on test set</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    recent = df.tail(show_history_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent['Date'], y=recent['Adj Close'], mode='lines', name='Historical',
        line=dict(color='#5e7bff', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_dates[:forecast_days]), y=forecast_prices_arr[:forecast_days],
        mode='lines+markers', name='Forecast',
        line=dict(color='#00ffb3', width=2.5),
        marker=dict(size=4, color='#00ffb3')
    ))
    fig.add_trace(go.Scatter(
        x=[selected_date], y=[sel_price], mode='markers', name='Selected',
        marker=dict(size=13, color=price_col, symbol='star',
                    line=dict(color='white', width=1.5))
    ))
    fig.add_vline(x=str(df['Date'].max().date()),
                  line=dict(color='#1e2340', dash='dash', width=1.5))
    styled_layout(fig, f"NVDA — Next {forecast_days} Business Days", height=420)
    fig.update_yaxes(tickprefix='$')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Full forecast table</div>", unsafe_allow_html=True)
    fc_df = pd.DataFrame({
        'Date':       [d.date() for d in forecast_dates[:forecast_days]],
        'Day':        [f"+{i+1}" for i in range(forecast_days)],
        'Forecast':   [f"${p:.2f}" for p in forecast_prices_arr[:forecast_days]],
        'Δ vs Today': [f"{((p-last_price)/last_price*100):+.2f}%" for p in forecast_prices_arr[:forecast_days]],
        'Signal':     ["▲ BUY" if p >= last_price else "▼ SELL" for p in forecast_prices_arr[:forecast_days]]
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Test set metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE",           f"${metrics['mae']:.2f}")
    c2.metric("RMSE",          f"${metrics['rmse']:.2f}")
    c3.metric("R²",            f"{metrics['r2']:.4f}")
    c4.metric("MAPE",          f"{metrics['mape']:.2f}%")
    c5.metric("Dir. Accuracy", f"{metrics['dir_acc']:.1f}%")

    st.markdown("<div class='section-title'>Real vs Predicted</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=test_dates, y=prices_real_df, mode='lines', name='Real',
        line=dict(color='#5e7bff', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=test_dates, y=prices_pred_arr, mode='lines', name='Predicted',
        line=dict(color='#ff6b35', width=2, dash='dash')
    ))
    fig2.add_trace(go.Scatter(
        x=list(test_dates) + list(test_dates[::-1]),
        y=list(prices_real_df) + list(prices_pred_arr[::-1]),
        fill='toself', fillcolor='rgba(255,107,53,0.06)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False
    ))
    styled_layout(
        fig2,
        f"MAE ${metrics['mae']:.2f}  ·  RMSE ${metrics['rmse']:.2f}  ·  R² {metrics['r2']:.4f}  ·  Dir {metrics['dir_acc']:.1f}%",
        height=400
    )
    fig2.update_yaxes(tickprefix='$')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Residuals distribution</div>", unsafe_allow_html=True)
    residuals = prices_real_df - prices_pred_arr
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color='#5e7bff', opacity=0.8))
    fig_r.add_vline(x=0, line=dict(color='#00ffb3', dash='dash', width=1.5))
    styled_layout(fig_r, "Residuals (Real − Predicted)", height=290)
    fig_r.update_xaxes(title='Residual ($)')
    fig_r.update_yaxes(title='Count')
    st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — TRAINING
# ══════════════════════════════════════════════════════════════
with tab3:
    ep   = list(range(1, len(history.history['loss']) + 1))
    tl   = history.history['loss']
    vl   = history.history['val_loss']
    tm   = history.history['mae']
    vm   = history.history['val_mae']
    best = int(np.argmin(vl)) + 1

    col_a, col_b = st.columns(2)
    with col_a:
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=ep, y=tl, mode='lines', name='Train',
                                   line=dict(color='#5e7bff', width=2)))
        fig_l.add_trace(go.Scatter(x=ep, y=vl, mode='lines', name='Val',
                                   line=dict(color='#ff4466', width=2, dash='dash')))
        fig_l.add_vline(x=best, line=dict(color='#00ffb3', dash='dot', width=1.5))
        styled_layout(fig_l, f"Huber Loss  (best epoch: {best})", height=290)
        st.plotly_chart(fig_l, use_container_width=True)

    with col_b:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=ep, y=tm, mode='lines', name='Train MAE',
                                   line=dict(color='#ffd060', width=2)))
        fig_m.add_trace(go.Scatter(x=ep, y=vm, mode='lines', name='Val MAE',
                                   line=dict(color='#ff6b35', width=2, dash='dash')))
        styled_layout(fig_m, "MAE (normalized returns)", height=290)
        st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("<div class='section-title'>Training summary</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs trained",   metrics['epochs_trained'])
    c2.metric("Best Val Loss",    f"{min(vl):.5f}")
    c3.metric("Final Train MAE",  f"{tm[-1]:.4f}")
    c4.metric("Final Val MAE",    f"{vm[-1]:.4f}")

    st.markdown("<div class='section-title'>Dataset split</div>", unsafe_allow_html=True)
    fig_p = go.Figure(go.Pie(
        labels=['Train', 'Validation', 'Test'],
        values=[metrics['train_size'], metrics['val_size'], metrics['test_size']],
        hole=0.6,
        marker=dict(colors=['#5e7bff', '#ffd060', '#ff6b35']),
        textinfo='label+percent'
    ))
    fig_p.update_layout(
        **DARK, height=280,
        margin=dict(t=30, b=20, l=20, r=20),
        font=dict(family='Syne', color='#e0e4ff'),
        legend=dict(bgcolor='rgba(0,0,0,0.4)')
    )
    st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════
with tab4:
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-title'>Model branches</div>", unsafe_allow_html=True)
        st.markdown("**LSTM Branch** — long-term temporal patterns")
        st.code(
            "Input (30, 19)\n"
            "→ BiLSTM(96) + LayerNorm + Dropout(0.25)\n"
            "→ BiLSTM(48) + LayerNorm + Dropout(0.20)\n"
            "→ output (96,)",
            language="text"
        )
        st.markdown("**CNN Branch** — short-term local patterns")
        st.code(
            "Input (30, 19)\n"
            "→ Conv1D(64, kernel=3, relu)\n"
            "→ Conv1D(32, kernel=3, relu)\n"
            "→ GlobalAvgPool1D\n"
            "→ output (32,)",
            language="text"
        )
        st.markdown("**Fusion Head**")
        st.code(
            "Concatenate → (128,)\n"
            "→ Dense(64, relu) + Dropout(0.15)\n"
            "→ Dense(32, relu)\n"
            "→ Dense(1)\n"
            "→ output: normalized return",
            language="text"
        )

    with col_r:
        st.markdown("<div class='section-title'>Hyperparameters</div>", unsafe_allow_html=True)
        params = {
            "Total Parameters": f"{metrics['total_params']:,}",
            "Window Size":      f"{WINDOW_SIZE} days",
            "Input Features":   str(N_FEATURES),
            "Optimizer":        "Adam  lr=2e-4",
            "Loss Function":    "Huber",
            "Scaler":           "MinMaxScaler(-1, 1)",
            "Batch Size":       "32",
            "Max Epochs":       "200",
            "Early Stopping":   "patience=20  monitor=val_loss",
            "LR Scheduler":     "ReduceOnPlateau ×0.4  patience=8",
            "Train/Val/Test":   "72% / 14% / 14%",
        }
        rows = "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in params.items()])
        st.markdown(f"<table class='info-table'>{rows}</table>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:1.5rem'>Features (19 total)</div>", unsafe_allow_html=True)
        feat_groups = {
            "Returns":    ["ret_1", "ret_3", "ret_5", "ret_10", "ret_20"],
            "MA Ratios":  ["ma5_ratio", "ma10_ratio", "ma20_ratio", "ma50_ratio"],
            "Volatility": ["vol_5", "vol_10", "vol_20"],
            "Technical":  ["rsi", "macd_norm", "macd_signal", "bb_pos"],
            "Candle/Vol": ["hl_pct", "oc_pct", "vol_ratio"],
        }
        tags = "".join([
            f"<span class='tag'>{f}</span>"
            for feats in feat_groups.values() for f in feats
        ])
        st.markdown(tags, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — DATASET INFO
# ══════════════════════════════════════════════════════════════
with tab5:
    di = data_info
    st.markdown("<div class='section-title'>General info</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        rows = f"""
        <tr><td>Total rows (original)</td><td>{di['raw_shape'][0]:,}</td></tr>
        <tr><td>Total columns</td><td>{di['raw_shape'][1]}</td></tr>
        <tr><td>Date range (original)</td><td>{di['raw_date_min']} → {di['raw_date_max']}</td></tr>
        <tr><td>Rows after 2020 filter</td><td>{di['filtered_rows']:,}</td></tr>
        <tr><td>Rows after feature eng.</td><td>{di['after_fe_rows']:,}</td></tr>
        <tr><td>Duplicate rows</td><td>{di['raw_dupes']}</td></tr>
        """
        st.markdown(f"<table class='info-table'>{rows}</table>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-title'>Null values</div>", unsafe_allow_html=True)
        null_rows = "".join([
            f"<tr><td>{col}</td><td>{'✅ 0' if v == 0 else f'⚠️ {v}'}</td></tr>"
            for col, v in di['raw_nulls'].items()
        ])
        st.markdown(f"<table class='info-table'>{null_rows}</table>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Outliers (IQR method, post-2020 data)</div>", unsafe_allow_html=True)
    out_cols = st.columns(len(di['outlier_info']))
    for i, (col, info) in enumerate(di['outlier_info'].items()):
        pct   = info['pct']
        bar_w = min(int(pct * 5), 100)
        out_cols[i].markdown(f"""
        <div class='stat'>
          <div class='s-label'>{col}</div>
          <div class='s-val'>{info['count']}</div>
          <div style='color:#4a5080; font-size:0.7rem'>{pct}% of rows</div>
          <div class='bar-outer'><div class='bar-inner' style='width:{bar_w}%'></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Descriptive statistics (post-2020)</div>", unsafe_allow_html=True)
    st.dataframe(di['describe'], use_container_width=True)

    st.markdown("<div class='section-title'>Price distribution</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df['Adj Close'], nbinsx=50,
                                        marker_color='#5e7bff', opacity=0.8))
        styled_layout(fig_hist, "Adj Close Distribution", height=280)
        fig_hist.update_xaxes(title='Price ($)')
        fig_hist.update_yaxes(title='Frequency')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df['Date'], y=df['Volume'] / 1e6, mode='lines',
            line=dict(color='#00ffb3', width=1.2), name='Volume (M)',
            fill='tozeroy', fillcolor='rgba(0,255,179,0.06)'
        ))
        styled_layout(fig_vol, "Volume Over Time", height=280)
        fig_vol.update_yaxes(title='Volume (M shares)')
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<div class='section-title'>Correlation matrix (features)</div>", unsafe_allow_html=True)
    corr = df[FEATURE_COLS].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, '#ff4466'], [0.5, '#0c0f1d'], [1, '#00ffb3']],
        zmid=0, text=corr.values, texttemplate="%{text}",
        textfont=dict(size=8),
        hovertemplate='%{x} · %{y}<br>r = %{z}<extra></extra>'
    ))
    styled_layout(fig_corr, "Feature Correlation Matrix", height=480)
    fig_corr.update_xaxes(tickfont=dict(size=8))
    fig_corr.update_yaxes(tickfont=dict(size=8))
    st.plotly_chart(fig_corr, use_container_width=True)