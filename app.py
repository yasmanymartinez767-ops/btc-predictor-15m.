import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="BTC Predictor 15M", page_icon="₿", layout="centered")
st.title("₿ BTC Predictor 15M")
st.caption("Señales educativas • Sin operaciones automáticas • Datos públicos de Coinbase")

API = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

@st.cache_data(ttl=30)
def cargar_velas():
    # Coinbase Exchange devuelve velas de 60 segundos, más reciente primero.
    response = requests.get(
        API,
        params={"granularity": 60},
        headers={"User-Agent": "BTC-Predictor-15M/1.0", "Accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or len(raw) < 60:
        raise ValueError("Coinbase no devolvió suficientes velas para calcular indicadores.")
    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
    for col in ["timestamp", "low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # Se excluye la vela actual (potencialmente incompleta).
    now = time.time()
    df = df[df["timestamp"] + 60 <= now].copy()
    if len(df) < 60:
        raise ValueError("No hay suficientes velas completas para analizar.")
    # Detecta retrasos o huecos recientes: evita señales con datos incompletos.
    last_end = float(df["timestamp"].iloc[-1]) + 60
    if now - last_end > 180:
        raise ValueError("Datos desactualizados: última vela cerrada hace más de 3 minutos.")
    if (df["timestamp"].diff().tail(35).dropna() > 60).any():
        raise ValueError("Hay huecos en las últimas velas de Coinbase. Espera y actualiza.")
    return df


def analizar(df):
    close = df["close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    returns = close.pct_change()
    vol15 = float(returns.tail(30).std() * np.sqrt(15) * 100)
    volume_base = df["volume"].iloc[-31:-1].mean()
    rel_volume = float(df["volume"].iloc[-1] / volume_base) if volume_base > 0 else 0.0
    momentum15 = float((close.iloc[-1] / close.iloc[-16] - 1) * 100)
    votes = [
        bool(ema9.iloc[-1] > ema21.iloc[-1]),
        bool(close.iloc[-1] > ema9.iloc[-1]),
        bool(rsi.iloc[-1] > 50),
        bool(momentum15 > 0),
        bool(close.iloc[-1] > close.iloc[-2]),
    ]
    score = sum(votes)
    if not np.isfinite(vol15) or vol15 > 1.2 or rel_volume < 0.5 or 2 <= score <= 3:
        signal = "⚪ ESPERAR / SIN SEÑAL CLARA"
    elif score >= 4:
        signal = "🟢 SESGO ALCISTA"
    else:
        signal = "🔴 SESGO BAJISTA"
    return signal, score, float(rsi.iloc[-1]), vol15, rel_volume, momentum15


if st.button("🔄 Actualizar datos"):
    cargar_velas.clear()

try:
    df = cargar_velas()
    signal, score, rsi, volatility, rel_volume, momentum = analizar(df)
    st.metric("BTC/USD • última vela cerrada", f"${df['close'].iloc[-1]:,.2f}")
    st.caption(f"Última vela cerrada: {df['fecha'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    st.subheader(signal)
    st.write("Horizonte orientativo: próximos 15 minutos. No es una probabilidad calibrada.")
    c1, c2 = st.columns(2)
    c1.metric("Votos alcistas", f"{score}/5")
    c2.metric("RSI (14)", f"{rsi:.1f}")
    c1.metric("Volatilidad 15m estimada", f"{volatility:.2f}%")
    c2.metric("Volumen relativo", f"{rel_volume:.2f}x")
    st.metric("Movimiento de los últimos 15 minutos", f"{momentum:+.2f}%")
    st.line_chart(df.set_index("fecha")["close"].tail(120))
    st.info("Coinbase BTC/USD y la referencia de liquidación de Kalshi pueden diferir. Comprueba las reglas y el precio de referencia de cada contrato.")
    st.warning("Señal experimental, no validada con backtesting fuera de muestra. No operar basándose únicamente en esta pantalla.")
except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
    st.error("No se pudieron obtener datos válidos y recientes. No operar.")
    st.code(str(exc))
