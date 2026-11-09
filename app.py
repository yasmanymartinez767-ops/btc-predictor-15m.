import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="BTC Predictor 15M", page_icon="₿", layout="centered")
st.title("₿ BTC Predictor 15M")
st.caption("Señales educativas · Coinbase BTC/USD · Sin operaciones automáticas")

API = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
HORIZON = 15
MIN_HISTORY = 60

@st.cache_data(ttl=30)
def cargar_velas():
    response = requests.get(API, params={"granularity": 60}, headers={
        "User-Agent": "BTC-Predictor-15M/1.1", "Accept": "application/json"
    }, timeout=15)
    response.raise_for_status()
    raw = response.json()
    if not isinstance(raw, list) or len(raw) < MIN_HISTORY + HORIZON:
        raise ValueError("Coinbase no devolvió suficientes velas para el análisis.")
    df = pd.DataFrame(raw, columns=["timestamp", "low", "high", "open", "close", "volume"])
    for col in ["timestamp", "low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now = time.time()
    df = df[df["timestamp"] + 60 <= now].copy().reset_index(drop=True)
    if len(df) < MIN_HISTORY + HORIZON:
        raise ValueError("No hay suficientes velas completas.")
    if now - (float(df["timestamp"].iloc[-1]) + 60) > 180:
        raise ValueError("Datos desactualizados: la última vela cerró hace más de 3 minutos.")
    # Coinbase puede omitir minutos sin operaciones. No se deben comparar ventanas con huecos.
    if (df["timestamp"].diff().tail(60).dropna() != 60).any():
        raise ValueError("Hay huecos en las últimas 60 velas; no se emite señal.")
    df["fecha"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def indicadores(df):
    close = df["close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    returns = close.pct_change()
    vol = returns.rolling(30).std() * np.sqrt(15) * 100
    base = df["volume"].shift(1).rolling(30).mean()
    rel_volume = df["volume"] / base.replace(0, np.nan)
    momentum = (close / close.shift(15) - 1) * 100
    votes = ((ema9 > ema21).astype(int) + (close > ema9).astype(int)
             + (rsi > 50).astype(int) + (momentum > 0).astype(int)
             + (close > close.shift(1)).astype(int))
    result = df.copy()
    result["rsi"] = rsi
    result["vol15"] = vol
    result["vol_rel"] = rel_volume
    result["momentum15"] = momentum
    result["votes"] = votes
    result["senal"] = np.where(
        vol.isna() | (vol > 1.2) | (rel_volume < 0.5) | votes.between(2, 3),
        "ESPERAR", np.where(votes >= 4, "ALCISTA", "BAJISTA")
    )
    return result


def evaluar_historico(result):
    # Muestras no superpuestas cada 15 minutos, alineadas al reloj UTC.
    # La predicción usa solo información disponible al cierre de la vela inicial.
    # Se exige que las 15 velas siguientes sean consecutivas.
    records = []
    for i in range(MIN_HISTORY - 1, len(result) - HORIZON):
        start = result.iloc[i]
        end = result.iloc[i + HORIZON]
        if int(start["timestamp"] + 60) % (HORIZON * 60) != 0:
            continue
        window = result.iloc[i:i + HORIZON + 1]
        if not (window["timestamp"].diff().dropna() == 60).all():
            continue
        signal = start["senal"]
        if signal == "ESPERAR":
            continue
        change = float(end["close"] / start["close"] - 1)
        correct = (signal == "ALCISTA" and change > 0) or (signal == "BAJISTA" and change < 0)
        records.append({
            "Fecha UTC": start["fecha"], "Señal": signal,
            "Precio inicial": float(start["close"]), "Precio +15m": float(end["close"]),
            "Cambio %": round(change * 100, 3),
            "Resultado": "Acierto" if correct else "Fallo",
        })
    return pd.DataFrame(records)

if st.button("🔄 Actualizar datos"):
    cargar_velas.clear()

try:
    data = indicadores(cargar_velas())
    last = data.iloc[-1]
    st.metric("BTC/USD · última vela cerrada", f"${last['close']:,.2f}")
    st.caption(f"Última vela cerrada: {(last['fecha'] + pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M UTC')}")
    signal = last["senal"]
    st.subheader({"ALCISTA": "🟢 SESGO ALCISTA", "BAJISTA": "🔴 SESGO BAJISTA", "ESPERAR": "⚪ ESPERAR"}[signal])
    st.caption("Horizonte orientativo: 15 minutos. No es una probabilidad calibrada.")
    a, b = st.columns(2)
    a.metric("Votos alcistas", f"{int(last['votes'])}/5")
    b.metric("RSI (14)", f"{last['rsi']:.1f}")
    a.metric("Volatilidad estimada 15m", f"{last['vol15']:.2f}%")
    b.metric("Volumen relativo", f"{last['vol_rel']:.2f}x")
    st.line_chart(data.set_index("fecha")["close"].tail(120))

    st.divider()
    st.subheader("📊 Aciertos históricos (prueba retrospectiva)")
    history = evaluar_historico(data)
    if history.empty:
        st.info("No hay suficientes señales históricas completas para evaluar.")
    else:
        wins = int((history["Resultado"] == "Acierto").sum())
        total = len(history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Aciertos", f"{wins}/{total}")
        c2.metric("Fallos", total - wins)
        c3.metric("% aciertos", f"{100 * wins / total:.1f}%")
        st.dataframe(history.sort_values("Fecha UTC", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Descargar evaluación CSV", history.to_csv(index=False).encode("utf-8"),
                           file_name="btc_evaluacion_historica.csv", mime="text/csv")
        if total < 30:
            st.warning("Muestra muy pequeña: este porcentaje NO demuestra capacidad predictiva.")
    st.info("Esta evaluación se recalcula con las últimas velas disponibles (aprox. hasta 300 minutos); no almacena predicciones futuras ni resultados permanentes. No incluye comisiones, spreads ni precios de Kalshi.")
    st.warning("No usar como recomendación de inversión. Coinbase BTC/USD puede diferir del índice y reglas de liquidación de Kalshi. Validar fuera de muestra antes de operar.")
except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
    st.error("No se pudieron obtener datos válidos y recientes. No operar.")
    st.code(str(exc))
