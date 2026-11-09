import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='BTC Predictor 15M', page_icon='₿', layout='centered')
st.title('₿ BTC Predictor 15M')
st.caption('Señales educativas • Sin operaciones automáticas • Datos públicos de Binance')

@st.cache_data(ttl=30)
def candles():
    url='https://api.binance.com/api/v3/klines'
    r=requests.get(url,params={'symbol':'BTCUSDT','interval':'1m','limit':500},timeout=12)
    r.raise_for_status()
    rows=r.json()
    df=pd.DataFrame(rows,columns=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore'])
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c])
    df['date']=pd.to_datetime(df.open_time,unit='ms',utc=True)
    return df.iloc[:-1].copy() # Exclude incomplete candle

def analyze(df):
    close=df.close
    ema9=close.ewm(span=9,adjust=False).mean()
    ema21=close.ewm(span=21,adjust=False).mean()
    delta=close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rsi=100-100/(1+gain/(loss.replace(0,np.nan)))
    returns=close.pct_change()
    vol=returns.tail(30).std()*np.sqrt(15)*100
    volume_ratio=df.volume.tail(15).mean()/max(df.volume.iloc[-75:-15].mean(),1e-10)
    momentum=(close.iloc[-1]/close.iloc[-16]-1)*100
    votes=[int(ema9.iloc[-1]>ema21.iloc[-1]),int(close.iloc[-1]>ema21.iloc[-1]),int(momentum>0),int(rsi.iloc[-1]>50),int(close.iloc[-1]>close.iloc[-6])]
    score=sum(votes)
    if vol>1.2 or volume_ratio<0.55 or (45<=rsi.iloc[-1]<=55) or score in (2,3):
        signal='⚪ NO OPERAR'
    elif score>=4:
        signal='🟢 SESGO ALCISTA'
    else:
        signal='🔴 SESGO BAJISTA'
    return signal,score,float(rsi.iloc[-1]),float(vol),float(volume_ratio),float(momentum)

try:
    df=candles()
    signal,score,rsi,vol,vr,mom=analyze(df)
    st.metric('BTC/USDT (última vela cerrada)',f'${df.close.iloc[-1]:,.2f}')
    st.caption('Última vela cerrada: '+df.date.iloc[-1].strftime('%Y-%m-%d %H:%M UTC'))
    st.subheader(signal)
    st.write('Horizonte orientativo: próximos 15 minutos. No es una probabilidad calibrada ni una recomendación de apuesta.')
    c1,c2=st.columns(2)
    c1.metric('Votos alcistas',f'{score}/5')
    c2.metric('RSI (14)',f'{rsi:.1f}')
    c1.metric('Volatilidad 15m estimada',f'{vol:.2f}%')
    c2.metric('Volumen relativo',f'{vr:.2f}x')
    st.metric('Movimiento previo de 15m',f'{mom:+.2f}%')
    st.line_chart(df.set_index('date').close.tail(120))
    st.info('Kalshi puede usar un índice y una hora de liquidación diferentes de BTC/USDT. Compara las reglas exactas del contrato antes de actuar.')
    st.warning('La señal NO está validada estadísticamente. Primero haz pruebas fuera de muestra y registra al menos varias semanas de predicciones con su resultado.')
    if st.button('Actualizar datos'):
        st.cache_data.clear()
        st.rerun()
except Exception as exc:
    st.error('No se pudieron obtener datos actuales. No operar. Verifica la conexión o disponibilidad de Binance en tu región.')
    st.caption(str(exc))
