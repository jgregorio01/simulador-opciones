import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Opciones Pro v2")

# --- LÓGICA MATEMÁTICA (BLACK-SCHOLES) ---
class OptionPricing:
    def __init__(self, S, K, T, r, sigma, option_type="Call"):
        self.S = S
        self.K = K
        self.T = T / 365
        self.r = r
        self.sigma = sigma
        self.type = option_type
    
    def price(self):
        if self.T <= 1e-5: return max(0, self.S - self.K) if self.type == "Call" else max(0, self.K - self.S)
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        if self.type == "Call":
            price = (self.S * si.norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0.0, 1.0) - self.S * si.norm.cdf(-d1, 0.0, 1.0))
        return price

    def greeks(self):
        if self.T <= 1e-5: return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        
        if self.type == "Call":
            delta = si.norm.cdf(d1)
            rho = (self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(d2)) / 100
            theta = (-(self.S * si.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2)) / 365
        else:
            delta = si.norm.cdf(d1) - 1
            rho = (-self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-d2)) / 100
            theta = (-(self.S * si.norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2)) / 365

        gamma = si.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = (self.S * si.norm.pdf(d1) * np.sqrt(self.T)) / 100
        
        return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# --- SIDEBAR: DATOS DEL MERCADO ---
st.sidebar.header("🔍 Datos del Mercado")

# 1. Buscador de Ticker
ticker = st.sidebar.text_input("Ticker (Ej: AAPL, TSLA, SPY)", value="").upper()
current_price = 100.0 # Precio por defecto

if ticker:
    try:
        stock_data = yf.Ticker(ticker)
        history = stock_data.history(period="1d")
        if not history.empty:
            fetched_price = history['Close'].iloc[-1]
            st.sidebar.success(f"Precio actual de {ticker}: ${fetched_price:.2f}")
            current_price = fetched_price
        else:
            st.sidebar.error("Ticker no encontrado.")
    except:
        st.sidebar.error("Error al buscar el ticker.")

# Input de Precio (se actualiza si encontramos el ticker, pero el usuario puede editarlo)
S = st.sidebar.number_input("Precio Subyacente ($)", value=float(current_price), step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración del Contrato")

option_type = st.sidebar.selectbox("Tipo de Opción", ["Call", "Put"])
K = st.sidebar.number_input("Strike Price ($)", value=S * 1.05, step=0.5)

# Fecha
today = date.today()
expiry_date = st.sidebar.date_input("Fecha de Expiración", value=today + timedelta(days=30), min_value=today + timedelta(days=1), max_value=date(2030, 12, 31))
T_days = (expiry_date - today).days

# 2. CAMBIO SOLICITADO: IV manual (Number Input en vez de Slider)
IV = st.sidebar.number_input("Volatilidad Implícita (IV %)", value=20.0, min_value=1.0, step=0.5, help="Escribe el valor manual.")

r_percent = st.sidebar.number_input("Tasa Libre de Riesgo (%)", value=4.5, step=0.1)
contracts = st.sidebar.number_input("Cantidad de Contratos", value=1, min_value=1)
premium_paid = st.sidebar.number_input("Prima Pagada por Acción ($)", value=0.0, step=0.01)

# --- CÁLCULOS ---
model = OptionPricing(S, K, T_days, r_percent/100, IV/100, option_type)
theo_price = model.price()
greeks = model.greeks()

entry_price = premium_paid if premium_paid > 0 else theo_price
total_cost = entry_price * 100 * contracts
breakeven = K + entry_price if option_type == "Call" else K - entry_price

# --- VISUALIZACIÓN ---
st.title(f"Simulador: {ticker if ticker else 'Personalizado'} {option_type} @ ${K}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Opción (Teórico)", f"${theo_price:.2f}")
col2.metric("Breakeven", f"${breakeven:.2f}")
col3.metric("Días al Vencimiento", f"{T_days}")
col4.metric("Inversión Total", f"${total_cost:.2f}")

# --- SECCIÓN DE ESCENARIOS (NUEVO) ---
st.markdown("---")
st.subheader("🎯 Calculadora de Retorno Objetivo")

# Inputs para el objetivo
sc_col1, sc_col2 = st.columns([1, 2])
with sc_col1:
    target_price = st.number_input("Price Target de la Acción ($)", value=breakeven * 1.05, step=0.5)

# Cálculos del escenario al vencimiento
if option_type == "Call":
    value_at_target = max(0, target_price - K)
else:
    value_at_target = max(0, K - target_price)

profit_at_target = (value_at_target - entry_price) * 100 * contracts
roi = (profit_at_target / total_cost) * 100 if total_cost > 0 else 0
premium_pct_spot = (entry_price / S) * 100 # % Pagado sobre la prima (relativo al precio acción)

# Tabla de Resultados
with sc_col2:
    st.write("#### Resultados al Vencimiento si llega al Target:")
    
    # Creamos un DataFrame para mostrarlo como tabla bonita
    df_results = pd.DataFrame({
        "Métrica": [
            "Precio Objetivo (Target)", 
            "Inversión Total Realizada", 
            "% Prima sobre Precio Acción", 
            "Valor de Salida (Venta)",
            "Ganancia Neta Estimada", 
            "% Retorno (ROI)"
        ],
        "Valor": [
            f"${target_price:.2f}",
            f"${total_cost:.2f}",
            f"{premium_pct_spot:.2f}%",
            f"${value_at_target * 100 * contracts:.2f}",
            f"${profit_at_target:.2f}",
            f"{roi:.2f}%"
        ]
    })
    
    # Estilizar la tabla: Si ROI es positivo verde, si negativo rojo
    def color_roi(val):
        color = 'green' if profit_at_target > 0 else 'red'
        return f'color: {color}; font-weight: bold'

    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    if profit_at_target > 0:
        st.success(f"🚀 ¡Potencial de ganancia del {roi:.1f}%!")
    else:
        st.error(f"⚠️ Pérdida estimada del {roi:.1f}% si expira en ese precio.")

# --- GRIEGAS Y GRÁFICOS ---
st.markdown("---")
st.subheader("🧩 Griegas y Gráficos")
g_col1, g_col2, g_col3, g_col4 = st.columns(4)
g_col1.metric("Delta", f"{greeks['Delta']:.3f}")
g_col2.metric("Theta", f"{greeks['Theta']:.3f}")
g_col3.metric("Vega", f"{greeks['Vega']:.3f}")
g_col4.metric("Gamma", f"{greeks['Gamma']:.3f}")

tab1, tab2 = st.tabs(["Curva de P&L", "Sensibilidad IV"])

with tab1:
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    pnl_expiry = []
    for spot in spot_range:
        val = max(0, spot - K) if option_type == "Call" else max(0, K - spot)
        pnl_expiry.append((val - entry_price) * 100 * contracts)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=pnl_expiry, name='P&L Expiración', line=dict(color='green', width=3)))
    fig.add_vline(x=target_price, line_dash="dash", line_color="purple", annotation_text="Target")
    fig.add_vline(x=breakeven, line_dash="dot", line_color="red", annotation_text="Breakeven")
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(title="Ganancia/Pérdida al Vencimiento", xaxis_title="Precio Acción", yaxis_title="P&L ($)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    vol_range = np.linspace(1, 200, 50)
    vega_prices = [OptionPricing(S, K, T_days, r_percent/100, v/100, option_type).price() for v in vol_range]
    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=vol_range, y=vega_prices, name='Precio vs IV'))
    fig_v.add_vline(x=IV, line_dash="dash", line_color="orange", annotation_text="IV Actual")
    st.plotly_chart(fig_v, use_container_width=True)
