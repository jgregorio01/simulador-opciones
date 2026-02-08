import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Opciones Pro v4")

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

# --- SIDEBAR ---
st.sidebar.header("🔍 Datos del Mercado")

ticker = st.sidebar.text_input("Ticker (Ej: AAPL, TSLA)", value="").upper()
current_price = 100.0

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

S = st.sidebar.number_input("Precio Subyacente ($)", value=float(current_price), step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuración")

option_type = st.sidebar.selectbox("Tipo de Opción", ["Call", "Put"])
K = st.sidebar.number_input("Strike Price ($)", value=S * 1.05, step=0.5)

today = date.today()
expiry_date = st.sidebar.date_input("Fecha de Expiración", value=today + timedelta(days=30), min_value=today + timedelta(days=1), max_value=date(2030, 12, 31))
T_days = (expiry_date - today).days

IV = st.sidebar.number_input("Volatilidad Implícita (IV %)", value=20.0, min_value=1.0, step=0.5)
r_percent = st.sidebar.number_input("Tasa Libre de Riesgo (%)", value=4.5, step=0.1)
contracts = st.sidebar.number_input("Cantidad de Contratos", value=1, min_value=1)
premium_paid = st.sidebar.number_input("Prima Pagada por Acción ($)", value=0.0, step=0.01)

# --- CÁLCULOS BASE ---
model = OptionPricing(S, K, T_days, r_percent/100, IV/100, option_type)
theo_price = model.price()
greeks = model.greeks()

entry_price = premium_paid if premium_paid > 0 else theo_price
total_cost = entry_price * 100 * contracts
breakeven = K + entry_price if option_type == "Call" else K - entry_price

# --- TÍTULO ---
st.title(f"Simulador: {ticker if ticker else 'Personalizado'} {option_type} @ ${K}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Opción", f"${theo_price:.2f}")
col2.metric("Breakeven", f"${breakeven:.2f}")
col3.metric("Días al Vencimiento", f"{T_days}")
col4.metric("Inversión Total", f"${total_cost:.2f}")

# --- CALCULADORA TARGET ---
st.markdown("---")
st.subheader("🎯 Calculadora de Retorno")
sc_col1, sc_col2 = st.columns([1, 2])
with sc_col1:
    target_price = st.number_input("Price Target ($)", value=breakeven * 1.05, step=0.5)

if option_type == "Call":
    value_at_target = max(0, target_price - K)
else:
    value_at_target = max(0, K - target_price)

profit_at_target = (value_at_target - entry_price) * 100 * contracts
roi = (profit_at_target / total_cost) * 100 if total_cost > 0 else 0
premium_pct_spot = (entry_price / S) * 100

with sc_col2:
    st.write("#### Resultados en el Target (al Vencimiento):")
    df_results = pd.DataFrame({
        "Métrica": ["Inversión", "% Prima/Acción", "Ganancia Neta", "% Retorno (ROI)"],
        "Valor": [f"${total_cost:.2f}", f"{premium_pct_spot:.2f}%", f"${profit_at_target:.2f}", f"{roi:.2f}%"]
    })
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# --- SECCIÓN DE GRIEGAS (AHORA ARRIBA Y CON DEFINICIONES) ---
st.markdown("---")
st.subheader("🧩 Panel de Griegas (Hoy)")

# Definiciones para los tooltips
help_delta = "Delta: Cuánto cambia el precio de la opción si la acción sube $1. También representa la probabilidad aproximada de que la opción expire 'In The Money'."
help_gamma = "Gamma: La aceleración del Delta. Mide cuánto cambiará el Delta si la acción se mueve $1 más."
help_theta = "Theta (Time Decay): Cuánto dinero pierde tu opción CADA DÍA que pasa, asumiendo que el precio de la acción no cambia."
help_vega = "Vega: Cuánto cambia el precio de la opción si la Volatilidad Implícita sube un 1%."
help_rho = "Rho: Cuánto cambia el precio si la tasa de interés libre de riesgo cambia un 1%."

g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
g_col1.metric("Delta", f"{greeks['Delta']:.3f}", help=help_delta)
g_col2.metric("Gamma", f"{greeks['Gamma']:.3f}", help=help_gamma)
g_col3.metric("Theta", f"{greeks['Theta']:.3f}", help=help_theta)
g_col4.metric("Vega", f"{greeks['Vega']:.3f}", help=help_vega)
g_col5.metric("Rho", f"{greeks['Rho']:.3f}", help=help_rho)

# --- MÁQUINA DEL TIEMPO (AHORA ABAJO) ---
st.markdown("---")
st.subheader("⏳ Máquina del Tiempo: Efecto Theta")
st.markdown("Mueve el deslizador para **avanzar días en el futuro** y ver cómo se 'desinfla' tu contrato si el precio no se mueve.")

# Deslizador de tiempo
days_passed = st.slider("Simular paso de días (Días transcurridos desde hoy)", 0, T_days - 1, 0)
days_remaining_sim = T_days - days_passed

# Cálculo del escenario Futuro (Manteniendo S e IV constantes)
model_future = OptionPricing(S, K, days_remaining_sim, r_percent/100, IV/100, option_type)
future_price = model_future.price()
future_greeks = model_future.greeks()
future_pnl = (future_price - entry_price) * 100 * contracts

# Métricas comparativas
t_col1, t_col2, t_col3, t_col4 = st.columns(4)
t_col1.metric("Fecha Simulada", f"En {days_passed} días", help=f"Quedarán {days_remaining_sim} días para vencer.")
t_col2.metric("Valor del Contrato", f"${future_price:.2f}", delta=f"{future_price - theo_price:.2f}")
t_col3.metric("P&L Latente", f"${future_pnl:.2f}", delta_color="normal" if future_pnl > 0 else "inverse")
t_col4.metric("Nuevo Theta", f"{future_greeks['Theta']:.3f}", help="El Theta suele aumentar (se vuelve más negativo) al acercarse el final.")

# --- GRÁFICA COMPARATIVA ---
spot_range = np.linspace(S * 0.7, S * 1.3, 100)

pnl_today = []
pnl_future_curve = []

for spot in spot_range:
    p_today = OptionPricing(spot, K, T_days, r_percent/100, IV/100, option_type).price()
    pnl_today.append((p_today - entry_price) * 100 * contracts)
    
    p_future = OptionPricing(spot, K, days_remaining_sim, r_percent/100, IV/100, option_type).price()
    pnl_future_curve.append((p_future - entry_price) * 100 * contracts)

fig_time = go.Figure()

fig_time.add_trace(go.Scatter(
    x=spot_range, y=pnl_today, 
    mode='lines', name='Curva P&L HOY', 
    line=dict(color='green', width=2, dash='dot')
))

fig_time.add_trace(go.Scatter(
    x=spot_range, y=pnl_future_curve, 
    mode='lines', name=f'Curva P&L en {days_passed} días', 
    line=dict(color='orange', width=4)
))

fig_time.add_vline(x=S, line_dash="dash", line_color="blue", annotation_text="Precio Actual")
fig_time.add_vline(x=breakeven, line_dash="dot", line_color="red", annotation_text="Breakeven")
fig_time.add_hline(y=0, line_color="white", opacity=0.3)

fig_time.update_layout(
    title=f"Impacto del Tiempo: Hoy vs Futuro (Día {days_passed})",
    xaxis_title="Precio de la Acción",
    yaxis_title="Ganancia / Pérdida ($)",
    hovermode="x unified"
)

st.plotly_chart(fig_time, use_container_width=True)
