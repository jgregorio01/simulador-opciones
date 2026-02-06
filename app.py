import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Opciones Pro")

# --- LÓGICA MATEMÁTICA (BLACK-SCHOLES & GRIEGAS) ---
class OptionPricing:
    def __init__(self, S, K, T, r, sigma, option_type="Call"):
        self.S = S        # Precio Subyacente
        self.K = K        # Strike Price
        self.T = T / 365  # Tiempo en años
        self.r = r        # Tasa libre de riesgo
        self.sigma = sigma # Volatilidad (decimal)
        self.type = option_type
    
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def price(self):
        # Evitar división por cero si T es muy pequeño
        if self.T <= 1e-5: return max(0, self.S - self.K) if self.type == "Call" else max(0, self.K - self.S)
        
        d1 = self.d1()
        d2 = self.d2()
        if self.type == "Call":
            price = (self.S * si.norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0.0, 1.0) - self.S * si.norm.cdf(-d1, 0.0, 1.0))
        return price

    def greeks(self):
        if self.T <= 1e-5: return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}
        
        d1 = self.d1()
        d2 = self.d2()
        
        # Delta
        if self.type == "Call":
            delta = si.norm.cdf(d1, 0.0, 1.0)
        else:
            delta = si.norm.cdf(d1, 0.0, 1.0) - 1
            
        # Gamma
        gamma = si.norm.pdf(d1, 0.0, 1.0) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Theta (Diario)
        aux1 = -(self.S * si.norm.pdf(d1, 0.0, 1.0) * self.sigma) / (2 * np.sqrt(self.T))
        if self.type == "Call":
            aux2 = self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0)
            theta_annual = aux1 - aux2
        else:
            aux2 = self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0.0, 1.0)
            theta_annual = aux1 + aux2
            
        theta = theta_annual / 365 
        
        # Vega
        vega = (self.S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(self.T)) / 100
        
        # Rho
        if self.type == "Call":
            rho = (self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0)) / 100
        else:
            rho = (-self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0.0, 1.0)) / 100
            
        return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# --- INTERFAZ DE USUARIO (SIDEBAR) ---
st.sidebar.header("📊 Parámetros del Contrato")

option_type = st.sidebar.selectbox("Tipo de Opción", ["Call", "Put"])
S = st.sidebar.number_input("Precio Actual Subyacente ($)", value=100.0, step=0.5)
K = st.sidebar.number_input("Strike Price ($)", value=105.0, step=0.5)

# --- NUEVO: SELECTOR DE FECHA ---
today = date.today()
default_date = today + timedelta(days=30)
max_date = date(2030, 12, 31)

expiry_date = st.sidebar.date_input(
    "Fecha de Expiración",
    value=default_date,
    min_value=today + timedelta(days=1), # Mínimo mañana (para evitar error matemático)
    max_value=max_date
)

# Calcular días automáticamente
T_days = (expiry_date - today).days
st.sidebar.caption(f"📅 Días hasta vencimiento: **{T_days} días**")

IV = st.sidebar.slider("Volatilidad Implícita (IV %)", 1.0, 200.0, 20.0)
r_percent = st.sidebar.number_input("Tasa Libre de Riesgo (%)", value=4.5, step=0.1)
contracts = st.sidebar.number_input("Cantidad de Contratos", value=1, min_value=1)
premium_paid = st.sidebar.number_input("Prima Pagada por Acción ($)", value=0.0, step=0.01, help="Dejar en 0 para usar precio teórico.")

# --- CÁLCULOS ---
model = OptionPricing(S, K, T_days, r_percent/100, IV/100, option_type)
theo_price = model.price()
greeks = model.greeks()

# Costo real de la operación
entry_price = premium_paid if premium_paid > 0 else theo_price
total_cost = entry_price * 100 * contracts

# Moneyness
intrinsic = max(0, S - K) if option_type == "Call" else max(0, K - S)
extrinsic = theo_price - intrinsic
moneyness = "ITM (In The Money)" if intrinsic > 0 else ("ATM (At The Money)" if S == K else "OTM (Out of The Money)")
breakeven = K + entry_price if option_type == "Call" else K - entry_price

# --- VISUALIZACIÓN PRINCIPAL ---
st.title(f"Simulador: Long {option_type} (Vence: {expiry_date})")

# Métricas Top
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Teórico", f"${theo_price:.2f}")
col2.metric("Valor Intrínseco", f"${intrinsic:.2f}")
col3.metric("Valor Extrínseco", f"${extrinsic:.2f}")
col4.metric("Estado", moneyness, delta_color="off")

st.info(f"💰 **Costo Total:** ${total_cost:.2f} (Breakeven: ${breakeven:.2f})")

# --- GRIEGAS ---
st.subheader("🧩 Análisis de Griegas")
g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
g_col1.metric("Delta", f"{greeks['Delta']:.3f}")
g_col2.metric("Gamma", f"{greeks['Gamma']:.3f}")
g_col3.metric("Theta", f"{greeks['Theta']:.3f}")
g_col4.metric("Vega", f"{greeks['Vega']:.3f}")
g_col5.metric("Rho", f"{greeks['Rho']:.3f}")

# --- GRÁFICOS ---
st.subheader("📈 Proyecciones")
tab1, tab2, tab3 = st.tabs(["P&L al Vencimiento", "Decadencia Theta", "Sensibilidad Vega"])

with tab1:
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    pnl_expiry = []
    for spot in spot_range:
        val_expiry = max(0, spot - K) if option_type == "Call" else max(0, K - spot)
        pnl_expiry.append((val_expiry - entry_price) * 100 * contracts)

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=spot_range, y=pnl_expiry, mode='lines', name='P&L', line=dict(color='green', width=3)))
    fig_pnl.add_vline(x=S, line_dash="dash", line_color="blue", annotation_text="Precio Actual")
    fig_pnl.add_vline(x=breakeven, line_dash="dot", line_color="red", annotation_text="Breakeven")
    fig_pnl.add_hline(y=0, line_color="white", opacity=0.3)
    fig_pnl.update_layout(title="Ganancia/Pérdida al Vencimiento", xaxis_title="Precio Acción", yaxis_title="P&L ($)")
    st.plotly_chart(fig_pnl, use_container_width=True)

with tab2:
    if T_days > 1:
        days_range = list(range(T_days, 0, -1))
        theta_prices = []
        for d in days_range:
            m_temp = OptionPricing(S, K, d, r_percent/100, IV/100, option_type)
            theta_prices.append(m_temp.price())
            
        fig_theta = go.Figure()
        fig_theta.add_trace(go.Scatter(x=days_range, y=theta_prices, fill='tozeroy', name='Valor Opción'))
        fig_theta.update_layout(title="Decadencia por Tiempo", xaxis_title="Días Restantes", yaxis_title="Precio", xaxis_autorange="reversed")
        st.plotly_chart(fig_theta, use_container_width=True)
    else:
        st.warning("La opción vence mañana, no hay gráfica de tiempo suficiente.")

with tab3:
    vol_range = np.linspace(1, 150, 50)
    vega_prices = [OptionPricing(S, K, T_days, r_percent/100, v/100, option_type).price() for v in vol_range]
    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(x=vol_range, y=vega_prices, name='Sensibilidad IV'))
    fig_vega.add_vline(x=IV, line_dash="dash", line_color="orange", annotation_text="IV Actual")
    fig_vega.update_layout(title="Impacto de Volatilidad", xaxis_title="Volatilidad (%)", yaxis_title="Precio")
    st.plotly_chart(fig_vega, use_container_width=True)
