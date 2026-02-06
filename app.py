import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Opciones Black-Scholes")

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
        d1 = self.d1()
        d2 = self.d2()
        if self.type == "Call":
            price = (self.S * si.norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0))
        else:
            price = (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0.0, 1.0) - self.S * si.norm.cdf(-d1, 0.0, 1.0))
        return price

    def greeks(self):
        d1 = self.d1()
        d2 = self.d2()
        
        # Delta
        if self.type == "Call":
            delta = si.norm.cdf(d1, 0.0, 1.0)
        else:
            delta = si.norm.cdf(d1, 0.0, 1.0) - 1
            
        # Gamma (Igual para Call y Put)
        gamma = si.norm.pdf(d1, 0.0, 1.0) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Theta (Diario)
        aux1 = -(self.S * si.norm.pdf(d1, 0.0, 1.0) * self.sigma) / (2 * np.sqrt(self.T))
        if self.type == "Call":
            aux2 = self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(d2, 0.0, 1.0)
            theta_annual = aux1 - aux2
        else:
            aux2 = self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-d2, 0.0, 1.0)
            theta_annual = aux1 + aux2
            
        theta = theta_annual / 365 # Theta por día
        
        # Vega (Por 1% de cambio en Volatilidad)
        vega = (self.S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(self.T)) / 100
        
        # Rho (Por 1% de cambio en Tasa)
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
T_days = st.sidebar.slider("Días hasta Expiración", 1, 365, 30)
IV = st.sidebar.slider("Volatilidad Implícita (IV %)", 1.0, 200.0, 20.0)
r_percent = st.sidebar.number_input("Tasa Libre de Riesgo (%)", value=4.5, step=0.1)
contracts = st.sidebar.number_input("Cantidad de Contratos", value=1, min_value=1)
premium_paid = st.sidebar.number_input("Prima Pagada por Acción ($)", value=0.0, step=0.01, help="Si dejas 0, usamos el precio teórico actual.")

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

# Breakeven
breakeven = K + entry_price if option_type == "Call" else K - entry_price

# --- VISUALIZACIÓN PRINCIPAL ---
st.title(f"Simulador Educativo: Long {option_type}")
st.markdown("Analiza cómo las **Griegas**, el **Tiempo** y la **Volatilidad** afectan tu dinero.")

# Métricas Top
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Teórico", f"${theo_price:.2f}", help="Valor justo calculado por Black-Scholes")
col2.metric("Valor Intrínseco", f"${intrinsic:.2f}", help="Valor real si se ejerciera hoy")
col3.metric("Valor Extrínseco", f"${extrinsic:.2f}", help="Valor de tiempo y volatilidad")
col4.metric("Estado", moneyness, delta_color="off")

st.info(f"💰 **Costo Total:** ${total_cost:.2f} (Breakeven: ${breakeven:.2f})")

# --- GRIEGAS EXPLICADAS ---
st.subheader("🧩 Análisis de Griegas")
g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)

g_col1.metric("Delta", f"{greeks['Delta']:.3f}", help="Cuánto cambia el precio de la opción si la acción sube $1.")
g_col2.metric("Gamma", f"{greeks['Gamma']:.3f}", help="Aceleración: Cuánto cambia Delta si la acción se mueve.")
g_col3.metric("Theta (Diario)", f"{greeks['Theta']:.3f}", help="Cuánto valor pierde la opción CADA DÍA que pasa.")
g_col4.metric("Vega", f"{greeks['Vega']:.3f}", help="Cuánto cambia el precio si la Volatilidad sube 1%.")
g_col5.metric("Rho", f"{greeks['Rho']:.3f}", help="Sensibilidad a las tasas de interés.")

# Explicación dinámica
st.markdown("---")
st.write(f"""
**Interpretación para tu posición:**
* **Dirección:** Necesitas que la acción se mueva **{'arriba' if greeks['Delta'] > 0 else 'abajo'}** (Delta).
* **Tiempo:** Estás **perdiendo ${abs(greeks['Theta']*100*contracts):.2f} por día** solo por el paso del tiempo (Theta Decay).
* **Volatilidad:** Te beneficia que la volatilidad **{'suba' if greeks['Vega'] > 0 else 'baje'}** (Vega positivo en compra de opciones).
""")

# --- SIMULACIÓN GRÁFICA ---
st.subheader("📈 Escenarios y Proyecciones")

tab1, tab2, tab3 = st.tabs(["P&L al Vencimiento", "Impacto del Tiempo (Theta)", "Impacto de Volatilidad (Vega)"])

with tab1:
    # Generar rango de precios para el gráfico
    spot_range = np.linspace(S * 0.7, S * 1.3, 100)
    pnl_expiry = []
    
    for spot in spot_range:
        if option_type == "Call":
            val_expiry = max(0, spot - K)
        else:
            val_expiry = max(0, K - spot)
        pnl = (val_expiry - entry_price) * 100 * contracts
        pnl_expiry.append(pnl)

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=spot_range, y=pnl_expiry, mode='lines', name='P&L al Vencimiento', line=dict(color='green', width=3)))
    fig_pnl.add_vline(x=S, line_dash="dash", line_color="blue", annotation_text="Precio Actual")
    fig_pnl.add_vline(x=breakeven, line_dash="dot", line_color="red", annotation_text="Breakeven")
    fig_pnl.add_hline(y=0, line_color="white", opacity=0.3)
    
    fig_pnl.update_layout(title="Ganancia/Pérdida al Vencimiento", xaxis_title="Precio de la Acción", yaxis_title="P&L ($)")
    st.plotly_chart(fig_pnl, use_container_width=True)

with tab2:
    # Simular paso del tiempo manteniendo precio constante
    days_range = list(range(T_days, 0, -1))
    time_decay_prices = []
    
    for d in days_range:
        m_temp = OptionPricing(S, K, d, r_percent/100, IV/100, option_type)
        time_decay_prices.append(m_temp.price())
        
    fig_theta = go.Figure()
    fig_theta.add_trace(go.Scatter(x=days_range, y=time_decay_prices, fill='tozeroy', name='Valor de la Opción'))
    fig_theta.update_layout(title="Decadencia del Valor por Tiempo (Theta)", xaxis_title="Días restantes", yaxis_title="Precio Opción", xaxis_autorange="reversed")
    st.plotly_chart(fig_theta, use_container_width=True)
    st.caption("Nota como la curva se vuelve más pronunciada (pierdes valor más rápido) en los últimos 30 días.")

with tab3:
    # Simular cambios en volatilidad
    vol_range = np.linspace(10, 100, 50) # De 10% a 100% IV
    vega_prices = []
    
    for v in vol_range:
        m_temp = OptionPricing(S, K, T_days, r_percent/100, v/100, option_type)
        vega_prices.append(m_temp.price())
        
    fig_vega = go.Figure()
    fig_vega.add_trace(go.Scatter(x=vol_range, y=vega_prices, mode='lines', name='Sensibilidad a IV'))
    fig_vega.add_vline(x=IV, line_dash="dash", line_color="orange", annotation_text="IV Actual")
    fig_vega.update_layout(title="Precio de la Opción vs Volatilidad Implícita", xaxis_title="Volatilidad (%)", yaxis_title="Precio Opción")
    st.plotly_chart(fig_vega, use_container_width=True)
