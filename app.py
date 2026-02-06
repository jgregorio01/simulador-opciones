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
