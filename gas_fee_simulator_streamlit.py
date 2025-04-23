
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

st.set_page_config(page_title="EIP-1559 Fee Simulator", layout="wide")
st.title("USDC-Based Avalanche Subnet • EIP-1559 Fee Simulator")

with st.sidebar:
    st.header("Simulation Parameters")

    # timeline
    num_blocks = st.number_input("Blocks to simulate", 10, 50000, 2000, step=100)
    block_time = st.number_input("Block interval [s]", 0.1, 10.0, 1.0, step=0.1)

    # demand
    st.subheader("Demand (gas / second)")
    demand_mean = st.number_input("Mean demand", 0.0, 20000000.0, 1000000.0, step=100000.0, format="%.0f")
    demand_std  = st.number_input("Volatility (std dev)", 0.0, 20000000.0, 500000.0, step=100000.0, format="%.0f")

    # capacity
    st.subheader("Supply / capacity")
    capacity_gps = st.number_input("Hardware capacity (gas / second)", 100000.0, 20000000.0, 2000000.0, step=100000.0, format="%.0f")
    target_util  = st.slider("Target utilisation", 0.1, 1.0, 0.5, step=0.05)

    # algorithm
    st.subheader("Fee algorithm")
    k_percent = st.slider("Learning-rate k (% at 2× target)", 0.1, 25.0, 12.5, 0.1)
    scheme    = st.selectbox("Smoothing scheme", ["EMA", "Rolling window"])
    if scheme == "EMA":
        alpha  = st.slider("EMA alpha", 0.01, 1.0, 0.2, 0.01)
        win_sec = None
    else:
        win_sec = st.number_input("Window length [s]", 1.0, 120.0, 10.0, step=1.0)
        alpha   = None

    # fees
    st.subheader("Fee bounds (USDC / gas)")
    base_fee0 = st.number_input("Initial base-fee", 0.0, 0.01, 1e-6, step=1e-6, format="%.9f")
    base_min  = st.number_input("Floor B_min", 0.0, 0.01, 1e-6, step=1e-6, format="%.9f")
    base_max  = st.number_input("Cap B_max", 0.000001, 1.0, 0.001, step=0.001, format="%.6f")
    seed      = st.number_input("Random seed (0=random)", 0, 10000, 0, step=1)

def simulate():
    rng = np.random.default_rng(None if seed == 0 else int(seed))
    gas_limit_block = capacity_gps * block_time
    g_target_window = capacity_gps * target_util

    base_fee = base_fee0
    bf   = np.zeros(int(num_blocks))
    guse = np.zeros(int(num_blocks))

    if scheme == "Rolling window":
        win_blocks = max(1, int(win_sec / block_time))
        q = deque()

    ema_hat = g_target_window

    for t in range(int(num_blocks)):
        demand_block = max(0.0, rng.normal(demand_mean*block_time, demand_std*np.sqrt(block_time)))
        gas_used = min(demand_block, gas_limit_block)

        bf[t]   = base_fee
        guse[t] = gas_used

        if scheme == "EMA":
            ema_hat = (1-alpha)*ema_hat + alpha*gas_used
            hatG = ema_hat / block_time
        else:
            q.append(gas_used)
            if len(q) > win_blocks:
                q.popleft()
            hatG = sum(q) / (len(q)*block_time)

        delta = hatG / g_target_window - 1.0
        base_fee = base_fee * (1 + k_percent/100 * delta)
        base_fee = min(base_max, max(base_min, base_fee))

    return bf, guse

if st.button("Run simulation"):
    bf, gu = simulate()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Base-fee trajectory")
        fig, ax = plt.subplots()
        ax.plot(bf)
        ax.set_xlabel("Block")
        ax.set_ylabel("Base-fee [USDC/gas]")
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.subheader("Gas used per block")
        fig2, ax2 = plt.subplots()
        ax2.plot(gu)
        ax2.set_xlabel("Block")
        ax2.set_ylabel("Gas used")
        st.pyplot(fig2, clear_figure=True)
    st.success("Simulation finished")
