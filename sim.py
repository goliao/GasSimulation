import numpy as np
import matplotlib.pyplot as plt
import textwrap
import os
from collections import deque

# ------------------------------------------------------------------
# 1) create a Streamlit UI app so the user can interactively explore
# ------------------------------------------------------------------
streamlit_code = textwrap.dedent("""
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import deque

    st.set_page_config(page_title="EIP-1559 Fee Simulator", layout="wide")
    st.title("USDC-Based Avalanche Subnet • EIP-1559 Fee Simulator")

    # ---------------- Sidebar configuration -----------------
    with st.sidebar:
        st.header("Simulation Parameters")

        # time / length
        num_blocks   = st.number_input("Blocks to simulate", 10, 50_000, 2_000, step=100)
        block_time   = st.number_input("Block interval [s]", 0.1, 10.0, 1.0, step=0.1)

        # demand process
        st.subheader("Demand (gas / second)")
        demand_mean  = st.number_input("Mean demand ", 0.0, 20_000_000.0, 1_000_000.0, step=100_000.0, format="%.0f")
        demand_std   = st.number_input("Volatility (std dev)", 0.0, 20_000_000.0, 500_000.0, step=100_000.0, format="%.0f")

        # hardware capacity
        st.subheader("Supply / capacity")
        capacity_gps = st.number_input("Hardware capacity (gas / second)", 100_000.0, 20_000_000.0, 2_000_000.0, step=100_000.0, format="%.0f")
        target_util  = st.slider("Target utilisation", 0.1, 1.0, 0.5, step=0.05)

        # fee algorithm
        st.subheader("Fee algorithm")
        k_percent    = st.slider("Learning-rate k (% change per update at 2× target)", 0.1, 25.0, 12.5, 0.1)
        scheme       = st.selectbox("Smoothing scheme", ["EMA", "Rolling window"])
        if scheme == "EMA":
            alpha     = st.slider("EMA α (0-1)", 0.01, 1.0, 0.2, 0.01)
            win_sec   = None
        else:
            win_sec   = st.number_input("Window length [s]", 1.0, 120.0, 10.0, step=1.0)
            alpha     = None

        # fees
        st.subheader("Fee bounds  (USDC / gas)")
        base_fee0   = st.number_input("Initial base-fee", 0.0, 0.01, 1e-6, step=1e-6, format="%.9f")
        base_min    = st.number_input("Base-fee floor  B_min", 0.0, 0.01, 1e-6, step=1e-6, format="%.9f")
        base_max    = st.number_input("Base-fee cap    B_max", 0.000001, 1.0, 0.001, step=0.001, format="%.6f")

        seed        = st.number_input("Random seed (0=random)", 0, 10_000, 0, step=1)

    # -------------- internal helper -----------------
    def simulate():
        rng = np.random.default_rng(None if seed == 0 else int(seed))

        gas_limit_block = capacity_gps * block_time
        g_target_window = capacity_gps * target_util

        # Pre-compute number of blocks per second & per window
        blocks = int(num_blocks)
        base_fee  = base_fee0
        base_fees = np.zeros(blocks)
        gas_used  = np.zeros(blocks)
        backlog   = deque()

        # helper for rolling window sum
        if scheme == "Rolling window":
            win_blocks = int(win_sec / block_time)
            win_blocks = max(1, win_blocks)

        ema_hatG = g_target_window  # start at target

        for t in range(blocks):
            # draw demand for THIS block in gas units
            mean_block = demand_mean * block_time
            std_block  = demand_std * np.sqrt(block_time)
            demand     = max(0.0, rng.normal(mean_block, std_block))

            gas       = min(demand, gas_limit_block)
            gas_used[t] = gas
            base_fees[t] = base_fee

            # ------ smoothing demand -------
            if scheme == "EMA":
                ema_hatG = (1 - alpha) * ema_hatG + alpha * gas
                hatG     = ema_hatG * (block_time)  # convert to per-second then compare
            else:
                backlog.append(gas)
                if len(backlog) > win_blocks:
                    backlog.popleft()
                hatG = sum(backlog) / (len(backlog) * block_time)  # per-second rate

            # ------ fee update --------------
            delta = (hatG / g_target_window) - 1.0
            base_fee = base_fee * (1 + k_percent/100.0 * delta)
            base_fee = max(base_min, min(base_max, base_fee))

        return base_fees, gas_used

    if st.button("Run simulation"):
        bf, gu = simulate()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Base-fee trajectory")
            fig, ax = plt.subplots()
            ax.plot(bf)
            ax.set_xlabel("Block")
            ax.set_ylabel("Base fee  [USDC / gas]")
            st.pyplot(fig, clear_figure=True)

        with col2:
            st.subheader("Gas used per block")
            fig2, ax2 = plt.subplots()
            ax2.plot(gu)
            ax2.axhline(capacity_gps * block_time, linestyle=':')
            ax2.set_xlabel("Block")
            ax2.set_ylabel("Gas used")
            st.pyplot(fig2, clear_figure=True)

        st.success("Done!")
""")

file_name = "/mnt/data/gas_fee_simulator_streamlit.py"
with open(file_name, "w") as f:
    f.write(streamlit_code)

# ------------------------------------------------------------------
# 2) quick example run (in-notebook) for the user to see output
# ------------------------------------------------------------------
def quick_sim():
    num_blocks      = 1000
    block_time      = 1.0
    demand_mean     = 1_000_000
    demand_std      = 500_000
    capacity_gps    = 2_000_000
    target_util     = 0.5
    k_percent       = 12.5
    scheme          = "EMA"
    alpha           = 0.2
    win_sec         = None
    base_fee0       = 1e-6
    base_min        = 1e-6
    base_max        = 0.001
    rng             = np.random.default_rng(42)

    gas_limit_block = capacity_gps * block_time
    g_target_window = capacity_gps * target_util

    base_fee  = base_fee0
    base_fees = np.zeros(num_blocks)
    gas_used  = np.zeros(num_blocks)
    ema_hatG  = g_target_window

    for t in range(num_blocks):
        demand = max(0.0, rng.normal(demand_mean * block_time,
                                     demand_std * np.sqrt(block_time)))
        gas    = min(demand, gas_limit_block)
        gas_used[t] = gas
        base_fees[t] = base_fee

        ema_hatG = (1 - alpha)*ema_hatG + alpha*gas
        delta    = (ema_hatG / g_target_window) - 1.0
        base_fee = base_fee * (1 + k_percent/100.0*delta)
        base_fee = np.clip(base_fee, base_min, base_max)

    return base_fees, gas_used

bf, gu = quick_sim()

# plots for preview
fig1, ax1 = plt.subplots()
ax1.plot(bf)
ax1.set_xlabel("Block")
ax1.set_ylabel("Base fee [USDC/gas]")
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(gu)
ax2.set_xlabel("Block")
ax2.set_ylabel("Gas used")
plt.tight_layout()
plt.show()

print(f"✅  Streamlit simulator saved to: {file_name}")
