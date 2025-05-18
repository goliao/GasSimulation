import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import base64
import io
from simulation_lib import (
    SimulationConfig, simulate, save_config_csv, load_config_csv,
    save_results_csv, load_results_csv, generate_report, calculate_ar1_coefficient
)
from scipy.stats import gaussian_kde

st.set_page_config(page_title="EIP-1559 Fee Simulator", layout="wide")
st.title("USDC-Based Avalanche Subnet • EIP-1559 Fee Simulator")

if 'simulations' not in st.session_state:
    st.session_state.simulations = []

with st.sidebar:
    st.header("Simulation Parameters")
    st.subheader("Save/Load Config")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        save_config_btn = st.button("Save Config")
    with config_col2:
        uploaded_config = st.file_uploader("Load Config (CSV)", type="csv")
        if uploaded_config is not None:
            load_config_btn = st.button("Apply Config")
            if load_config_btn:
                config = load_config_csv(uploaded_config)
                st.success("Configuration loaded successfully!")
    num_blocks = st.number_input("Blocks to simulate", 10, 50000, 20000, step=100)
    block_time = st.number_input("Block interval [s]", 0.1, 10.0, 1.0, step=0.1)
    st.subheader("Demand Source")
    demand_source = st.radio("Select demand source", ["Simulate demand", "Load from CSV"])
    if demand_source == "Simulate demand":
        st.subheader("Demand (gas / second)")
        demand_model = st.selectbox("Demand model", ["Normal distribution", "AR(1) process"])
        demand_mean = st.number_input("Mean demand", 0.0, 20000000.0, 10000000.0, step=100000.0, format="%.0f")
        volatility_ratio = st.slider("Volatility ratio (std/mean)", 0.0, 10.0, 0.5, step=0.1)
        ar1_persistence = None
        if demand_model == "AR(1) process":
            ar1_persistence = st.slider("AR(1) persistence", -0.99, 0.99, 0.15, step=0.01)
        csv_file = None
        csv_gas_column = None
        use_range = False
        start_index = 0
        end_index = None
    else:
        st.subheader("CSV File Input")
        csv_file = st.text_input("CSV file path", "gas_used_basediv6_sim.csv")
        csv_gas_column = st.text_input("Gas demand column name", "gas_used")
        use_range = st.checkbox("Select specific data range", False)
        if use_range:
            if os.path.exists(csv_file):
                try:
                    total_rows = len(pd.read_csv(csv_file))
                    st.info(f"CSV file contains {total_rows} rows.")
                    max_end_point = total_rows - 1
                except Exception:
                    max_end_point = 100000
            else:
                max_end_point = 100000
            col1, col2 = st.columns(2)
            with col1:
                start_index = st.number_input("Start index", 0, max_end_point, 0, step=100)
            with col2:
                end_index = st.number_input("End index", 0, max_end_point, min(2000, max_end_point), step=100)
            if end_index <= start_index:
                st.warning("End index must be greater than start index. Using default range.")
        else:
            start_index = 0
            end_index = None
        demand_model = None
        demand_mean = None
        volatility_ratio = None
        ar1_persistence = None
    st.subheader("Price Elasticity")
    use_elasticity = st.checkbox("Enable price-elastic demand", True)
    elasticity = None
    reference_price = None
    if use_elasticity:
        elasticity = st.slider("Price elasticity coefficient", -3.0, 0.0, -1.0, 0.1)
        reference_price = st.number_input("Reference price (USDC-cent/gas)", 0.000001, 0.001, 0.00002, step=0.000001, format="%.8f")
    capacity_gps = st.number_input("Hardware capacity (gas / second)", 100000.0, 50000000.0, 20000000.0, step=100000.0, format="%.0f")
    target_util  = st.slider("Target utilisation", 0.1, 1.0, 0.5, step=0.05)
    st.subheader("Fee algorithm")
    k_percent = st.slider("Learning-rate k (% at 2× target)", 0.1, 25.0, 2.0, 0.1)
    scheme    = st.selectbox("Smoothing scheme", ["EMA", "Rolling window"])
    if scheme == "EMA":
        alpha  = st.slider("EMA alpha", 0.01, 1.0, 0.2, 0.01)
        win_sec = None
    else:
        win_sec = st.number_input("Window length [s]", 1.0, 120.0, 10.0, step=1.0)
        alpha   = None
    st.subheader("Fee bounds (USDC-cent / gas)")
    base_fee0 = st.number_input("Initial base-fee", 0.0, 0.01, 0.00002, step=1e-6, format="%.9f")
    base_min  = st.number_input("Floor B_min", 0.0, 0.01, 0.00002, step=1e-6, format="%.9f")
    base_max  = st.number_input("Cap B_max", 0.000001, 1.0, 0.001, step=0.001, format="%.6f")
    seed      = st.number_input("Random seed (0=random)", 0, 10000, 0, step=1)
    st.subheader("Transaction Types")
    erc20_transfer_gas = st.number_input("ERC-20 Transfer Gas Units", 10000, 100000, 50000, step=1000)
    st.subheader("Debug Options")
    enable_debug = st.checkbox("Enable debug output (print & CSV)", value=False)
    config = SimulationConfig(
        num_blocks=int(num_blocks),
        block_time=float(block_time),
        demand_source=demand_source,
        demand_model=demand_model,
        demand_mean=float(demand_mean) if demand_mean is not None else None,
        volatility_ratio=float(volatility_ratio) if volatility_ratio is not None else None,
        ar1_persistence=float(ar1_persistence) if ar1_persistence is not None else None,
        csv_file=csv_file,
        csv_gas_column=csv_gas_column,
        use_range=use_range,
        start_index=int(start_index) if start_index is not None else 0,
        end_index=int(end_index) if end_index is not None else None,
        use_elasticity=use_elasticity,
        elasticity=float(elasticity) if elasticity is not None else None,
        reference_price=float(reference_price) if reference_price is not None else None,
        capacity_gps=float(capacity_gps),
        target_util=float(target_util),
        k_percent=float(k_percent),
        scheme=scheme,
        alpha=float(alpha) if alpha is not None else None,
        win_sec=float(win_sec) if win_sec is not None else None,
        base_fee0=float(base_fee0),
        base_min=float(base_min),
        base_max=float(base_max),
        seed=int(seed),
        erc20_transfer_gas=int(erc20_transfer_gas)
    )
    if save_config_btn:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = f"fee_sim_config_{date_str}.csv"
        save_config_csv(config, config_path)
        st.sidebar.success(f"Config saved as {config_path}")

if st.button("Run simulation"):
    try:
        results = simulate(config)
        st.session_state.simulations.append(results)
        st.success("Simulation finished")
    except Exception as e:
        st.error(f"Simulation failed: {e}")

if st.session_state.simulations:
    results = st.session_state.simulations[-1]
    df = results.to_dataframe()
    st.subheader("Export Results")
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"fee_sim_results_{date_str}.csv"
    df_csv = df.to_csv(index=False).encode()
    st.download_button("Download Simulation Data (CSV)", df_csv, file_name=csv_name, mime="text/csv")
    pdf_name = f"fee_sim_report_{date_str}.pdf"
    pdf_path = f"./{pdf_name}"
    if st.button("Generate PDF Report"):
        generate_report(results, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name, mime="application/pdf")
    # Two-column layout for results
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Base Fee and Gas Used (Subplots)")
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        ax3a.plot(results.base_fees, color='tab:blue')
        ax3a.set_ylabel('Base Fee [USDC-cent/gas]')
        ax3a.set_title('Base Fee Trajectory')
        ax3b.plot(results.gas_used, color='tab:green')
        ax3b.set_ylabel('Gas Used')
        ax3b.set_xlabel('Block')
        ax3b.set_title('Gas Used per Block')
        plt.tight_layout()
        st.pyplot(fig3)

        st.subheader("Gas Price Distribution Histogram")
        fig1, ax1 = plt.subplots()
        ax1.hist(results.base_fees, bins=30, color='skyblue', edgecolor='black')
        ax1.set_xlabel("Base-fee [USDC-cent/gas]")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

        st.subheader("Original Demand vs Actual Gas Used Histogram")
        fig2, ax2 = plt.subplots()
        # KDE for demand and gas_used
        x_min = min(np.min(results.demand), np.min(results.gas_used))
        x_max = max(np.max(results.demand), np.max(results.gas_used))
        x_grid = np.linspace(x_min, x_max, 1000)
        kde_demand = gaussian_kde(results.demand)
        kde_gas_used = gaussian_kde(results.gas_used)
        ax2.plot(x_grid, kde_demand(x_grid), label='Original Demand', color='orange', linewidth=2)
        ax2.plot(x_grid, kde_gas_used(x_grid), label='Actual Gas Used', color='blue', linewidth=2)
        gas_limit = results.config.capacity_gps * results.config.block_time
        ax2.axvline(gas_limit, color='red', linestyle='--', label=f'Gas Limit: {gas_limit:.0f}')
        ax2.set_xlabel("Gas")
        ax2.set_ylabel("Density")
        ax2.set_title("Demand vs Gas Used (PDF)")
        ax2.legend()
        st.pyplot(fig2)
    with col2:
        st.subheader("ERC-20 Transfer Cost Statistics")
        transfer_costs = results.transfer_costs
        transfer_stats = {
            "Min": transfer_costs.min(),
            "25%": np.percentile(transfer_costs, 25),
            "50% (Median)": np.percentile(transfer_costs, 50),
            "Mean": transfer_costs.mean(),
            "75%": np.percentile(transfer_costs, 75),
            "99%": np.percentile(transfer_costs, 99),
            "Max": transfer_costs.max(),
            "Std Dev": transfer_costs.std(),
        }
        gas_limit = results.config.capacity_gps * results.config.block_time
        percent_blocks_above_limit = 100.0 * np.sum(results.demand > gas_limit) / len(results.demand)
        st.table(pd.DataFrame(list(transfer_stats.items()), columns=['Statistic', 'Value [USDC-cent]']))
        st.write(f"Percent of blocks with demand above gas limit: {percent_blocks_above_limit:.2f}%")
        st.write(f"ERC-20 Transfer Cost AR(1) coefficient: {calculate_ar1_coefficient(results.transfer_costs):.4f}")
        mean_demand_reduction = (1 - np.mean(results.elasticity_factors)) * 100
        max_demand_reduction = (1 - np.min(results.elasticity_factors)) * 100
        st.write(f"Mean demand reduction: {mean_demand_reduction:.2f}%")
        st.write(f"Max demand reduction: {max_demand_reduction:.2f}%")
        st.write(f"Gas Used AR(1) coefficient: {calculate_ar1_coefficient(results.gas_used):.4f}")
