import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional
from collections import deque
import matplotlib.pyplot as plt
import io
import os
from fpdf import FPDF
import tempfile
import matplotlib as mpl
import scipy.stats

@dataclass
class SimulationConfig:
    num_blocks: int
    block_time: float
    demand_source: str
    demand_model: Optional[str] = None
    demand_mean: Optional[float] = None
    volatility_ratio: Optional[float] = None
    ar1_persistence: Optional[float] = None
    csv_file: Optional[str] = None
    csv_gas_column: Optional[str] = None
    use_range: Optional[bool] = False
    start_index: Optional[int] = 0
    end_index: Optional[int] = None
    use_elasticity: bool = True
    elasticity: Optional[float] = None
    reference_price: Optional[float] = None
    capacity_gps: float = 20000000.0
    target_util: float = 0.5
    k_percent: float = 2.0
    scheme: str = "EMA"
    alpha: Optional[float] = 0.2
    win_sec: Optional[float] = None
    base_fee0: float = 0.00002
    base_min: float = 0.00002
    base_max: float = 0.001
    seed: int = 0
    erc20_transfer_gas: int = 50000

@dataclass
class SimulationResults:
    base_fees: np.ndarray
    gas_used: np.ndarray
    demand: np.ndarray
    adjustment_factors: np.ndarray
    elasticity_factors: np.ndarray
    transfer_costs: np.ndarray
    base_means: np.ndarray
    adjusted_means: np.ndarray
    config: SimulationConfig

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'Block': np.arange(len(self.base_fees)),
            'Base_Fee': self.base_fees,
            'Gas_Used': self.gas_used,
            'Original_Demand': self.demand,
            'Adjustment_Factor': self.adjustment_factors,
            'Elasticity_Factor': self.elasticity_factors,
            'ERC20_Transfer_Cost': self.transfer_costs,
            'Base_Mean': self.base_means,
            'Adjusted_Mean': self.adjusted_means
        })

def calculate_ar1_coefficient(series: np.ndarray) -> float:
    series = series[~np.isnan(series)]
    if len(series) <= 1:
        return np.nan
    return np.corrcoef(series[1:], series[:-1])[0, 1]

def load_demand_from_csv(csv_file, column_name, num_blocks=None, start_index=0, end_index=None):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    df = pd.read_csv(csv_file)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV file. Available columns: {', '.join(df.columns)}")
    full_data = df[column_name].values
    if end_index is not None and end_index > start_index:
        end_index = min(end_index, len(full_data))
        demand_data = full_data[start_index:end_index]
    else:
        demand_data = full_data[start_index:]
    if num_blocks is not None:
        if len(demand_data) > num_blocks:
            demand_data = demand_data[:num_blocks]
        elif len(demand_data) < num_blocks:
            repetitions = int(np.ceil(num_blocks / len(demand_data)))
            demand_data = np.tile(demand_data, repetitions)[:num_blocks]
    return demand_data

def simulate(config: SimulationConfig) -> SimulationResults:
    rng = np.random.default_rng(None if config.seed == 0 else int(config.seed))
    blocks = int(config.num_blocks)
    base_fees = np.zeros(blocks)
    gas_used = np.zeros(blocks)
    demand_blocks = np.zeros(blocks)
    adjustment_factors = np.zeros(blocks)
    elasticity_factors = np.ones(blocks)
    base_means = np.zeros(blocks)
    adjusted_means = np.zeros(blocks)

    gas_limit_block = config.capacity_gps * config.block_time
    g_target_window = config.capacity_gps * config.target_util
    if config.scheme == "Rolling window":
        win_blocks = max(1, int(config.win_sec / config.block_time))
        q = deque()
    else:
        ema_hat = g_target_window

    if config.demand_source == "Simulate demand":
        base_fee = config.base_fee0
        prev_demand = None
        for t in range(blocks):
            base_mean = config.demand_mean * config.block_time
            base_means[t] = base_mean
            if config.use_elasticity and config.elasticity is not None and config.reference_price is not None:
                ref_price = config.reference_price if config.reference_price > 0 else 0.00001
                price_ratio = base_fee / ref_price
                elasticity_factor = price_ratio ** config.elasticity if price_ratio > 0 else 1.0
                mean = base_mean * elasticity_factor
                elasticity_factors[t] = elasticity_factor
            else:
                mean = base_mean
                elasticity_factors[t] = 1.0
            adjusted_means[t] = mean

            if config.demand_model == "Normal distribution":
                std = config.volatility_ratio * mean
                demand = max(0.0, rng.normal(mean, std))
            elif config.demand_model == "AR(1) process":
                persistence = config.ar1_persistence
                std = config.volatility_ratio * mean
                if t == 0:
                    if abs(persistence) < 1:
                        demand = max(0.0, rng.normal(mean, std / np.sqrt(1 - persistence**2)))
                    else:
                        demand = max(0.0, rng.normal(mean, std))
                else:
                    shock = rng.normal(0, std)
                    demand = max(0.0, mean + persistence * (prev_demand - mean) + shock)
                prev_demand = demand
            else:
                raise ValueError(f"Unknown demand_model: {config.demand_model}")
            demand_blocks[t] = demand

            gas = min(demand, gas_limit_block)
            gas_used[t] = gas

            if config.scheme == "EMA":
                ema_hat = (1 - config.alpha) * ema_hat + config.alpha * gas
                hatG = ema_hat / config.block_time
            else:
                q.append(gas)
                if len(q) > win_blocks:
                    q.popleft()
                hatG = sum(q) / (len(q) * config.block_time)

            delta = hatG / g_target_window - 1.0
            adjustment_factor = (1 + config.k_percent / 100 * delta)
            adjustment_factors[t] = adjustment_factor
            base_fees[t] = base_fee
            base_fee = base_fee * adjustment_factor
            base_fee = min(config.base_max, max(config.base_min, base_fee))
        transfer_costs = base_fees * config.erc20_transfer_gas
        return SimulationResults(
            base_fees, gas_used, demand_blocks, adjustment_factors, elasticity_factors,
            transfer_costs, base_means, adjusted_means, config
        )

    elif config.demand_source == "Load from CSV":
        if config.use_range:
            demand_blocks = load_demand_from_csv(
                config.csv_file, config.csv_gas_column, blocks,
                start_index=config.start_index, end_index=config.end_index
            )
        else:
            demand_blocks = load_demand_from_csv(
                config.csv_file, config.csv_gas_column, blocks
            )
        base_fee = config.base_fee0
        if config.scheme == "Rolling window":
            win_blocks = max(1, int(config.win_sec / config.block_time))
            q = deque()
        else:
            ema_hat = g_target_window
        for t in range(blocks):
            demand = demand_blocks[t]
            base_means[t] = np.mean(demand_blocks)
            adjusted_means[t] = base_means[t]
            elasticity_factors[t] = 1.0
            gas = min(demand, gas_limit_block)
            gas_used[t] = gas
            if config.scheme == "EMA":
                ema_hat = (1 - config.alpha) * ema_hat + config.alpha * gas
                hatG = ema_hat / config.block_time
            else:
                q.append(gas)
                if len(q) > win_blocks:
                    q.popleft()
                hatG = sum(q) / (len(q) * config.block_time)
            delta = hatG / g_target_window - 1.0
            adjustment_factor = (1 + config.k_percent / 100 * delta)
            adjustment_factors[t] = adjustment_factor
            base_fees[t] = base_fee
            base_fee = base_fee * adjustment_factor
            base_fee = min(config.base_max, max(config.base_min, base_fee))
        transfer_costs = base_fees * config.erc20_transfer_gas
        return SimulationResults(
            base_fees, gas_used, demand_blocks, adjustment_factors, elasticity_factors,
            transfer_costs, base_means, adjusted_means, config
        )
    else:
        raise ValueError("Unknown demand source.")

def save_config_csv(config: SimulationConfig, path: str):
    pd.DataFrame([asdict(config)]).to_csv(path, index=False)

def load_config_csv(path: str) -> SimulationConfig:
    df = pd.read_csv(path)
    d = df.iloc[0].to_dict()
    for k, v in d.items():
        if pd.isna(v):
            d[k] = None
    return SimulationConfig(**d)

def save_results_csv(results: SimulationResults, path: str):
    results.to_dataframe().to_csv(path, index=False)

def load_results_csv(path: str, config: SimulationConfig) -> SimulationResults:
    df = pd.read_csv(path)
    return SimulationResults(
        base_fees=df['Base_Fee'].values,
        gas_used=df['Gas_Used'].values,
        demand=df['Original_Demand'].values,
        adjustment_factors=df['Adjustment_Factor'].values,
        elasticity_factors=df['Elasticity_Factor'].values,
        transfer_costs=df['ERC20_Transfer_Cost'].values,
        base_means=df['Base_Mean'].values,
        adjusted_means=df['Adjusted_Mean'].values,
        config=config
    )

def render_latex_to_image(latex, width=1200, fontsize=11):
    fig = plt.figure(figsize=(width/100, 1))
    plt.axis('off')
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    plt.text(0.0, 0.0, f'${latex}$', fontsize=fontsize, ha='center', va='center')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0, dpi=300, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_report(results: SimulationResults, output_pdf_path: str):
    config = results.config
    df = results.to_dataframe()
    transfer_costs = results.transfer_costs
    elasticity_factors = results.elasticity_factors
    demand = results.demand
    gas_used = results.gas_used
    base_fees = results.base_fees
    base_means = results.base_means
    adjusted_means = results.adjusted_means
    gas_limit = config.capacity_gps * config.block_time
    # --- Output stats table ---
    percent_blocks_above_limit = 100.0 * np.sum(demand > gas_limit) / len(demand)
    transfer_stats = {
        "Min": np.min(transfer_costs),
        "25%": np.percentile(transfer_costs, 25),
        "50% (Median)": np.percentile(transfer_costs, 50),
        "Mean": np.mean(transfer_costs),
        "75%": np.percentile(transfer_costs, 75),
        "99%": np.percentile(transfer_costs, 99),
        "Max": np.max(transfer_costs),
        "Std Dev": np.std(transfer_costs),
        "Percent blocks with demand above gas limit": percent_blocks_above_limit
    }
    transfer_ar1 = calculate_ar1_coefficient(transfer_costs)
    gas_used_ar1 = calculate_ar1_coefficient(gas_used)
    mean_demand_reduction = (1 - np.mean(elasticity_factors)) * 100
    max_demand_reduction = (1 - np.min(elasticity_factors)) * 100
    # --- Add mean gas used / original demand metric (guard against zero demand) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        gas_used_over_demand = np.where(demand != 0, gas_used / demand, np.nan)
    mean_gas_used_over_demand = np.nanmean(gas_used_over_demand) * 100
    # --- Math formulas as inline ASCII text ---
    base_fee_formula = "B_{t+1} = clip(B_t [1 + k (G_hat_t / G_target - 1)], B_min, B_max)"
    smoothing_formula = "G_hat_t = (1-alpha)G_hat_{t-1} + alpha G_t"
    demand_formula = "D_t = max{0, mu_t + rho(D_{t-1} - mu_t) + epsilon_t}"
    elasticity_formula = "mu_t = mu_0 (P_t / P_ref)^eta"
    # --- Plots ---
    fig1, ax1 = plt.subplots()
    ax1.hist(base_fees, bins=30, color='skyblue', edgecolor='black')
    ax1.set_title("Gas Price Distribution")
    ax1.set_xlabel("Base-fee [USDC-cent/gas]")
    ax1.set_ylabel("Frequency")
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    plt.close(fig1)
    fig2, ax2 = plt.subplots()
    # KDE for demand and gas_used
    x_min = min(np.min(demand), np.min(gas_used))
    x_max = max(np.max(demand), np.max(gas_used))
    x_grid = np.linspace(x_min, x_max, 1000)
    kde_demand = scipy.stats.gaussian_kde(demand)
    kde_gas_used = scipy.stats.gaussian_kde(gas_used)
    ax2.plot(x_grid, kde_demand(x_grid), label='Original Demand', color='orange', linewidth=2)
    ax2.plot(x_grid, kde_gas_used(x_grid), label='Actual Gas Used', color='blue', linewidth=2)
    ax2.axvline(gas_limit, color='red', linestyle='--', label=f'Gas Limit: {gas_limit:.0f}')
    ax2.set_title("Demand vs Gas Used (PDF)")
    ax2.set_xlabel("Gas")
    ax2.set_ylabel("Density")
    ax2.legend()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig2)
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
    ax3a.plot(base_fees, color='tab:blue')
    ax3a.set_ylabel('Base Fee [USDC-cent/gas]')
    ax3a.set_title('Base Fee Trajectory')
    ax3b.plot(gas_used, color='tab:green')
    ax3b.set_ylabel('Gas Used')
    ax3b.set_xlabel('Block')
    ax3b.set_title('Gas Used per Block')
    plt.tight_layout()
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', bbox_inches='tight')
    plt.close(fig3)
    # Save plots to temp files for FPDF
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
        tmp1.write(buf1.getvalue())
        tmp1.flush()
        img1_path = tmp1.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
        tmp2.write(buf2.getvalue())
        tmp2.flush()
        img2_path = tmp2.name
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp3:
        tmp3.write(buf3.getvalue())
        tmp3.flush()
        img3_path = tmp3.name
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "EIP-1559 Fee Simulation Report", ln=True, align='C')
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "1. Key Model Inputs", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Target Utilization (u_target): {config.target_util:.2f}", ln=True)
    pdf.cell(0, 6, f"Capacity (C, gas/sec): {config.capacity_gps:.0f}", ln=True)
    pdf.cell(0, 6, f"Learning Rate (k): {config.k_percent:.2f} (%)", ln=True)
    pdf.cell(0, 6, f"Base Fee Floor (B_min): {config.base_min}", ln=True)
    pdf.cell(0, 6, f"Base Fee Cap (B_max): {config.base_max}", ln=True)
    pdf.cell(0, 6, f"Smoothing Scheme: {config.scheme} (alpha={config.alpha if config.scheme=='EMA' else 'N/A'}, win_sec={config.win_sec if config.scheme=='Rolling window' else 'N/A'})", ln=True)
    pdf.cell(0, 6, f"Base-fee Update Rule: {base_fee_formula}", ln=True)
    pdf.cell(0, 6, f"Gas Smoothing (EMA): {smoothing_formula}", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "2. Key Simulation Assumptions", ln=True)
    pdf.set_font("Arial", '', 10)
    if config.demand_source == "Simulate demand":
        pdf.cell(0, 6, f"Demand Model: {config.demand_model}", ln=True)
        if config.demand_model == "AR(1) process":
            pdf.cell(0, 6, f"AR(1) Persistence (rho): {config.ar1_persistence}", ln=True)
        pdf.cell(0, 6, f"Mean Demand (mu_0): {config.demand_mean}", ln=True)
        pdf.cell(0, 6, f"Volatility Ratio (nu): {config.volatility_ratio}", ln=True)
    else:
        pdf.cell(0, 6, f"Demand Source: CSV ({config.csv_file}, column: {config.csv_gas_column})", ln=True)
        if config.use_range:
            pdf.cell(0, 6, f"CSV Range: {config.start_index} to {config.end_index}", ln=True)
    pdf.cell(0, 6, f"Price Elasticity Enabled: {config.use_elasticity}", ln=True)
    if config.use_elasticity:
        pdf.cell(0, 6, f"Elasticity Coefficient (eta): {config.elasticity}", ln=True)
        pdf.cell(0, 6, f"Reference Price (P_ref) [USDC-cent/gas]: {config.reference_price}", ln=True)
    pdf.cell(0, 6, f"ERC-20 Transfer Gas Units: {config.erc20_transfer_gas}", ln=True)
    pdf.cell(0, 6, f"Demand Model Equation: {demand_formula}", ln=True)
    pdf.cell(0, 6, f"Elasticity Adjustment: {elasticity_formula}", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "3. Simulation Run Settings", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Blocks to Simulate: {config.num_blocks}", ln=True)
    pdf.cell(0, 6, f"Block Interval (Delta, sec): {config.block_time}", ln=True)
    pdf.cell(0, 6, f"Initial Base Fee (B_0): {config.base_fee0}", ln=True)
    pdf.cell(0, 6, f"Random Seed: {config.seed}", ln=True)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Simulation Output Statistics", ln=True)
    pdf.set_font("Arial", '', 10)
    # --- Move subplots first ---
    page_width = pdf.w - 2 * pdf.l_margin
    pdf.cell(0, 6, "Base Fee and Gas Used (Subplots):", ln=True)
    pdf.image(img3_path, w=page_width*.7)
    pdf.cell(0, 6, "Demand vs Gas Used (PDF):", ln=True)
    pdf.image(img2_path, w=page_width*.7)
    pdf.cell(0, 6, "Gas Price Distribution Histogram:", ln=True)
    pdf.image(img1_path, w=page_width*.7)
    # --- Label the statistics table ---
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 7, "ERC-20 Transfer Cost Statistics", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(70, 6, "Statistic", border=1)
    pdf.cell(60, 6, "Value", border=1, ln=True)
    for k, v in transfer_stats.items():
        pdf.cell(70, 6, k, border=1)
        if k == "Percent blocks with demand above gas limit":
            pdf.cell(60, 6, f"{v:.2f}%", border=1, ln=True)
        else:
            pdf.cell(60, 6, f"{v:.8f}", border=1, ln=True)
    pdf.cell(70, 6, "ERC-20 Transfer Cost AR(1) coefficient", border=1)
    pdf.cell(60, 6, f"{transfer_ar1:.4f}", border=1, ln=True)
    pdf.cell(70, 6, "Gas Used AR(1) coefficient", border=1)
    pdf.cell(60, 6, f"{gas_used_ar1:.4f}", border=1, ln=True)
    pdf.cell(70, 6, "Mean demand reduction (%)", border=1)
    pdf.cell(60, 6, f"{mean_demand_reduction:.2f}", border=1, ln=True)
    pdf.cell(70, 6, "Max demand reduction (%)", border=1)
    pdf.cell(60, 6, f"{max_demand_reduction:.2f}", border=1, ln=True)
    # --- Display new metric ---
    pdf.cell(70, 6, "Mean gas used / original demand (%)", border=1)
    pdf.cell(60, 6, f"{mean_gas_used_over_demand:.2f}", border=1, ln=True)
    pdf.output(output_pdf_path)
    for f in [img1_path, img2_path, img3_path]:
        try:
            os.remove(f)
        except Exception:
            pass 