import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Added for CSV handling
import os  # Added to check file existence
from collections import deque
from scipy import stats  # Added for AR(1) coefficient estimation
import seaborn as sns  # Added for improved visualizations
import json  # Added for configuration saving/loading
import base64  # Added for download functionality
import io  # Added for in-memory file operations
import datetime  # Added for timestamping exports

st.set_page_config(page_title="EIP-1559 Fee Simulator", layout="wide")
st.title("USDC-Based Avalanche Subnet • EIP-1559 Fee Simulator")

# Initialize session state for saving results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# Function to download dataframe as CSV
def get_csv_download_link(df, filename, link_text):
    """Generate a link to download a dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to download plot as PNG
def get_plot_download_link(fig, filename, link_text):
    """Generate a link to download a matplotlib figure as PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to save configuration to JSON
def save_configuration(config):
    """Save the current configuration to a JSON string for download"""
    json_str = json.dumps(config, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fee_sim_config_{date_str}.json"
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Configuration</a>'
    return href

# Function to load configuration from uploaded JSON
def load_configuration(uploaded_file):
    """Load configuration from an uploaded JSON file"""
    if uploaded_file is not None:
        try:
            return json.loads(uploaded_file.read())
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
    return None

with st.sidebar:
    st.header("Simulation Parameters")
    
    # Configuration save/load tools
    st.subheader("Save/Load Configuration")
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        save_config = st.button("Save Config")
    
    with config_col2:
        uploaded_config = st.file_uploader("Load Config", type="json")
        if uploaded_config is not None:
            load_config = st.button("Apply Config")
            if load_config:
                config = load_configuration(uploaded_config)
                if config is not None:
                    # We'll set these values after loading the config below
                    st.success("Configuration loaded successfully!")
    
    # timeline
    num_blocks = st.number_input("Blocks to simulate", 10, 50000, 2000, step=100)
    block_time = st.number_input("Block interval [s]", 0.1, 10.0, 1.0, step=0.1)

    # demand source selection
    st.subheader("Demand Source")
    demand_source = st.radio("Select demand source", ["Simulate demand", "Load from CSV"])
    
    if demand_source == "Simulate demand":
        # demand
        st.subheader("Demand (gas / second)")
        demand_model = st.selectbox("Demand model", ["Normal distribution", "AR(1) process"])
        
        demand_mean = st.number_input("Mean demand", 0.0, 20000000.0, 10000000.0, step=100000.0, format="%.0f")
        volatility_ratio = st.slider("Volatility ratio (std/mean)", 0.0, 10.0, 0.5, step=0.1, 
                                  help="Standard deviation as a fraction of mean demand")
        
        # AR(1) specific parameters
        if demand_model == "AR(1) process":
            ar1_persistence = st.slider("AR(1) persistence", -0.99, 0.99, 0.15, step=0.01, 
                                help="Persistence of demand shocks (-1 to 1, where 0 = no persistence)")
    else:  # Load from CSV
        st.subheader("CSV File Input")
        csv_file = st.text_input("CSV file path", "gas_used_basediv6_sim.csv")
        csv_gas_column = st.text_input("Gas demand column name", "gas_used")
        
        # Add checkbox to enable range selection
        use_range = st.checkbox("Select specific data range", False)
        
        if use_range:
            # If CSV file exists, try to get total rows for better UX
            if os.path.exists(csv_file):
                try:
                    total_rows = len(pd.read_csv(csv_file))
                    st.info(f"CSV file contains {total_rows} rows.")
                    max_end_point = total_rows - 1
                except Exception:
                    max_end_point = 100000  # Fallback value if can't read file
            else:
                max_end_point = 100000  # Fallback value if file doesn't exist
            
            # Allow selecting start and end points
            col1, col2 = st.columns(2)
            with col1:
                start_index = st.number_input("Start index", 0, max_end_point, 0, step=100)
            with col2:
                end_index = st.number_input("End index", 0, max_end_point, min(2000, max_end_point), step=100)
                
            if end_index <= start_index:
                st.warning("End index must be greater than start index. Using default range.")
        else:
            start_index = 0
            end_index = None  # Will use num_blocks to determine end point
            
        st.info("CSV file should contain a column with gas demand values.")

    # Price elasticity parameters
    st.subheader("Price Elasticity")
    use_elasticity = st.checkbox("Enable price-elastic demand", True)
    if use_elasticity:
        elasticity = st.slider("Price elasticity coefficient", -2.0, 0.0, -0.5, 0.1,
                            help="How demand responds to price changes (-1.0 = 1% price increase reduces demand by 1%)")
        reference_price = st.number_input("Reference price (USDC-cent/gas)", 0.000001, 0.001, 0.00002, 
                                        step=0.000001, format="%.8f",
                                        help="Base price at which demand is unchanged")

    # capacity
    st.subheader("Supply / capacity")
    capacity_gps = st.number_input("Hardware capacity (gas / second)", 100000.0, 50000000.0, 20000000.0, step=100000.0, format="%.0f")
    target_util  = st.slider("Target utilisation", 0.1, 1.0, 0.5, step=0.05)

    # algorithm
    st.subheader("Fee algorithm")
    k_percent = st.slider("Learning-rate k (% at 2× target)", 0.1, 25.0, 1/48*100, 0.1)
    scheme    = st.selectbox("Smoothing scheme", ["EMA", "Rolling window"])
    if scheme == "EMA":
        alpha  = st.slider("EMA alpha", 0.01, 1.0, 0.2, 0.01)
        win_sec = None
    else:
        win_sec = st.number_input("Window length [s]", 1.0, 120.0, 10.0, step=1.0)
        alpha   = None

    # fees
    st.subheader("Fee bounds (USDC-cent / gas)")
    base_fee0 = st.number_input("Initial base-fee", 0.0, 0.01, 0.00002, step=1e-6, format="%.9f")
    base_min  = st.number_input("Floor B_min", 0.0, 0.01, 0.00002, step=1e-6, format="%.9f")
    base_max  = st.number_input("Cap B_max", 0.000001, 1.0, 0.001, step=0.001, format="%.6f")
    seed      = st.number_input("Random seed (0=random)", 0, 10000, 0, step=1)

    # ERC-20 transfer gas used
    st.subheader("Transaction Types")
    erc20_transfer_gas = st.number_input("ERC-20 Transfer Gas Units", 10000, 100000, 50000, step=1000)
    
    # Transaction inclusion probability model
    st.subheader("Transaction Inclusion Model")
    model_tx_inclusion = st.checkbox("Model transaction inclusion probability", True,
                                  help="Estimate probability of transaction inclusion at different fee levels")
    if model_tx_inclusion:
        inclusion_thresholds = st.slider("Inclusion probability thresholds", 3, 10, 5, 
                                     help="Number of probability thresholds to calculate")

    # Debug output option
    st.subheader("Debug Options")
    enable_debug = st.checkbox("Enable debug output (print & CSV)", value=False, help="If checked, prints debug info and saves debug CSV during simulation.")

    # Create current configuration dictionary
    current_config = {
        "num_blocks": int(num_blocks),
        "block_time": float(block_time),
        "demand_source": demand_source,
        "use_elasticity": bool(use_elasticity),
        "capacity_gps": float(capacity_gps),
        "target_util": float(target_util),
        "k_percent": float(k_percent),
        "scheme": scheme,
        "base_fee0": float(base_fee0),
        "base_min": float(base_min),
        "base_max": float(base_max),
        "seed": int(seed),
        "erc20_transfer_gas": int(erc20_transfer_gas),
        "model_tx_inclusion": bool(model_tx_inclusion)
    }
    
    # Add conditional parameters based on selections
    if demand_source == "Simulate demand":
        current_config["demand_model"] = demand_model
        current_config["demand_mean"] = float(demand_mean)
        current_config["volatility_ratio"] = float(volatility_ratio)
        if demand_model == "AR(1) process":
            current_config["ar1_persistence"] = float(ar1_persistence)
    else:  # Load from CSV
        current_config["csv_file"] = csv_file
        current_config["csv_gas_column"] = csv_gas_column
        current_config["use_range"] = bool(use_range)
        if use_range:
            current_config["start_index"] = int(start_index)
            if end_index is not None:
                current_config["end_index"] = int(end_index)
    
    if use_elasticity:
        current_config["elasticity"] = float(elasticity)
        current_config["reference_price"] = float(reference_price)
    
    if scheme == "EMA":
        current_config["alpha"] = float(alpha)
    else:
        current_config["win_sec"] = float(win_sec)
    
    if model_tx_inclusion:
        current_config["inclusion_thresholds"] = int(inclusion_thresholds)
    
    # Load configuration if requested
    if uploaded_config is not None and 'load_config' in locals() and load_config:
        # This code would ideally set all the values from the loaded config
        # but for simplicity, we'll just show a notice to reload the page with the new config
        st.sidebar.info("Configuration applied. Please rerun the simulation.")
        # We would need to set each widget's value here, but Streamlit doesn't easily allow this
        # A full implementation would require session state management for each parameter

    # Save configuration if requested
    if save_config:
        st.sidebar.markdown(save_configuration(current_config), unsafe_allow_html=True)

def calculate_ar1_coefficient(series):
    """
    Calculate the AR(1) coefficient for a time series.
    
    Args:
        series: The time series to analyze
        
    Returns:
        The AR(1) coefficient (persistence) of the series
    """
    # Remove any NaN values
    series = series[~np.isnan(series)]
    
    if len(series) <= 1:
        return np.nan
        
    # Calculate the correlation between the series and its lag-1 version
    correlation = np.corrcoef(series[1:], series[:-1])[0, 1]
    return correlation

def calculate_inclusion_probabilities(base_fees, num_thresholds=5):
    """
    Calculate transaction inclusion probabilities at different fee levels.
    
    Args:
        base_fees: Array of base fees from simulation
        num_thresholds: Number of fee thresholds to evaluate
        
    Returns:
        Dictionary with thresholds and corresponding inclusion probabilities
    """
    # Sort fees to analyze distribution
    sorted_fees = np.sort(base_fees)
    
    # Calculate percentiles for analysis
    percentiles = np.linspace(0, 100, num_thresholds + 1)[1:]  # Exclude 0th percentile
    fee_thresholds = np.percentile(sorted_fees, percentiles)
    
    # Calculate inclusion probabilities (probability that a tx with this fee would be included)
    probabilities = {}
    for i, p in enumerate(percentiles):
        threshold = fee_thresholds[i]
        prob = 100 - p  # If you bid at the Xth percentile, you have (100-X)% chance of inclusion
        probabilities[f"{threshold:.8f}"] = f"{prob:.1f}%"
    
    return probabilities

def load_demand_from_csv(csv_file, column_name, num_blocks=None, start_index=0, end_index=None):
    """
    Load gas demand data from a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        column_name: Name of the column containing gas demand data
        num_blocks: Number of blocks to use (will truncate or repeat if necessary)
        start_index: Starting index in the CSV data
        end_index: Ending index in the CSV data (exclusive)
        
    Returns:
        Array of demand blocks from the CSV file
    """
    if not os.path.exists(csv_file):
        st.error(f"CSV file '{csv_file}' not found.")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        if column_name not in df.columns:
            st.error(f"Column '{column_name}' not found in CSV file. Available columns: {', '.join(df.columns)}")
            return None
        
        # Get the full data
        full_data = df[column_name].values
        
        # Apply range selection
        if end_index is not None and end_index > start_index:
            # Ensure end_index doesn't exceed data length
            end_index = min(end_index, len(full_data))
            demand_data = full_data[start_index:end_index]
        else:
            # If end_index is None or invalid, start from start_index
            demand_data = full_data[start_index:]
        
        # If num_blocks is specified and different from selected range, adjust the data to match
        if num_blocks is not None:
            if len(demand_data) > num_blocks:
                # Truncate
                demand_data = demand_data[:num_blocks]
            elif len(demand_data) < num_blocks:
                # Repeat the data to fill the required number of blocks
                repetitions = int(np.ceil(num_blocks / len(demand_data)))
                demand_data = np.tile(demand_data, repetitions)[:num_blocks]
        
        return demand_data
    
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def simulate(enable_debug=False):
    """
    Run the full simulation with demand and price update in a single loop.
    For simulated demand, at each block:
      1. Compute mean demand using current base fee and elasticity
      2. Draw demand (normal or AR(1))
      3. Apply gas limit
      4. Update smoothed gas usage
      5. Update base fee for next block
    For AR(1), use previous demand. For CSV, keep the logic but update price in the same loop.
    Returns:
        Tuple of (base_fees, gas_used, demand_blocks, adjustment_factors, elasticity_factors, base_means, adjusted_means)
    """
    rng = np.random.default_rng(None if seed == 0 else int(seed))
    blocks = int(num_blocks)
    # Output arrays
    base_fees = np.zeros(blocks)
    gas_used = np.zeros(blocks)
    demand_blocks = np.zeros(blocks)
    adjustment_factors = np.zeros(blocks)
    elasticity_factors = np.ones(blocks)
    base_means = np.zeros(blocks)
    adjusted_means = np.zeros(blocks)

    gas_limit_block = capacity_gps * block_time
    g_target_window = capacity_gps * target_util
    # Smoothing state
    if scheme == "Rolling window":
        win_blocks = max(1, int(win_sec / block_time))
        q = deque()
    else:
        ema_hat = g_target_window

    # --- Simulated demand ---
    if demand_source == "Simulate demand":
        base_fee = base_fee0
        prev_demand = None
        debug_rows = []
        for t in range(blocks):
            # 1. Compute mean demand (with elasticity if enabled)
            base_mean = demand_mean * block_time
            base_means[t] = base_mean
            if use_elasticity:
                ref_price = reference_price if reference_price > 0 else 0.00001
                price_ratio = base_fee / ref_price
                elasticity_factor = price_ratio ** elasticity if price_ratio > 0 else 1.0
                mean = base_mean * elasticity_factor
                elasticity_factors[t] = elasticity_factor
            else:
                mean = base_mean
                elasticity_factors[t] = 1.0
            adjusted_means[t] = mean

            # 2. Draw demand
            if demand_model == "Normal distribution":
                std = volatility_ratio * mean
                demand = max(0.0, rng.normal(mean, std))
            elif demand_model == "AR(1) process":
                persistence = ar1_persistence
                std = volatility_ratio * mean
                if t == 0:
                    # Unconditional variance for AR(1)
                    if abs(persistence) < 1:
                        demand = max(0.0, rng.normal(mean, std / np.sqrt(1 - persistence**2)))
                    else:
                        demand = max(0.0, rng.normal(mean, std))
                else:
                    shock = rng.normal(0, std)
                    demand = max(0.0, mean + persistence * (prev_demand - mean) + shock)
                prev_demand = demand
            else:
                raise ValueError(f"Unknown demand_model: {demand_model}")
            demand_blocks[t] = demand

            # 3. Apply gas limit
            gas = min(demand, gas_limit_block)
            gas_used[t] = gas

            # 4. Smoothing
            if scheme == "EMA":
                ema_hat = (1 - alpha) * ema_hat + alpha * gas
                hatG = ema_hat / block_time
            else:
                q.append(gas)
                if len(q) > win_blocks:
                    q.popleft()
                hatG = sum(q) / (len(q) * block_time)

            # 5. Update base fee for next block
            delta = hatG / g_target_window - 1.0
            adjustment_factor = (1 + k_percent / 100 * delta)
            adjustment_factors[t] = adjustment_factor
            base_fees[t] = base_fee
            
            # Debug print and CSV saving only if enabled
            if enable_debug:
                if t < 100:
                    print(f"Block {t}: elasticity_factor={elasticity_factors[t]:.7f}, mean_demand={mean:.2f}, delta={delta:.4f}, adjustment_factor={adjustment_factor:.4f}, base_fee={base_fee:.9f}")
                debug_rows.append({
                    'block': t,
                    'mean': mean,
                    'elasticity_factor': elasticity_factors[t],
                    'gas_used': gas,
                    'adjustment_factor': adjustment_factor,
                    'base_fee': base_fee,
                    'delta': delta
                })
            base_fee = base_fee * adjustment_factor
            base_fee = min(base_max, max(base_min, base_fee))
        # Save debug data to CSV with alpha or window in filename, only if enabled
        if enable_debug:
            if scheme == "EMA":
                debug_filename = f"DEBUG_alpha_{alpha}.csv"
            else:
                debug_filename = f"DEBUG_win_{win_sec}.csv"
            debug_df = pd.DataFrame(debug_rows)
            debug_df.to_csv(debug_filename, index=False)
        return base_fees, gas_used, demand_blocks, adjustment_factors, elasticity_factors, base_means, adjusted_means

    # --- CSV demand ---
    elif demand_source == "Load from CSV":
        if use_range:
            demand_blocks = load_demand_from_csv(
                csv_file, 
                csv_gas_column, 
                blocks, 
                start_index=start_index, 
                end_index=end_index
            )
        else:
            demand_blocks = load_demand_from_csv(
                csv_file, 
                csv_gas_column, 
                blocks
            )
        if demand_blocks is None:
            return None, None, None, None, None, None, None
        base_fee = base_fee0
        # For CSV, just use the loaded demand, but update price in the same loop
        if scheme == "Rolling window":
            win_blocks = max(1, int(win_sec / block_time))
            q = deque()
        else:
            ema_hat = g_target_window
        for t in range(blocks):
            demand = demand_blocks[t]
            base_means[t] = np.mean(demand_blocks)  # Not meaningful, but for compatibility
            adjusted_means[t] = base_means[t]
            # Elasticity factor is not meaningful for CSV, but keep for compatibility
            elasticity_factors[t] = 1.0
            # Apply gas limit
            gas = min(demand, gas_limit_block)
            gas_used[t] = gas
            # Smoothing
            if scheme == "EMA":
                ema_hat = (1 - alpha) * ema_hat + alpha * gas
                hatG = ema_hat / block_time
            else:
                q.append(gas)
                if len(q) > win_blocks:
                    q.popleft()
                hatG = sum(q) / (len(q) * block_time)
            # Update base fee for next block
            delta = hatG / g_target_window - 1.0
            adjustment_factor = (1 + k_percent / 100 * delta)
            adjustment_factors[t] = adjustment_factor
            base_fees[t] = base_fee
            base_fee = base_fee * adjustment_factor
            base_fee = min(base_max, max(base_min, base_fee))
        return base_fees, gas_used, demand_blocks, adjustment_factors, elasticity_factors, base_means, adjusted_means

    else:
        st.error("Unknown demand source.")
        return None, None, None, None, None, None, None

if st.button("Run simulation"):
    bf, gu, demand, adj_factors, elasticity_factors, base_means, adjusted_means = simulate(enable_debug=enable_debug)
    
    if bf is None:  # Error loading CSV
        st.error("Simulation failed. Please check the CSV file and settings.")
    else:
        # Calculate ERC-20 transfer costs
        transfer_costs = bf * erc20_transfer_gas
        
        # Calculate statistics for transfer costs
        transfer_stats = {
            "Min": np.min(transfer_costs),
            "25%": np.percentile(transfer_costs, 25),
            "50% (Median)": np.percentile(transfer_costs, 50),
            "Mean": np.mean(transfer_costs),
            "75%": np.percentile(transfer_costs, 75),
            "99%": np.percentile(transfer_costs, 99),
            "Max": np.max(transfer_costs)
        }
        
        # Calculate elasticity statistics if enabled
        if 'use_elasticity' in locals() and use_elasticity:
            # Store elasticity factors for visualization
            elasticity_factors_for_display = elasticity_factors
            
            # The original demand now already includes elasticity effects in its mean
            # So we don't multiply by elasticity factors again
            adjusted_demand = demand  # demand is already elasticity-adjusted
            
            # Calculate gas limit for comparison
            gas_limit = capacity_gps * block_time
            
            # Calculate percent of demand that's greater than gas limit
            pct_over_limit = np.mean(demand > gas_limit) * 100
            
            # Display elasticity factor statistics
            elasticity_stats = {
                "Min": np.min(elasticity_factors),
                "25%": np.percentile(elasticity_factors, 25),
                "50% (Median)": np.percentile(elasticity_factors, 50),
                "Mean": np.mean(elasticity_factors),
                "75%": np.percentile(elasticity_factors, 75),
                "Max": np.max(elasticity_factors)
            }
        
        # Calculate AR(1) coefficients
        transfer_ar1 = calculate_ar1_coefficient(transfer_costs)
        demand_ar1 = calculate_ar1_coefficient(demand)
        gas_used_ar1 = calculate_ar1_coefficient(gu)
        
        # Calculate volatility metrics
        price_volatility = {
            "Standard Deviation": np.std(bf),
            "Coefficient of Variation": np.std(bf) / np.mean(bf) if np.mean(bf) > 0 else np.nan,
            "Min-Max Range": np.max(bf) - np.min(bf),
            "Interquartile Range": np.percentile(bf, 75) - np.percentile(bf, 25)
        }
        
        # Calculate transaction inclusion probabilities if enabled
        if 'model_tx_inclusion' in locals() and model_tx_inclusion:
            inclusion_probs = calculate_inclusion_probabilities(bf, num_thresholds=inclusion_thresholds)
        
        # Save results to session state for downloads
        st.session_state.simulation_results = {
            'base_fees': bf,
            'gas_used': gu,
            'demand': demand,
            'adjustment_factors': adj_factors,
            'elasticity_factors': elasticity_factors,
            'transfer_costs': transfer_costs,
            'base_means': base_means,
            'adjusted_means': adjusted_means,
            'config': current_config
        }
        
        # Create results dataframe for export
        results_df = pd.DataFrame({
            'Block': range(len(bf)),
            'Base_Fee': bf,
            'Gas_Used': gu,
            'Original_Demand': demand,
            'Adjustment_Factor': adj_factors,
            'Elasticity_Factor': elasticity_factors,
            'ERC20_Transfer_Cost': transfer_costs,
            'Base_Mean': base_means,
            'Adjusted_Mean': adjusted_means
        })
        
        # Add download buttons
        st.subheader("Export Results")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.markdown(get_csv_download_link(results_df, 
                                        f"fee_sim_results_{date_str}.csv", 
                                        "Download Simulation Data (CSV)"), 
                    unsafe_allow_html=True)
        
        # Left column for graphs
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Base fee trajectory
            st.subheader("Base-fee trajectory")
            fig, ax = plt.subplots()
            ax.plot(bf)
            ax.set_xlabel("Block")
            ax.set_ylabel("Base-fee [USDC-cent/gas]")
            st.pyplot(fig, clear_figure=True)
            
            # Option to download the plot
            st.markdown(get_plot_download_link(fig, 
                                          f"base_fee_trajectory_{date_str}.png", 
                                          "Download Plot"), 
                    unsafe_allow_html=True)
            
            # Gas price distribution histogram
            st.subheader("Gas Price Distribution")
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(bf, bins=30, kde=True, ax=ax_hist)
            ax_hist.set_xlabel("Base-fee [USDC-cent/gas]")
            ax_hist.set_ylabel("Frequency")
            
            # Add vertical lines for key percentiles
            ax_hist.axvline(np.percentile(bf, 25), color='r', linestyle='--', alpha=0.7, label="25th percentile")
            ax_hist.axvline(np.percentile(bf, 50), color='g', linestyle='--', alpha=0.7, label="Median")
            ax_hist.axvline(np.percentile(bf, 75), color='b', linestyle='--', alpha=0.7, label="75th percentile")
            ax_hist.legend()
            
            st.pyplot(fig_hist, clear_figure=True)
            
            # Option to download the plot
            st.markdown(get_plot_download_link(fig_hist, 
                                          f"gas_price_distribution_{date_str}.png", 
                                          "Download Plot"), 
                    unsafe_allow_html=True)
            
            # Elasticity factor plot (if price elasticity is enabled)
            if 'use_elasticity' in locals() and use_elasticity:
                st.subheader("Price Elasticity Factor")
                fig_elastic, ax_elastic = plt.subplots()
                ax_elastic.plot(elasticity_factors, color='red')
                ax_elastic.axhline(y=1.0, color='k', linestyle='--', 
                                 label='No elasticity effect (factor = 1.0)')
                ax_elastic.set_xlabel("Block")
                ax_elastic.set_ylabel("Elasticity Factor (price_ratio ^ elasticity)")
                ax_elastic.set_title(f"Elasticity Coefficient: {elasticity} (Applied to Current Period Mean)")
                ax_elastic.legend()
                st.pyplot(fig_elastic, clear_figure=True)
                
                # Option to download the plot
                st.markdown(get_plot_download_link(fig_elastic, 
                                              f"elasticity_factors_{date_str}.png", 
                                              "Download Plot"), 
                        unsafe_allow_html=True)
                
                # Plot showing demand (which already includes elasticity effect on the mean)
                st.subheader("Demand (With Current-Period Elasticity)")
                fig_demand_comp, ax_demand_comp = plt.subplots()
                # Demand (already includes elasticity effects in the mean)
                ax_demand_comp.plot(demand, color='blue', label='Demand (with elasticity applied)')
                
                # Gas used (after limit constraint)
                ax_demand_comp.plot(gu, color='green', linestyle=':', 
                                   label='Gas used (after limit constraint)')
                
                # Add a line for the gas limit
                gas_limit = capacity_gps * block_time
                ax_demand_comp.axhline(y=gas_limit, color='k', linestyle='--', 
                                      label=f'Gas limit: {gas_limit:.0f}')
                
                ax_demand_comp.set_xlabel("Block")
                ax_demand_comp.set_ylabel("Gas")
                ax_demand_comp.legend()
                st.pyplot(fig_demand_comp, clear_figure=True)
                
                # Option to download the plot
                st.markdown(get_plot_download_link(fig_demand_comp, 
                                              f"demand_comparison_{date_str}.png", 
                                              "Download Plot"), 
                        unsafe_allow_html=True)
            
            # ERC-20 transfer cost
            st.subheader(f"ERC-20 Transfer Cost ({erc20_transfer_gas:,} gas units)")
            fig_transfer, ax_transfer = plt.subplots()
            ax_transfer.plot(transfer_costs, color='green')
            ax_transfer.set_xlabel("Block")
            ax_transfer.set_ylabel("Transfer Cost [USDC-cent]")
            st.pyplot(fig_transfer, clear_figure=True)
            
            # Option to download the plot
            st.markdown(get_plot_download_link(fig_transfer, 
                                          f"erc20_transfer_cost_{date_str}.png", 
                                          "Download Plot"), 
                    unsafe_allow_html=True)
            
            # Gas used per block
            st.subheader("Gas used per block")
            fig2, ax2 = plt.subplots()
            ax2.plot(gu)
            ax2.set_xlabel("Block")
            ax2.set_ylabel("Gas used")
            st.pyplot(fig2, clear_figure=True)
            
            # Option to download the plot
            st.markdown(get_plot_download_link(fig2, 
                                          f"gas_used_{date_str}.png", 
                                          "Download Plot"), 
                    unsafe_allow_html=True)
            
            # Demand blocks visualization
            st.subheader("Demand blocks")
            fig3, ax3 = plt.subplots()
            ax3.plot(demand, color='orange', label='Original demand')
            
            # If elasticity is enabled, show the actual gas used for comparison
            if 'use_elasticity' in locals() and use_elasticity:
                ax3.plot(gu, color='blue', alpha=0.7, linestyle='--', 
                         label='Actual gas used (price-adjusted)')
            
            ax3.set_xlabel("Block")
            ax3.set_ylabel("Demand (gas)")
            
            # Add a line for the gas limit
            gas_limit = capacity_gps * block_time
            ax3.axhline(y=gas_limit, color='r', linestyle='--', 
                        label=f'Gas limit per block: {gas_limit:.0f}')
            ax3.legend()
            
            st.pyplot(fig3, clear_figure=True)
            
            # Option to download the plot
            st.markdown(get_plot_download_link(fig3, 
                                          f"demand_blocks_{date_str}.png", 
                                          "Download Plot"), 
                    unsafe_allow_html=True)
            
            # Add plot for adjustment factors
            st.subheader("Base Fee Adjustment Factor")
            fig_adj, ax_adj = plt.subplots()
            ax_adj.plot(adj_factors, color='purple')
            ax_adj.axhline(y=1.0, color='r', linestyle='--', 
                          label='No adjustment (factor = 1.0)')
            ax_adj.set_xlabel("Block")
            ax_adj.set_ylabel("Adjustment Factor (1 + k% * delta)")
            ax_adj.legend()
            st.pyplot(fig_adj, clear_figure=True)
            
            # Option to download the plot
            st.markdown(get_plot_download_link(fig_adj, 
                                          f"adjustment_factors_{date_str}.png", 
                                          "Download Plot"), 
                    unsafe_allow_html=True)
            
            # Add diagnostic plot for base_mean vs. adjusted_mean
            if 'use_elasticity' in locals() and use_elasticity:
                st.subheader("Mean Demand Diagnostic")
                fig_mean, ax_mean = plt.subplots()
                ax_mean.plot(base_means, color='blue', label='Base Mean (Before Elasticity)')
                ax_mean.plot(adjusted_means, color='red', linestyle='--', label='Adjusted Mean (After Elasticity)')
                # ax_mean.plot(bf, color='green', linestyle=':', alpha=0.7, label='Base Fee')
                
                # Add a reference line for the average adjusted mean
                ax_mean.axhline(y=np.mean(adjusted_means), color='darkred', linestyle=':', 
                             label=f'Avg Adjusted Mean: {np.mean(adjusted_means):.0f}')
                
                # Add a reference line for the base mean
                ax_mean.axhline(y=np.mean(base_means), color='darkblue', linestyle=':', 
                             label=f'Base Mean: {np.mean(base_means):.0f}')
                
                ax_mean.set_xlabel("Block")
                ax_mean.set_ylabel("Mean Demand (gas/block)")
                ax_mean.legend(loc='upper right')
                ax_mean.set_title("Base Mean vs. Current-Period Elasticity-Adjusted Mean")
                st.pyplot(fig_mean, clear_figure=True)
                
                # Option to download the plot
                st.markdown(get_plot_download_link(fig_mean, 
                                              f"mean_demand_diagnostic_{date_str}.png", 
                                              "Download Plot"), 
                        unsafe_allow_html=True)
            
            # Display information about price elasticity
            if 'use_elasticity' in locals() and use_elasticity:
                elasticity_info = f"Price elasticity enabled (coefficient: {elasticity}, applied to current period mean with volatility proportional to adjusted mean)"
                st.info(elasticity_info)
        
        # Right column for statistics
        with col2:
            st.subheader("ERC-20 Transfer Cost Statistics")
            
            # Create a DataFrame for better display
            stats_df = pd.DataFrame(list(transfer_stats.items()), 
                                   columns=['Statistic', 'Value [USDC-cent]'])
            
            # Format the values
            stats_df['Value [USDC-cent]'] = stats_df['Value [USDC-cent]'].apply(
                lambda x: f"{x:.8f}" if x < 0.0001 else f"{x:.6f}")
            
            # Display the statistics
            st.table(stats_df)
            
            # Display elasticity factor statistics if enabled
            if 'use_elasticity' in locals() and use_elasticity and 'elasticity_stats' in locals():
                st.subheader("Elasticity Factor Statistics")
                
                # First display the key elasticity metrics requested at the top
                key_metrics = {
                    "% of demand > gas limit": f"{pct_over_limit:.2f}%",
                    "Mean elasticity factor applied to demand": f"{np.mean(elasticity_factors):.4f}"
                }
                key_df = pd.DataFrame(list(key_metrics.items()),
                                    columns=['Key Metric', 'Value'])
                st.table(key_df)
                
                # Then display the detailed elasticity factor statistics
                elast_df = pd.DataFrame(list(elasticity_stats.items()), 
                                      columns=['Statistic', 'Value'])
                # Format the values (elasticity factors are unitless multipliers)
                elast_df['Value'] = elast_df['Value'].apply(
                    lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                st.table(elast_df)
                
                # Calculate and show elasticity effect
                elasticity_effect = {
                    "Mean demand reduction": f"{(1 - np.mean(elasticity_factors)) * 100:.2f}%",
                    "Max demand reduction": f"{(1 - np.min(elasticity_factors)) * 100:.2f}%",
                    "Impact description": "Elasticity is now applied to the mean when generating demand"
                }
                effect_df = pd.DataFrame(list(elasticity_effect.items()),
                                       columns=['Metric', 'Value'])
                st.table(effect_df)
            
            # Display volatility metrics
            st.subheader("Price Volatility Metrics")
            vol_df = pd.DataFrame(list(price_volatility.items()), 
                                columns=['Metric', 'Value'])
            # Format the values
            vol_df['Value'] = vol_df['Value'].apply(
                lambda x: f"{x:.8f}" if abs(x) < 0.0001 else f"{x:.6f}" if not pd.isna(x) else "N/A")
            st.table(vol_df)
            
            # Display transaction inclusion probabilities if enabled
            if 'model_tx_inclusion' in locals() and model_tx_inclusion and 'inclusion_probs' in locals():
                # This section has been removed as per user's request
                pass
            
            # Display AR(1) coefficient estimates
            st.subheader("Time Series Properties")
            ar1_stats = {
                "ERC-20 Transfer Cost AR(1) coefficient": f"{transfer_ar1:.4f}",
                "Demand AR(1) coefficient": f"{demand_ar1:.4f}",
                "Gas Used AR(1) coefficient": f"{gas_used_ar1:.4f}"
            }
            
            # Add elasticity factor AR(1) if enabled
            if 'use_elasticity' in locals() and use_elasticity:
                elasticity_ar1 = calculate_ar1_coefficient(elasticity_factors)
                ar1_stats["Elasticity Factor AR(1) coefficient"] = f"{elasticity_ar1:.4f}"
            
            ar1_df = pd.DataFrame(list(ar1_stats.items()), 
                                 columns=['Metric', 'Value'])
            st.table(ar1_df)
            
            # Display information about the demand source
            if demand_source == "Load from CSV":
                if use_range:
                    range_info = f" (using rows {start_index} to {end_index if end_index else 'end'})"
                else:
                    range_info = ""
                st.info(f"Demand loaded from CSV file: {csv_file}{range_info}")
        
        st.success("Simulation finished")
