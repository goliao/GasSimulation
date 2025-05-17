import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the CSV files
file_005 = 'DEBUG_alpha_0.05.csv'
file_1 = 'DEBUG_alpha_1.0.csv'
df_005 = pd.read_csv(file_005)
df_1 = pd.read_csv(file_1)

# Variables to plot (exclude 'block')
variables = [col for col in df_005.columns if col != 'block']

with PdfPages('debug_comparison_plots.pdf') as pdf:
    for var in variables:
        plt.figure(figsize=(10, 6))
        plt.plot(df_005['block'], df_005[var], label='alpha=0.05')
        plt.plot(df_1['block'], df_1[var], label='alpha=1.0')
        plt.xlabel('block')
        plt.ylabel(var)
        plt.title(f'Comparison of {var} for alpha=0.05 and alpha=1.0')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print('All plots saved to debug_comparison_plots.pdf')
