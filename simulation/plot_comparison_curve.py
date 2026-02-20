import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def plot_comparison():
    # --- 1. Load Data ---
    try:
        # Load the two simulation CSVs
        df_random = pd.read_csv("daily_summary_DistanceOnly.csv")
        df_baseline = pd.read_csv("daily_summary_Baseline.csv")
        
        # Load and process the real data
        df_real_raw = pd.read_csv("calibration_targets_30days.csv")
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure all required CSV files are in the directory.")
        return

    # --- 2. Process Data ---
    start_date = datetime(2017, 3, 21)
    
    # --- Process Baseline Simulation Data ---
    df_baseline['date'] = pd.to_datetime(df_baseline['date'])
    df_baseline['day'] = (df_baseline['date'] - start_date).dt.days
    
    # --- Process Random (Distance-Only) Simulation Data ---
    df_random['date'] = pd.to_datetime(df_random['date'])
    df_random['day'] = (df_random['date'] - start_date).dt.days

    # --- Process Real Data ---
    df_real_positive = df_real_raw[df_real_raw['state'] == 'POSITIVO'].copy()
    df_real_positive['date'] = pd.to_datetime(df_real_positive['date'])
    df_real_positive['day'] = (df_real_positive['date'] - start_date).dt.days
    
    real_daily_counts = df_real_positive.groupby('day').size().reset_index(name='new_infections')
    real_daily_counts = real_daily_counts.sort_values('day')
    real_daily_counts['cumulative_infected'] = real_daily_counts['new_infections'].cumsum()
    
    # --- 3. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each curve using the consistent column names
    ax.plot(df_baseline['day'], df_baseline['cumulative_cases'], label='Simulation (Baseline)', color='royalblue', linewidth=2)
    ax.plot(df_random['day'], df_random['cumulative_cases'], label='Random Spread (Distance-Only)', color='darkorange', linestyle='--', linewidth=2)
    ax.plot(real_daily_counts['day'], real_daily_counts['cumulative_infected'], label='Real Data', color='forestgreen', marker='o', linestyle='-', markersize=4)

    # --- 4. Styling ---
    ax.set_title('Comparison of Epidemic Curves', fontsize=16)
    ax.set_xlabel('Days Since Initial Infection', fontsize=12)
    ax.set_ylabel('Cumulative Infected Trees', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True)
    
    max_day = max(df_baseline['day'].max(), df_random['day'].max(), real_daily_counts['day'].max())
    ax.set_xlim(0, max_day + 10)
    ax.set_ylim(0)
    ax.set_ylim(0)

    # --- 5. Save Figure ---
    output_filename = 'Fig_Comparison_Random_Real_Sim.pdf'
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    plot_comparison()
