import simulation_engine
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
ITERATIONS = 1000
BETA = 0.1
KERNEL_K = 1.5
DELTA = 175
RANKED_NODES_FILE = 'ranked_nodes_by_betweenness.csv'
ORCHARD_FILE = "filtered_trees_2017.csv"
TARGET_FILE_30 = "calibration_targets_30days.csv"

SEASON = {
    1: 0.0, 2: 0.0, 3: 0.01, 4: 0.05, 5: 0.20, 6: 0.72,
    7: 0.57, 8: 0.45, 9: 0.54, 10: 0.32, 11: 0.02, 12: 0.0
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_top_nodes_to_remove(filename, count):
    """Reads the ranked CSV and returns the top 'count' node IDs."""
    nodes_to_remove = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < count:
                    cleaned_row = {k.strip(): v for k, v in row.items()}
                    nodes_to_remove.append(cleaned_row['Node ID'])
                else: break
        return nodes_to_remove
    except FileNotFoundError:
        print(f"ERROR: {filename} not found.")
        return None

def get_positive_nodes(filename):
    """Loads ground truth data from a file."""
    if not os.path.exists(filename): return set()
    df = pd.read_csv(filename)
    df['node_id'] = df['node_id'].astype(str)
    if 'state' in df.columns:
        return set(df[df['state'].str.strip().str.upper() == 'POSITIVO']['node_id'])
    elif 'infected' in df.columns:
        return set(df[df['infected'] == 1]['node_id'])
    return set()

def run_monte_carlo_scenario(description, exclude_nodes_list, node_ids_for_hits):
    """Runs a Monte Carlo simulation and returns rich data for plotting."""
    print(f"\nRunning Scenario: {description} ({ITERATIONS} iterations)...")
    
    all_runs_curves = []
    all_runs_daily_new = []
    node_hits = {nid: 0 for nid in node_ids_for_hits}
    final_infection_sizes = []  # CORRECT: List to store final size of each run

    for _ in tqdm(range(ITERATIONS), desc=description):
        log = simulation_engine.run_simulation_full(
            beta_multiplier=BETA, kernel_multiplier=KERNEL_K, season_profile=SEASON,
            file_path=ORCHARD_FILE, exclude_nodes=exclude_nodes_list
        )
        
        final_infection_sizes.append(len(log)) # CORRECT: Record the final size of this run

        # Data for plots
        infected_doy = [d.timetuple().tm_yday for d in log.values()]
        
        daily_new_infections = [0] * 365
        for doy in infected_doy:
            if 1 <= doy <= 365:
                daily_new_infections[doy-1] += 1
        all_runs_daily_new.append(daily_new_infections)
        
        daily_counts = np.cumsum(daily_new_infections).tolist()
        all_runs_curves.append(daily_counts)
        
        for infected_node in log.keys():
            if str(infected_node) in node_hits:
                node_hits[str(infected_node)] += 1
                
    avg_infected = np.mean(final_infection_sizes) # CORRECT: Calculate mean of the correct list
    
    return {'curves': all_runs_curves, 'daily_new': all_runs_daily_new, 'hits': node_hits, 'avg_total': avg_infected}

def generate_output_figures(scenario_name, results, df_map, is_baseline=False):
    """Generates and saves all plots for a given scenario's results."""
    print(f"\nGenerating figures for '{scenario_name}' scenario...")
    
    # --- FIG 1: CURVE ---
    mean_curve = np.mean(results['curves'], axis=0)
    std_curve = np.std(results['curves'], axis=0)
    days = np.arange(1, 366)
    plt.figure(figsize=(10, 6))
    plt.fill_between(days, mean_curve - std_curve, mean_curve + std_curve, color='blue', alpha=0.15)
    plt.plot(days, mean_curve, color='darkblue', linewidth=2, label=f'Mean of {ITERATIONS} runs')
    plt.title(f"Epidemic Progression - {scenario_name}", fontsize=14)
    plt.xlabel("Day of Year", fontsize=12)
    plt.ylabel("Cumulative Infected Trees", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'Fig1_Curve_{scenario_name}.pdf')
    plt.close()
    print(f"   -> Saved Fig1_Curve_{scenario_name}.pdf")

    # --- FIG 2: MAPS ---
    df_results = df_map.copy()
    df_results['inf_prob'] = df_results['node_id'].map(results['hits']).fillna(0) / ITERATIONS
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(df_results['longitude'], df_results['latitude'], c=df_results['inf_prob'], cmap='YlOrRd', s=50, vmin=0, vmax=1)
    plt.colorbar(sc, label='Infection Probability')
    plt.title(f"Simulated Risk Map - {scenario_name}", fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'Fig2_RiskMap_{scenario_name}.pdf')
    plt.close()
    print(f"   -> Saved Fig2_RiskMap_{scenario_name}.pdf")

    # --- FIG 3: VALIDATION (Only for Baseline) ---
    if is_baseline:
        truth_30d = get_positive_nodes(TARGET_FILE_30)
        truth_full = get_positive_nodes(ORCHARD_FILE)
        predicted_infected = set(df_results[df_results['inf_prob'] >= 0.20]['node_id'])
        inter_30d = len(truth_30d.intersection(predicted_infected))
        union_30d = len(truth_30d.union(predicted_infected))
        score_30d = inter_30d / union_30d if union_30d > 0 else 0
        inter_full = len(truth_full.intersection(predicted_infected))
        union_full = len(truth_full.union(predicted_infected))
        score_full = inter_full / union_full if union_full > 0 else 0

        plt.figure(figsize=(8, 6))
        bars = plt.bar(['vs 30-Day Target', 'vs Full Year Data'], [score_30d, score_full], color=['#377eb8', '#4daf4a'], width=0.5)
        plt.ylim(0, 1.0)
        plt.ylabel('Jaccard Similarity Index')
        plt.title('Model Validation (Baseline vs Ground Truth)')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'Fig3_Validation_{scenario_name}.pdf')
        plt.close()
        print(f"   -> Saved Fig3_Validation_{scenario_name}.pdf")

def generate_csv_output(scenario_name, results):
    """Generates and saves a CSV file with daily and cumulative infection data."""
    print(f"\nGenerating CSV summary for '{scenario_name}' scenario...")
    
    mean_daily_new_cases = np.mean(results['daily_new'], axis=0)
    mean_cumulative_cases = np.mean(results['curves'], axis=0)
    
    # Create a date range for the year. The simulation runs for 365 days.
    # Using 2017 as the base year, consistent with the input file naming.
    dates = pd.to_datetime('2017-01-01') + pd.to_timedelta(np.arange(365), 'D')
    
    df_output = pd.DataFrame({
        'date': dates,
        'mean_daily_new_cases': mean_daily_new_cases,
        'cumulative_cases': mean_cumulative_cases
    })
    
    output_filename = f'daily_summary_{scenario_name}.csv'
    df_output.to_csv(output_filename, index=False, date_format='%Y-%m-%d')
    print(f"   -> Saved {output_filename}")

def generate_combined_curve_plot(baseline_results, betweenness_results, closeness_results):
    """Generates a single plot comparing the epidemic curves of all three scenarios."""
    print("\nGenerating combined epidemic curve figure...")
    
    plt.figure(figsize=(12, 8))
    days = np.arange(1, 366)

    # Plot Baseline
    if baseline_results:
        mean_curve_baseline = np.mean(baseline_results['curves'], axis=0)
        plt.plot(days, mean_curve_baseline, color='black', linestyle='--', linewidth=2, label='Baseline')

    # Plot Betweenness
    if betweenness_results:
        mean_curve_betweenness = np.mean(betweenness_results['curves'], axis=0)
        plt.plot(days, mean_curve_betweenness, color='#1f77b4', linewidth=2, label=f'Betweenness (Top {DELTA} Removed)')

    # Plot Closeness
    if closeness_results:
        mean_curve_closeness = np.mean(closeness_results['curves'], axis=0)
        plt.plot(days, mean_curve_closeness, color='#ff7f0e', linewidth=2, label=f'Closeness (Top {DELTA} Removed)')

    plt.title("Comparison of Epidemic Progression", fontsize=16)
    plt.xlabel("Day of Year", fontsize=12)
    plt.ylabel("Cumulative Infected Trees", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig1_Epidemic_Curve_centrality_test.pdf')
    plt.close()
    print("   -> Saved Fig1_Epidemic_Curve_centrality_test.pdf")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    print("--- Final Comparison with Full Plotting (Corrected) ---")
    
    df_map = pd.read_csv(ORCHARD_FILE).drop_duplicates(subset=['node_id']).reset_index(drop=True)
    df_map['node_id'] = df_map['node_id'].astype(str)
    all_node_ids = df_map['node_id'].tolist()
    
    # --- BASELINE ---
    baseline_results = run_monte_carlo_scenario("Baseline", [], all_node_ids)
    generate_output_figures("Baseline", baseline_results, df_map, is_baseline=True)
    generate_csv_output("Baseline", baseline_results)

    # --- BETWEENNESS ---
    nodes_to_exclude_betweenness = get_top_nodes_to_remove('ranked_nodes_by_betweenness.csv', DELTA)
    if nodes_to_exclude_betweenness is None: return
    betweenness_results = run_monte_carlo_scenario(f"Betweenness (Top {DELTA} Removed)", nodes_to_exclude_betweenness, all_node_ids)
    generate_output_figures("Betweenness", betweenness_results, df_map, is_baseline=False)
    generate_csv_output("Betweenness", betweenness_results)

    # --- CLOSENESS ---
    closeness_results = None  # Initialize to handle case where file is not found
    RANKED_NODES_FILE_CLOSENESS = 'ranked_nodes_by_closeness.csv' 
    nodes_to_exclude_closeness = get_top_nodes_to_remove(RANKED_NODES_FILE_CLOSENESS, DELTA)
    if nodes_to_exclude_closeness is not None:
        closeness_results = run_monte_carlo_scenario(f"Closeness (Top {DELTA} Removed)", nodes_to_exclude_closeness, all_node_ids)
        generate_output_figures("Closeness", closeness_results, df_map, is_baseline=False)
        generate_csv_output("Closeness", closeness_results)
    else:
        print(f"\nSkipping Closeness scenario because {RANKED_NODES_FILE_CLOSENESS} was not found.")


    print("\n--- FINAL COMPARISON SUMMARY (AVERAGE OF {} RUNS) ---".format(ITERATIONS))
    print(f"Average Baseline Spread: {baseline_results['avg_total']:.2f} infected nodes")
    print(f"Average Betweenness Spread:  {betweenness_results['avg_total']:.2f} infected nodes")
    if closeness_results:
        print(f"Average Closeness Spread:  {closeness_results['avg_total']:.2f} infected nodes")

    if baseline_results['avg_total'] > 0:
        reduction_betweenness = ((baseline_results['avg_total'] - betweenness_results['avg_total']) / baseline_results['avg_total']) * 100
        print(f"\nBy removing the top {DELTA} nodes (betweenness), we achieved an average reduction of {reduction_betweenness:.2f}%")
        if closeness_results:
            reduction_closeness = ((baseline_results['avg_total'] - closeness_results['avg_total']) / baseline_results['avg_total']) * 100
            print(f"By removing the top {DELTA} nodes (closeness), we achieved an average reduction of {reduction_closeness:.2f}%")

    # --- GENERATE COMBINED PLOT ---
    generate_combined_curve_plot(baseline_results, betweenness_results, closeness_results)

    print("--------------------------------------------------")

if __name__ == "__main__":
    main()