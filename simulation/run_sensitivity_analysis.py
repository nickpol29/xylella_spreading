import simulation_engine
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
ITERATIONS = 1000
BETA = 0.1
KERNEL_K = 1.5
DELTAS = [0, 35, 70, 105, 140, 175]
ORCHARD_FILE = "filtered_trees_2017.csv"
TARGET_FILE_30 = "calibration_targets_30days.csv"
OUTPUT_DIR = "sensitivity_analysis_results"

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
    final_infection_sizes = []

    for _ in tqdm(range(ITERATIONS), desc=description):
        log = simulation_engine.run_simulation_full(
            beta_multiplier=BETA, kernel_multiplier=KERNEL_K, season_profile=SEASON,
            file_path=ORCHARD_FILE, exclude_nodes=exclude_nodes_list
        )
        
        final_infection_sizes.append(len(log))

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
                
    avg_infected = np.mean(final_infection_sizes)
    
    return {'curves': all_runs_curves, 'daily_new': all_runs_daily_new, 'hits': node_hits, 'avg_total': avg_infected}

def generate_output_figures(scenario_name, results, df_map, output_dir, is_baseline=False):
    """Generates and saves all plots for a given scenario's results."""
    print(f"\nGenerating figures for '{scenario_name}' in '{output_dir}'...")
    
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
    plt.savefig(os.path.join(output_dir, f'Fig1_Curve_{scenario_name}.pdf'))
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
    plt.savefig(os.path.join(output_dir, f'Fig2_RiskMap_{scenario_name}.pdf'))
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
        plt.savefig(os.path.join(output_dir, f'Fig3_Validation_{scenario_name}.pdf'))
        plt.close()
        print(f"   -> Saved Fig3_Validation_{scenario_name}.pdf")

def generate_csv_output(scenario_name, results, output_dir):
    """Generates and saves a CSV file with daily and cumulative infection data."""
    print(f"\nGenerating CSV summary for '{scenario_name}' in '{output_dir}'...")
    
    mean_daily_new_cases = np.mean(results['daily_new'], axis=0)
    mean_cumulative_cases = np.mean(results['curves'], axis=0)
    
    dates = pd.to_datetime('2017-01-01') + pd.to_timedelta(np.arange(365), 'D')
    
    df_output = pd.DataFrame({
        'date': dates,
        'mean_daily_new_cases': mean_daily_new_cases,
        'cumulative_cases': mean_cumulative_cases
    })
    
    output_filename = os.path.join(output_dir, f'daily_summary_{scenario_name}.csv')
    df_output.to_csv(output_filename, index=False, date_format='%Y-%m-%d')
    print(f"   -> Saved {os.path.basename(output_filename)}")

def generate_sensitivity_comparison_plot(all_results, deltas, output_dir):
    """Generates plots comparing epidemic curves for each centrality metric across different delta values."""
    print("\nGenerating sensitivity analysis comparison plots...")
    
    days = np.arange(1, 366)
    baseline_results = all_results['Baseline_Delta_0']
    mean_curve_baseline = np.mean(baseline_results['curves'], axis=0)
    
    # Define colors for the delta values
    colors = plt.cm.viridis(np.linspace(0, 1, len(deltas)))
    
    # --- Plot for Betweenness ---
    plt.figure(figsize=(12, 8))
    plt.plot(days, mean_curve_baseline, color='black', linestyle='--', linewidth=2.5, label='Baseline (Delta=0)')
    
    for i, delta in enumerate(deltas):
        if delta == 0: continue
        scenario_name = f'Betweenness_Delta_{delta}'
        if scenario_name in all_results:
            results = all_results[scenario_name]
            mean_curve = np.mean(results['curves'], axis=0)
            plt.plot(days, mean_curve, color=colors[i], linewidth=2, label=f'Delta = {delta}')

    plt.title("Sensitivity Analysis: Epidemic Progression vs. Delta (Betweenness)", fontsize=16)
    plt.xlabel("Day of Year", fontsize=12)
    plt.ylabel("Cumulative Infected Trees", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Sensitivity_Comparison_Betweenness.pdf'))
    plt.close()
    print("   -> Saved Sensitivity_Comparison_Betweenness.pdf")

    # --- Plot for Closeness ---
    plt.figure(figsize=(12, 8))
    plt.plot(days, mean_curve_baseline, color='black', linestyle='--', linewidth=2.5, label='Baseline (Delta=0)')
    
    for i, delta in enumerate(deltas):
        if delta == 0: continue
        scenario_name = f'Closeness_Delta_{delta}'
        if scenario_name in all_results:
            results = all_results[scenario_name]
            mean_curve = np.mean(results['curves'], axis=0)
            plt.plot(days, mean_curve, color=colors[i], linewidth=2, label=f'Delta = {delta}')

    plt.title("Sensitivity Analysis: Epidemic Progression vs. Delta (Closeness)", fontsize=16)
    plt.xlabel("Day of Year", fontsize=12)
    plt.ylabel("Cumulative Infected Trees", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Sensitivity_Comparison_Closeness.pdf'))
    plt.close()
    print("   -> Saved Sensitivity_Comparison_Closeness.pdf")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    print(f"--- Sensitivity Analysis for DELTAS = {DELTAS} ---")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' already exists. Overwriting content.")
    else:
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: '{OUTPUT_DIR}'")

    df_map = pd.read_csv(ORCHARD_FILE).drop_duplicates(subset=['node_id']).reset_index(drop=True)
    df_map['node_id'] = df_map['node_id'].astype(str)
    all_node_ids = df_map['node_id'].tolist()
    
    all_results = {}
    final_spreads = []

    for delta in DELTAS:
        is_baseline = (delta == 0)
        
        # --- BASELINE / DELTA=0 Case ---
        if is_baseline:
            scenario_name = f"Baseline_Delta_0"
            baseline_results = run_monte_carlo_scenario("Baseline (Delta=0)", [], all_node_ids)
            generate_output_figures(scenario_name, baseline_results, df_map, OUTPUT_DIR, is_baseline=True)
            generate_csv_output(scenario_name, baseline_results, OUTPUT_DIR)
            all_results[scenario_name] = baseline_results
            final_spreads.append({'Metric': 'Baseline', 'Delta': 0, 'AvgSpread': baseline_results['avg_total']})
            continue

        # --- BETWEENNESS for current DELTA ---
        scenario_name_b = f"Betweenness_Delta_{delta}"
        nodes_to_exclude_b = get_top_nodes_to_remove('ranked_nodes_by_betweenness.csv', delta)
        if nodes_to_exclude_b is not None:
            results_b = run_monte_carlo_scenario(f"Betweenness (Top {delta} Removed)", nodes_to_exclude_b, all_node_ids)
            generate_output_figures(scenario_name_b, results_b, df_map, OUTPUT_DIR)
            generate_csv_output(scenario_name_b, results_b, OUTPUT_DIR)
            all_results[scenario_name_b] = results_b
            final_spreads.append({'Metric': 'Betweenness', 'Delta': delta, 'AvgSpread': results_b['avg_total']})

        # --- CLOSENESS for current DELTA ---
        scenario_name_c = f"Closeness_Delta_{delta}"
        nodes_to_exclude_c = get_top_nodes_to_remove('ranked_nodes_by_closeness.csv', delta)
        if nodes_to_exclude_c is not None:
            results_c = run_monte_carlo_scenario(f"Closeness (Top {delta} Removed)", nodes_to_exclude_c, all_node_ids)
            generate_output_figures(scenario_name_c, results_c, df_map, OUTPUT_DIR)
            generate_csv_output(scenario_name_c, results_c, OUTPUT_DIR)
            all_results[scenario_name_c] = results_c
            final_spreads.append({'Metric': 'Closeness', 'Delta': delta, 'AvgSpread': results_c['avg_total']})

    # --- FINAL SUMMARY ---
    print("\n--- FINAL SENSITIVITY ANALYSIS SUMMARY (AVERAGE OF {} RUNS) ---".format(ITERATIONS))
    df_summary = pd.DataFrame(final_spreads)
    print(df_summary.to_string(index=False))
    summary_filename = os.path.join(OUTPUT_DIR, 'final_spread_summary.csv')
    df_summary.to_csv(summary_filename, index=False)
    print(f"\nSummary saved to {summary_filename}")

    # --- GENERATE COMBINED PLOTS ---
    generate_sensitivity_comparison_plot(all_results, DELTAS, OUTPUT_DIR)

    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
