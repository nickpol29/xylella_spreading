import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simulation_engine
import os

# ==========================================
# 1. ΡΥΘΜΙΣΕΙΣ (MASTER CONFIGURATION)
# ==========================================
ITERATIONS = 1000          
BETA = 0.1                 
KERNEL_K = 1.5             
ORCHARD_FILE = "filtered_trees_2017.csv"  # Input Map & Full Year Truth
TARGET_FILE_30 = "calibration_targets_30days.csv"  # Short-term Truth
#EXCLUDED_NODES = ["228", "183", "119", "129", "233", "48", "127", "128", "182", "143",]

EXCLUDED_NODES = ["145", "156", "167", "129", "135", "121", "168", "120", "166", "128", "146", "144", "165", "134", "155", "122", "119", "183", "107", "147", "157", "182", "196", "106", "169", "197", "108", "195", "118", "105"]


SEASON = { 
    1: 0.0, 2: 0.0, 3: 0.01, 4: 0.05, 5: 0.20, 6: 0.72, 
    7: 0.57, 8: 0.45, 9: 0.54, 10: 0.32, 11: 0.02, 12: 0.0
}

print(f"--- STARTING CENTRALITY TEST (EXCLUDING {len(EXCLUDED_NODES)} NODES) ---")
print(f"Parameters: Iterations={ITERATIONS}, Beta={BETA}, Kernel={KERNEL_K}")

# ==========================================
# 2. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ & GROUND TRUTHS
# ==========================================
print("\n[1/5] Loading Data...")

def get_positive_nodes(filename):
    if not os.path.exists(filename):
        print(f"WARNING: File {filename} not found.")
        return set()
    
    df = pd.read_csv(filename)
    df['node_id'] = df['node_id'].astype(str)
    
    if 'state' in df.columns:
        positive_df = df[df['state'].str.strip().str.upper() == 'POSITIVO']
        return set(positive_df['node_id'].tolist())
    elif 'infected' in df.columns:
        positive_df = df[df['infected'] == 1]
        return set(positive_df['node_id'].tolist())
    return set()

# A. Φόρτωση Χάρτη (Graph Nodes) - Καθαρισμός Διπλότυπων
df_full = pd.read_csv(ORCHARD_FILE)
df_full['node_id'] = df_full['node_id'].astype(str)
df_map = df_full.drop_duplicates(subset=['node_id']).reset_index(drop=True) # Μοναδικά δέντρα για χάρτη
print(f"   -> Graph Nodes loaded: {len(df_map)}")

# B. Φόρτωση Ground Truths (Δύο διαφορετικά σετ)
truth_30d = get_positive_nodes(TARGET_FILE_30)
truth_full = get_positive_nodes(ORCHARD_FILE)

print(f"   -> Truth Set A (30 Days):  {len(truth_30d)} trees")
print(f"   -> Truth Set B (Full Year): {len(truth_full)} trees")

# ==========================================
# 3. ΕΚΤΕΛΕΣΗ MONTE CARLO (1000 RUNS)
# ==========================================
print(f"\n[2/5] Running Monte Carlo Simulation ({ITERATIONS} runs)...")

all_runs_curves = []       
node_hits = {}             
for nid in df_map['node_id']:
    node_hits[nid] = 0

for i in range(ITERATIONS):
    if (i+1) % 100 == 0: print(f"   ... Iteration {i+1}/{ITERATIONS}")
    
    # Τρέχουμε το μοντέλο, εξαιρώντας τους κόμβους
    log = simulation_engine.run_simulation_full(BETA, KERNEL_K, SEASON, ORCHARD_FILE, exclude_nodes=EXCLUDED_NODES)
    
    # Curve processing
    infected_doy = []
    for date in log.values():
        doy = date.timetuple().tm_yday
        if date.year > 2017: doy += 365
        infected_doy.append(doy)
    
    daily_counts = [sum(1 for d in infected_doy if d <= day) for day in range(1, 366)]
    all_runs_curves.append(daily_counts)
    
    # Map processing
    for infected_node in log.keys():
        nid_str = str(infected_node)
        if nid_str in node_hits:
            node_hits[nid_str] += 1

# ==========================================
# 4. ΥΠΟΛΟΓΙΣΜΟΣ METRICS & JACCARD
# ==========================================
print("\n[3/5] Calculating Metrics (Dual Validation)...")

df_results = df_map.copy()
# Υπολογισμός πιθανότητας
df_results['inf_prob'] = df_results['node_id'].map(node_hits).fillna(0) / ITERATIONS

# Thresholding
THRESHOLD = 0.20
predicted_infected = set(df_results[df_results['inf_prob'] >= THRESHOLD]['node_id'].tolist())

# Συνάρτηση Jaccard
def calc_jaccard(truth_set, pred_set):
    inter = len(truth_set.intersection(pred_set))
    union = len(truth_set.union(pred_set))
    return inter / union if union > 0 else 0

# Υπολογισμός ΔΥΟ scores
score_30d = calc_jaccard(truth_30d, predicted_infected)
score_full = calc_jaccard(truth_full, predicted_infected)

print(f"   -> Predicted Infected (>= {THRESHOLD}): {len(predicted_infected)}")
print(f"   -> Jaccard vs 30 Days:  {score_30d:.4f}")
print(f"   -> Jaccard vs Full Year: {score_full:.4f}")

# ==========================================
# 5. ΑΠΟΘΗΚΕΥΣΗ CSV
# ==========================================
print("\n[4/5] Saving Results to CSV...")

# Σώζουμε και τα δύο Truths στο CSV για έλεγχο
df_results['is_truth_30d'] = df_results['node_id'].isin(truth_30d).astype(int)
df_results['is_truth_full'] = df_results['node_id'].isin(truth_full).astype(int)

csv_filename = "final_simulation_results_centrality_test.csv"
df_results.to_csv(csv_filename, index=False)
print(f"   -> Saved raw data to: {csv_filename}")

# ==========================================
# 6. ΠΑΡΑΓΩΓΗ ΔΙΑΓΡΑΜΜΑΤΩΝ
# ==========================================
print("\n[5/5] Generating Figures (PDF & SVG)...")

# --- FIG 1: CURVE ---
mean_curve = np.mean(all_runs_curves, axis=0)
std_curve = np.std(all_runs_curves, axis=0)
days = np.arange(1, 366)

plt.figure(figsize=(10, 6))
plt.fill_between(days, mean_curve - std_curve, mean_curve + std_curve, color='blue', alpha=0.15, label='Standard Deviation')
plt.plot(days, mean_curve, color='darkblue', linewidth=2, label='Mean Epidemic Curve')
plt.title(f"Epidemic Progression (Mean of {ITERATIONS} runs)\nScenario: Beta={BETA}, Kernel={KERNEL_K}", fontsize=14)
plt.xlabel("Day of Year (1 = March 21st)", fontsize=12)
plt.ylabel("Cumulative Infected Trees", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('Fig1_Epidemic_Curve_centrality_test.pdf', format='pdf')
plt.savefig('Fig1_Epidemic_Curve_centrality_test.svg', format='svg')
plt.close()

# --- FIG 2: MAPS (Δείχνουμε το Full Year ως κύριο Ground Truth) ---
plt.figure(figsize=(16, 7))

# Left: Ground Truth (Full Year)
plt.subplot(1, 2, 1)
plt.scatter(df_map['longitude'], df_map['latitude'], c='lightgrey', s=40, label='Healthy')
real_inf_df = df_map[df_map['node_id'].isin(truth_full)]
plt.scatter(real_inf_df['longitude'], real_inf_df['latitude'], c='red', s=50, label='Confirmed (Full Year)')
plt.title("Ground Truth (Observed 2017)", fontsize=14)
plt.axis('equal')
plt.legend()

# Right: Simulation
plt.subplot(1, 2, 2)
sc = plt.scatter(df_results['longitude'], df_results['latitude'], 
                 c=df_results['inf_prob'], cmap='YlOrRd', s=50, vmin=0, vmax=1)
plt.colorbar(sc, label='Infection Probability')
plt.title(f"Simulated Risk Map (Avg of {ITERATIONS} runs)", fontsize=14)
plt.axis('equal')

plt.tight_layout()
plt.savefig('Fig2_Risk_Maps_centrality_test.pdf', format='pdf')
plt.savefig('Fig2_Risk_Maps_centrality_test.svg', format='svg')
plt.close()

# --- FIG 3: DUAL VALIDATION METRICS (ΝΕΟ!) ---
plt.figure(figsize=(8, 6))
scenarios = ['vs 30-Day Target', 'vs Full Year Data']
scores = [score_30d, score_full]
colors = ['#377eb8', '#4daf4a'] # Blue, Green

bars = plt.bar(scenarios, scores, color=colors, width=0.5)
plt.ylim(0, 1.0)
plt.ylabel('Jaccard Similarity Index', fontsize=12)
plt.title('Model Robustness: Validation across Datasets', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('Fig3_Validation_Metrics_centrality_test.pdf', format='pdf')
plt.savefig('Fig3_Validation_Metrics_centrality_test.svg', format='svg')
plt.close()

print("\n--- ALL DONE. FIGURES AND CSV SAVED. ---")