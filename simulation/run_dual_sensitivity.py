import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import simulation_engine
import os

# --- 1. ΡΥΘΜΙΣΕΙΣ (GRID SEARCH) ---
BETAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]  # Εύρος μεταδοτικότητας
KERNELS = [1.0, 1.5, 2.0, 2.5, 3.0]      # Εύρος εμβέλειας
ITERATIONS = 15                          # Επαναλήψεις ανά κελί (για ταχύτητα, βάλε 50 για τελικό)

# Τα δύο σενάρια που θέλουμε να ελέγξουμε
SCENARIOS = [
    {"name": "Scenario_A_30Days", "target_file": "calibration_targets_30days.csv"},
    {"name": "Scenario_B_FullYear", "target_file": "filtered_trees_2017.csv"}
]

# Το αρχείο που έχει τις συντεταγμένες για να τρέξει το μοντέλο (Input Data)
INPUT_MAP = "filtered_trees_2017.csv"

# Εποχικότητα
SEASON = {1:0, 2:0, 3:0.01, 4:0.05, 5:0.2, 6:0.72, 7:0.57, 8:0.45, 9:0.54, 10:0.32, 11:0.02, 12:0}

print("--- STARTING DUAL SENSITIVITY ANALYSIS ---")

# Συνάρτηση που διαβάζει τα POSITIVO από οποιοδήποτε αρχείο
def extract_positives(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return set()
    
    df = pd.read_csv(filename)
    df['node_id'] = df['node_id'].astype(str)
    
    # Ψάχνουμε για 'state' == POSITIVO
    if 'state' in df.columns:
        # Χειρισμός κεφαλαίων/μικρών και κενών
        pos = df[df['state'].str.strip().str.upper() == 'POSITIVO']
        return set(pos['node_id'].tolist())
    # Backup: Ψάχνουμε για 'infected' == 1
    elif 'infected' in df.columns:
        pos = df[df['infected'] == 1]
        return set(pos['node_id'].tolist())
    
    return set()

# --- 2. ΚΥΡΙΟΣ ΒΡΟΧΟΣ (Τρέχει και για τα 2 σενάρια) ---
for scen in SCENARIOS:
    name = scen["name"]
    target_file = scen["target_file"]
    
    print(f"\n=== Running Analysis for: {name} ===")
    print(f"Target File: {target_file}")
    
    # Φόρτωση του "Στόχου" (Ground Truth) για αυτό το σενάριο
    ground_truth_ids = extract_positives(target_file)
    print(f"Found {len(ground_truth_ids)} positive trees in target file.")
    
    if len(ground_truth_ids) == 0:
        print("Skipping due to lack of targets.")
        continue

    # Πίνακας για να αποθηκεύσουμε τα scores
    results_matrix = np.zeros((len(BETAS), len(KERNELS)))

    # Grid Search Loop
    for i, b in enumerate(BETAS):
        for j, k in enumerate(KERNELS):
            # Τρέχουμε το μοντέλο πολλές φορές
            scores = []
            for _ in range(ITERATIONS):
                # Τρέχουμε την προσομοίωση
                # Χρησιμοποιούμε πάντα το πλήρες map για την κίνηση, αλλά ελέγχουμε με τον στόχο
                log = simulation_engine.run_simulation_full(b, k, SEASON, INPUT_MAP)
                
                # Ποια μολύνθηκαν στην προσομοίωση;
                simulated_infected = set(str(n) for n in log.keys())
                
                # Υπολογισμός Jaccard με τον συγκεκριμένο στόχο
                intersection = len(ground_truth_ids.intersection(simulated_infected))
                union = len(ground_truth_ids.union(simulated_infected))
                jaccard = intersection / union if union > 0 else 0
                scores.append(jaccard)
            
            # Μέσος όρος σκορ
            avg_score = np.mean(scores)
            results_matrix[i, j] = avg_score
            # print(f"   Beta={b}, K={k} -> Jaccard={avg_score:.3f}") # Ξε-σχολίασε αν θες details

    # --- 3. CREATING HEATMAP ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_matrix, annot=True, cmap='viridis', fmt=".3f",
                xticklabels=KERNELS, yticklabels=BETAS)
    
    plt.title(f'Calibration Heatmap: {name}\n(Target: {len(ground_truth_ids)} trees)', fontsize=14)
    plt.xlabel('Kernel Multiplier (K)')
    plt.ylabel('Transmission Rate (Beta)')
    
    # Αποθήκευση
    filename_pdf = f"Heatmap_{name}.pdf"
    filename_svg = f"Heatmap_{name}.svg"
    plt.savefig(filename_pdf, bbox_inches='tight')
    plt.savefig(filename_svg, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap: {filename_pdf}")

print("\n--- DUAL ANALYSIS COMPLETE ---")