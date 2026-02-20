import csv
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import random
from datetime import datetime, timedelta
import os
import pandas as pd

# --- 1. BOΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ (from simulation_engine.py) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def create_graph(file_path, max_distance):
    G = nx.Graph()
    nodes = []
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nid = row['node_id'].strip()
                lat, lon = float(row['latitude']), float(row['longitude'])
                nodes.append((nid, lat, lon))
                G.add_node(nid, pos=(lat, lon))
            except: continue
            
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = haversine(nodes[i][1], nodes[i][2], nodes[j][1], nodes[j][2])
            if dist <= max_distance:
                G.add_edge(nodes[i][0], nodes[j][0], weight=dist)
    return G

# --- 2. DISTANCE-ONLY SIMULATION ---
def run_distance_only_simulation(
    G, # Pass graph directly to avoid recreating it 1000 times
    seed_node="72",
    infection_prob=0.5
):
    start_node = None
    if seed_node in G.nodes():
        start_node = seed_node
    elif len(G.nodes()) > 0:
        start_node = list(G.nodes())[0]

    if start_node is None: return {}

    start_date = datetime(2017, 3, 21)
    infected_nodes = {start_node: start_date}
    
    # We can simulate for 365 days or until all nodes are infected
    for day in range(365):
        current_date = start_date + timedelta(days=day + 1)
        
        # Use a copy to prevent issues with iterating while modifying
        newly_infected_today = set()

        for infected_node in list(infected_nodes.keys()):
            for neighbor in G.neighbors(infected_node):
                if neighbor not in infected_nodes:
                    if random.random() < infection_prob:
                        newly_infected_today.add(neighbor)
        
        for node in newly_infected_today:
            infected_nodes[node] = current_date

        if len(infected_nodes) == len(G.nodes()):
            break
            
    return infected_nodes

# --- 3. EXECUTION & OUTPUT (Monte Carlo with Correct Averaging) ---
if __name__ == "__main__":
    MONTE_CARLO_RUNS = 1000
    INFECTION_RADIUS_M = 400
    INFECTION_PROB = 0.001
    SIMULATION_DAYS = 365
    
    print(f"Running Distance-Only Simulation with Monte Carlo ({MONTE_CARLO_RUNS} runs)...")
    print(f"Parameters: Radius={INFECTION_RADIUS_M}m, Probability={INFECTION_PROB}")

    # Create the graph once
    G = create_graph("filtered_trees_2017.csv", INFECTION_RADIUS_M)
    if G is None:
        print("Graph could not be created. Exiting.")
        exit()
        
    total_node_count = G.number_of_nodes()

    all_runs_daily_counts = []
    start_date = datetime(2017, 3, 21)

    for i in range(MONTE_CARLO_RUNS):
        print(f"  - Running simulation {i+1}/{MONTE_CARLO_RUNS}...", end='\r')
        
        results = run_distance_only_simulation(
            G,
            seed_node="72",
            infection_prob=INFECTION_PROB
        )
        
        # Convert this run's results to a simple list of infection day numbers
        run_daily_infections = [0] * SIMULATION_DAYS
        if results:
            for node, infected_date in results.items():
                day_index = (infected_date - start_date).days
                if 0 <= day_index < SIMULATION_DAYS:
                    run_daily_infections[day_index] += 1
        all_runs_daily_counts.append(run_daily_infections)

    print("\nSimulations complete. Averaging results...")

    # Create a DataFrame from the results, with days as index and runs as columns
    df_all_runs = pd.DataFrame(all_runs_daily_counts).T
    
    # Get cumulative cases for each run
    df_cumulative = df_all_runs.cumsum()

    # Calculate the mean across all runs for each day for both new and cumulative cases
    mean_new_cases = df_all_runs.mean(axis=1)
    mean_cumulative = df_cumulative.mean(axis=1)

    # Save to CSV in the desired format
    output_filename = "daily_summary_DistanceOnly.csv"
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["date", "mean_daily_new_cases", "cumulative_cases"])
        
        for day in range(SIMULATION_DAYS):
            date_obj = start_date + timedelta(days=day)
            date_str = date_obj.strftime('%Y-%m-%d')
            writer.writerow([date_str, mean_new_cases.iloc[day], mean_cumulative.iloc[day]])
            
    print(f"Averaged results saved to {output_filename}")
    print(f"Average total infected nodes at end of period: {mean_cumulative.iloc[-1]:.2f}")
