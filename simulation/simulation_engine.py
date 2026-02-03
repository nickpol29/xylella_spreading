import csv
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import random
from datetime import datetime, timedelta
import os

# --- 1. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ---
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

# --- 2. ΚΥΡΙΑ ΠΡΟΣΟΜΟΙΩΣΗ (UPDATED) ---
def run_simulation_full(
    beta_multiplier=1.0, 
    kernel_multiplier=1.0,
    season_profile=None,
    file_path="filtered_trees_2017.csv",
    seed_node="72",
    recovery_days=21,
    exclude_nodes=None
):
    MAX_DISTANCE = 2500
    G = create_graph(file_path, MAX_DISTANCE)
    if G is None: return {}

    # Remove excluded nodes if any
    if exclude_nodes:
        G.remove_nodes_from(exclude_nodes)

    # Default Profile (Μεσογειακό) αν δεν δωθεί άλλο
    if season_profile is None:
        season_profile = {
            1: 0.0, 2: 0.0, 3: 0.01, 4: 0.05, 
            5: 0.20, 6: 0.72, 7: 0.57, 8: 0.45, 
            9: 0.54, 10: 0.32, 11: 0.02, 12: 0.0
        }

    # Εύρεση Seed
    start_node = None
    if seed_node in G.nodes():
        start_node = seed_node
    else:
        # Αν δεν βρεθεί το seed (π.χ. το 72), ξεκινάμε από το πρώτο διαθέσιμο
        if len(G.nodes()) > 0: start_node = list(G.nodes())[0]

    if start_node is None: return {}

    start_date = datetime(2017, 3, 21)
    infected_dates = {start_node: start_date}
    current_date = start_date
    
    for day in range(365):
        current_date += timedelta(days=1)
        month = current_date.month
        
        # Λήψη εποχικότητας από το προφίλ
        season_factor = season_profile.get(month, 0.0)
        
        if season_factor <= 0: continue

        infected_list = list(infected_dates.keys())
        
        for infected in infected_list:
            neighbors = G[infected]
            for neighbor, data in neighbors.items():
                if neighbor not in infected_dates:
                    dist = data['weight']
                    
                    # KERNEL: Πιθανότητες ανά ζώνη απόστασης
                    base_prob = 0.0
                    if dist <= 35: base_prob = 0.015
                    elif dist <= 200: base_prob = 0.002
                    elif dist <= 400: base_prob = 0.001
                    elif dist <= 1000: base_prob = 0.0005
                    else: base_prob = 0.00005
                    
                    # Εφαρμογή των Multipliers
                    # Τύπος: P = P_base * K * Beta * Season
                    final_prob = base_prob * kernel_multiplier * beta_multiplier * season_factor
                    
                    if random.random() < final_prob:
                        infected_dates[neighbor] = current_date
        
        # Αν κολλήσουν όλα, σταματάμε
        if len(infected_dates) == len(G.nodes()): break

    return infected_dates