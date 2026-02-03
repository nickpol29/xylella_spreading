
import simulation_engine
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_robustness(graph, ranking_strategy, max_nodes_to_remove):
    """
    Analyzes the robustness of a graph by iteratively removing nodes based on a given ranking.

    Args:
        graph (nx.Graph): The graph to analyze.
        ranking_strategy (list): A sorted list of nodes (tuples of node, score) to remove.
        max_nodes_to_remove (int): The maximum number of nodes to remove in the simulation.

    Returns:
        list: A list containing the size of the largest connected component after each node removal.
    """
    original_graph = graph.copy()
    nodes_to_remove = [node for node, score in ranking_strategy]
    
    lcc_sizes = []
    
    # Calculate the size of the largest connected component of the original graph
    initial_lcc_size = len(max(nx.connected_components(original_graph), key=len))
    lcc_sizes.append(initial_lcc_size)

    # Use tqdm for a progress bar
    for k in tqdm(range(1, max_nodes_to_remove + 1), desc=f"Analyzing {ranking_strategy[0][1]:.2f} strategy"):
        # Ensure we don't try to remove more nodes than exist
        if k > len(nodes_to_remove):
            break
            
        node_to_remove = nodes_to_remove[k-1]
        
        if original_graph.has_node(node_to_remove):
            original_graph.remove_node(node_to_remove)
        
        # If the graph becomes empty, the lcc size is 0
        if len(original_graph) == 0:
            lcc_sizes.append(0)
            continue

        # Calculate the size of the largest connected component
        largest_component = max(nx.connected_components(original_graph), key=len)
        lcc_sizes.append(len(largest_component))
        
    return lcc_sizes

def main():
    """
    Main function to run the robustness analysis.
    """
    # 1. Create the graph
    print("Creating the graph from 'filtered_trees_2017.csv'...")
    graph = simulation_engine.create_graph("filtered_trees_2017.csv", 2500)
    
    if not graph:
        print("Failed to create the graph. Exiting.")
        return

    print(f"Total number of nodes in the graph: {graph.number_of_nodes()}")

    # --- Calculate Centralities ---
    print("Calculating Betweenness Centrality...")
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
    
    print("Calculating Closeness Centrality...")
    closeness_centrality = nx.closeness_centrality(graph, distance='weight')

    # --- Sort nodes by centrality ---
    sorted_by_betweenness = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    sorted_by_closeness = sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True)

    # --- DEBUG: Print top 10 nodes for each strategy ---
    print("\n--- Top 10 Nodes by Betweenness Centrality ---")
    for i, (node, score) in enumerate(sorted_by_betweenness[:10]):
        print(f"{i+1:2d}. Node {node}: {score:.6f}")

    print("\n--- Top 10 Nodes by Closeness Centrality ---")
    for i, (node, score) in enumerate(sorted_by_closeness[:10]):
        print(f"{i+1:2d}. Node {node}: {score:.6f}")
    # --- END DEBUG ---

    # --- Run Analysis ---
    num_nodes_to_remove = min(500, graph.number_of_nodes() - 1) # Remove up to 500 nodes or almost all nodes
    print(f"\nStarting robustness analysis by removing up to {num_nodes_to_remove} nodes...")
    
    # Pass dummy scores to the ranking_strategy tuples to identify them in the progress bar
    betweenness_strategy_for_analysis = [(node, 1.0) for node, score in sorted_by_betweenness]
    closeness_strategy_for_analysis = [(node, 0.0) for node, score in sorted_by_closeness]

    lcc_sizes_betweenness = analyze_robustness(graph.copy(), betweenness_strategy_for_analysis, num_nodes_to_remove)
    lcc_sizes_closeness = analyze_robustness(graph.copy(), closeness_strategy_for_analysis, num_nodes_to_remove)

    # --- Plotting ---
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(12, 8))
    
    k_values = range(num_nodes_to_remove + 1)
    
    plt.plot(k_values, lcc_sizes_betweenness, marker='o', linestyle='-', markersize=4, label='Betweenness Centrality')
    plt.plot(k_values, lcc_sizes_closeness, marker='x', linestyle='--', markersize=4, label='Closeness Centrality')
    
    plt.title('Network Robustness under Targeted Attacks', fontsize=16)
    plt.xlabel('Number of Nodes Removed', fontsize=12)
    plt.ylabel('Size of Largest Connected Component', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_filename = 'robustness_comparison.pdf'
    plt.savefig(output_filename)
    
    print(f"   -> Analysis complete. Plot saved to: {output_filename}")

if __name__ == "__main__":
    main()
