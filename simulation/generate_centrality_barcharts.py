
import simulation_engine
import networkx as nx
import matplotlib.pyplot as plt

def generate_barchart(centrality_dict, title, filename):
    """
    Generates and saves a bar chart for a given centrality dictionary.
    """
    # 1. Sort the nodes by centrality score in descending order
    sorted_centrality = sorted(centrality_dict.items(), key=lambda item: item[1], reverse=True)
    
    # 2. Prepare data for plotting
    nodes = [str(item[0]) for item in sorted_centrality]
    scores = [item[1] for item in sorted_centrality]
    
    # 3. Generate and save the bar chart
    print(f"Generating bar chart: {title}...")
    
    plt.figure(figsize=(12, 8))
    plt.bar(nodes, scores, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel('Node', fontsize=12)
    plt.ylabel('Centrality Score', fontsize=12)
    # Remove x-axis labels to avoid clutter, as there are many nodes
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(filename)
    print(f"   -> Bar chart saved to: {filename}")

def main():
    """
    Main function to calculate centralities and generate barcharts.
    """
    # 1. Create the graph
    print("Creating the graph from 'filtered_trees_2017.csv'...")
    graph = simulation_engine.create_graph("filtered_trees_2017.csv", 2500)
    
    if not graph:
        print("Failed to create the graph. Exiting.")
        return

    # --- Calculate Centralities ---
    print("\nCalculating Betweenness Centrality...")
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
    
    print("Calculating Closeness Centrality...")
    closeness_centrality = nx.closeness_centrality(graph, distance='weight')

    # --- Generate Barcharts ---
    generate_barchart(
        betweenness_centrality, 
        'Betweenness Centrality of All Nodes',
        'betweenness_centrality_barchart.pdf'
    )
    
    generate_barchart(
        closeness_centrality, 
        'Closeness Centrality of All Nodes',
        'closeness_centrality_barchart.pdf'
    )

if __name__ == "__main__":
    main()
