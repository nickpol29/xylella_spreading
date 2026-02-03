import simulation_engine
import networkx as nx
import matplotlib.pyplot as plt
import csv

# 1. Create the graph using the project's own simulation engine function
print("Creating the graph from 'filtered_trees_2017.csv'...")
graph = simulation_engine.create_graph("filtered_trees_2017.csv", 2500)

if graph:
    # 2. Calculate closeness centrality using networkx
    print("Calculating closeness centrality for all nodes...")
    centrality = nx.closeness_centrality(graph, distance='weight')

    # 3. Sort the nodes by centrality score in descending order
    sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)

    # 4. Print the top 15 results
    print("\n--- Top 15 Nodes by Closeness Centrality ---")
    for i, (node, score) in enumerate(sorted_centrality[:15]):
        print(f"{i+1:2d}. Node {node}: {score:.6f}")
        
    # 5. Save the ranked list to a CSV file
    print("\nSaving ranked list to 'ranked_nodes_by_closeness.csv'...")
    output_csv_filename = 'ranked_nodes_by_closeness.csv'
    try:
        with open(output_csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Node ID', 'Closeness Score'])
            for i, (node, score) in enumerate(sorted_centrality):
                writer.writerow([i + 1, node, score])
        print(f"   -> Successfully saved to {output_csv_filename}")
    except IOError as e:
        print(f"   -> ERROR: Could not write to file {output_csv_filename}. Reason: {e}")

    # 6. Generate and save the bar chart for all nodes
    print("\nGenerating centrality bar chart for all nodes...")
    
    nodes = [str(item[0]) for item in sorted_centrality]
    scores = [item[1] for item in sorted_centrality]
    
    plt.figure(figsize=(12, 8))
    plt.bar(nodes, scores, color='skyblue', edgecolor='black')
    plt.title('Closeness Centrality of All Nodes', fontsize=16)
    plt.xlabel('Node', fontsize=12)
    plt.ylabel('Closeness Centrality Score', fontsize=12)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Remove x-axis labels to avoid clutter
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_filename_barchart = 'closeness_centrality_barchart.pdf'
    plt.savefig(output_filename_barchart)
    print(f"   -> Bar chart saved to: {output_filename_barchart}")

    # 7. Generate and save the histogram for all nodes
    print("\nGenerating centrality histogram for all nodes...")
    centrality_values = list(centrality.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(centrality_values, bins=50, color='lightcoral', edgecolor='black') # Adjusted bins for better visualization
    plt.title('Distribution of Closeness Centrality Scores (All Nodes)', fontsize=14)
    plt.xlabel('Closeness Centrality Score', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_filename_histogram = 'all_nodes_centrality_histogram.pdf'
    plt.savefig(output_filename_histogram)
    print(f"   -> Histogram saved to: {output_filename_histogram}")

else:
    print("Failed to create the graph. Please check file paths and data.")