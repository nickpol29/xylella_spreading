
import pandas as pd

try:
    df = pd.read_csv('filtered_trees_2017.csv')
    min_lat = df['latitude'].min()
    max_lat = df['latitude'].max()
    min_lon = df['longitude'].min()
    max_lon = df['longitude'].max()
    print(f"min_lat={min_lat}")
    print(f"max_lat={max_lat}")
    print(f"min_lon={min_lon}")
    print(f"max_lon={max_lon}")
except FileNotFoundError:
    print("Error: filtered_trees_2017.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
