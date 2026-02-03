#Each point’s latitude and longitude are entered in two separate inputs.
#Boundaries (min/max lat/lon) are automatically calculated from z1 and z2.
#Filters your Excel file to keep only rows inside the rectangle.
#Exports the result to a new file filtered_trees.xlsx.

import pandas as pd

# Load Excel file
input_file = 'fixed_test2.csv'  # change to your file name
df = pd.read_excel(input_file)

# Ask user for coordinates (latitude and longitude separately for each point)
a1_lat = float(input("Enter latitude for a1: "))
a1_lon = float(input("Enter longitude for a1: "))

b1_lat = float(input("Enter latitude for b1: "))
b1_lon = float(input("Enter longitude for b1: "))

a2_lat = float(input("Enter latitude for a2: "))
a2_lon = float(input("Enter longitude for a2: "))

b2_lat = float(input("Enter latitude for b2: "))
b2_lon = float(input("Enter longitude for b2: "))

# Calculate z1 and z2
z1_lat, z1_lon = b1_lat, a1_lon
z2_lat, z2_lon = a2_lat, b2_lon

# Determine boundaries
min_lat = min(z2_lat, z1_lat)
max_lat = max(z2_lat, z1_lat)
min_lon = min(z1_lon, z2_lon)
max_lon = max(z1_lon, z2_lon)

# Filter dataframe
filtered_df = df[
    (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
    (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
]

# Save to new Excel file
output_file = 'filtered_trees.csv'
filtered_df.to_excel(output_file, index=False)

print(f"Filtered data saved to {output_file}. {len(filtered_df)} rows retained.")
