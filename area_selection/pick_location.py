import pandas as pd

# 1. Φόρτωση αρχείου CSV
input_file = 'fixed_test2.csv' 

# ΑΛΛΑΓΗ: Χρησιμοποιούμε pd.read_csv αντί για pd.read_excel
# Σημείωση: Αν το csv σου έχει διαχωριστικό το ελληνικό ερωτηματικό (;), 
# πρόσθεσε sep=';' μέσα στην παρένθεση. Π.χ.: pd.read_csv(input_file, sep=';')
df = pd.read_csv(input_file)

# Εισαγωγή συντεταγμένων από τον χρήστη
print("--- Εισαγωγή Συντεταγμένων ---")
a1_lat = float(input("Enter latitude for a1: "))
a1_lon = float(input("Enter longitude for a1: "))

b1_lat = float(input("Enter latitude for b1: "))
b1_lon = float(input("Enter longitude for b1: "))

a2_lat = float(input("Enter latitude for a2: "))
a2_lon = float(input("Enter longitude for a2: "))

b2_lat = float(input("Enter latitude for b2: "))
b2_lon = float(input("Enter longitude for b2: "))

# Υπολογισμός των σημείων z1 και z2
z1_lat, z1_lon = b1_lat, a1_lon
z2_lat, z2_lon = a2_lat, b2_lon

# Καθορισμός ορίων (Boundaries)
min_lat = min(z2_lat, z1_lat)
max_lat = max(z2_lat, z1_lat)
min_lon = min(z1_lon, z2_lon)
max_lon = max(z1_lon, z2_lon)

print(f"\nFiltering inputs inside box: Lat({min_lat} - {max_lat}), Lon({min_lon} - {max_lon})")

# Φιλτράρισμα του dataframe
# Βεβαιώσου ότι οι στήλες στο CSV ονομάζονται 'latitude' και 'longitude'
filtered_df = df[
    (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
    (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
]

# 2. Αποθήκευση σε νέο αρχείο CSV
output_file = 'filtered_trees.csv'

# ΑΛΛΑΓΗ: Χρησιμοποιούμε to_csv αντί για to_excel
filtered_df.to_csv(output_file, index=False)

print(f"Επιτυχία! Τα φιλτραρισμένα δεδομένα αποθηκεύτηκαν στο {output_file}.")
print(f"Κρατήθηκαν {len(filtered_df)} εγγραφές.")