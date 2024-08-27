import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Function to calculate Haversine distance between two points on Earth
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    
    return distance

# Load vessel data
df = pd.read_csv('./assets/sample_data.csv')  # Replace with the actual file path

# Define threshold distance in kilometers (e.g., 500 meters)
threshold_distance = 0.5

# Dictionary to store proximity events
proximity_events = defaultdict(list)

# Identify proximity events based on the threshold distance
for timestamp in df['timestamp'].unique():
    subset = df[df['timestamp'] == timestamp]
    for i, vessel1 in subset.iterrows():
        for j, vessel2 in subset.iterrows():
            if vessel1['mmsi'] != vessel2['mmsi']:
                distance = haversine(vessel1['lat'], vessel1['lon'], vessel2['lat'], vessel2['lon'])
                if distance <= threshold_distance:
                    proximity_events[vessel1['mmsi']].append((vessel2['mmsi'], timestamp))

# Convert proximity events to a DataFrame
result = []
for mmsi, interactions in proximity_events.items():
    for interaction in interactions:
        result.append({'mmsi': mmsi, 'vessel_proximity': interaction[0], 'timestamp': interaction[1]})

result_df = pd.DataFrame(result)

# Save proximity events to a CSV file
result_df.to_csv('vessel_proximity_events.csv', index=False)

print("Vessel proximity events detected and saved to vessel_proximity_events.csv")

# Visualization using Cartopy
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Plot vessel locations
ax.scatter(df['lon'], df['lat'], color='red', s=10, label='Vessels', transform=ccrs.PlateCarree())

# Highlight proximity events
for i, row in result_df.iterrows():
    vessel1 = df[df['mmsi'] == row['mmsi']].iloc[0]
    vessel2 = df[df['mmsi'] == row['vessel_proximity']].iloc[0]
    plt.plot([vessel1['lon'], vessel2['lon']],
             [vessel1['lat'], vessel2['lat']],
             color='yellow', linewidth=2, transform=ccrs.PlateCarree())

plt.title('Vessel Proximity Events')
plt.legend()
plt.show()
