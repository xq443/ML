import csv
import json
import numpy as np

def compute_spatial_features(data):
    """
    data: list of dicts with keys: 'timestamp', 'x_pixel', 'y_pixel'
    Returns list of dicts with added velocity and acceleration
    """
    features = []
    
    for i, curr in enumerate(data):
        t = float(curr['timestamp'])
        x = float(curr['x_pixel'])
        y = float(curr['y_pixel'])

        # Initialize velocity and acceleration
        vx = vy = ax = ay = 0.0

        if i > 0:
            prev = data[i-1]
            dt = t - float(prev['timestamp'])
            if dt > 0:
                vx = (x - float(prev['x_pixel'])) / dt
                vy = (y - float(prev['y_pixel'])) / dt

        if i > 1:
            prev_vx = features[i-1]['v_x']
            prev_vy = features[i-1]['v_y']
            dt = t - float(data[i-1]['timestamp'])
            if dt > 0:
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt

        features.append({
            'timestamp': t,
            'x_pixel': x,
            'y_pixel': y,
            'v_x': vx,
            'v_y': vy,
            'a_x': ax,
            'a_y': ay
        })

    return features

def read_csv_filter_player(filepath, player_id):
    """Read CSV file and filter rows for given player_id"""
    filtered_data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['player_id']) == player_id:
                filtered_data.append(row)
    return filtered_data

def save_features_to_json(features, output_path):
    """Save computed features list to a JSON file"""
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=4)
    print(f"Features saved to {output_path}")

if __name__ == "__main__":
    csv_path = "extracted_frames/player_tracking.csv"  
    player_id_to_test = 1
    output_json_path = "extracted_frames/player_spatial_features.json"

    data = read_csv_filter_player(csv_path, player_id_to_test)
    if not data:
        print(f"No data found for player_id={player_id_to_test}")
    else:
        features = compute_spatial_features(data)
        save_features_to_json(features, output_json_path)
