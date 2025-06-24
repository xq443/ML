import numpy as np
import pandas as pd

# Helper: center of bounding box
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

# Helper: distance to a horizontal court line (e.g., center line or baseline)
def distance_to_area(player_pos, area_y):
    return abs(player_pos[1] - area_y)

def compute_spatiotemporal_attributes(frame_data, court_center_y=360, baseline_y=50):
    records = []

    for i in range(len(frame_data)):
        frame = frame_data[i]
        timestamp = frame['timestamp']
        bbox = frame['bbox_player']
        x, y = get_center(bbox)

        record = {
            'timestamp': timestamp,
            'x': x,
            'y': y,
        }

        # Velocity
        if i > 0:
            prev_frame = frame_data[i - 1]
            prev_x, prev_y = get_center(prev_frame['bbox_player'])
            dt = frame['timestamp'] - prev_frame['timestamp']
            if dt > 0:
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
            else:
                vx = vy = 0
            record['vx'] = vx
            record['vy'] = vy
        else:
            record['vx'] = record['vy'] = 0

        # Acceleration
        if i > 1:
            prev_vx = records[i-1]['vx']
            prev_vy = records[i-1]['vy']
            dt = frame['timestamp'] - frame_data[i - 1]['timestamp']
            if dt > 0:
                ax = (record['vx'] - prev_vx) / dt
                ay = (record['vy'] - prev_vy) / dt
            else:
                ax = ay = 0
            record['ax'] = ax
            record['ay'] = ay
        else:
            record['ax'] = record['ay'] = 0

        # Distances to key court areas
        record['dist_to_center_line'] = distance_to_area((x, y), court_center_y)
        record['dist_to_baseline'] = distance_to_area((x, y), baseline_y)

        # Pose joint coordinates (if any)
        if 'pose_joints' in frame:
            for j, (jx, jy) in enumerate(frame['pose_joints']):
                record[f'joint_{j}_x'] = jx
                record[f'joint_{j}_y'] = jy

        records.append(record)

    return pd.DataFrame(records)


def test_spatiotemporal_analysis():
    test_data = [
        {
            'timestamp': 0.0,
            'bbox_player': [100, 200, 140, 250],
            'pose_joints': [(120, 220), (122, 218), (119, 225)]
        },
        {
            'timestamp': 0.04,
            'bbox_player': [104, 204, 144, 254],
            'pose_joints': [(124, 224), (126, 222), (123, 229)]
        },
        {
            'timestamp': 0.08,
            'bbox_player': [110, 210, 150, 260],
            'pose_joints': [(130, 230), (132, 228), (129, 235)]
        }
    ]

    df = compute_spatiotemporal_attributes(test_data)

    print("\n=== Spatiotemporal Attributes DataFrame ===")
    print(df.round(2))

test_spatiotemporal_analysis()
