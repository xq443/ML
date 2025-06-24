import json
import matplotlib.pyplot as plt

# Load JSON data from file
with open('extracted_frames/player_spatial_features.json', 'r') as f:
    data = json.load(f)

# Extract timestamps, x, and y coordinates
timestamps = [entry['timestamp'] for entry in data]
x_pixels = [entry['x_pixel'] for entry in data]
y_pixels = [entry['y_pixel'] for entry in data]

# Optional: Flip y-axis if necessary (image coordinates usually start from top-left)
# y_pixels = [max(y_pixels) - y for y in y_pixels]

# Plot trajectory
plt.figure(figsize=(10, 6))
plt.plot(x_pixels, y_pixels, marker='o', linestyle='-', color='blue', label='Trajectory')

# Annotate timestamps for better understanding (optional)
for i, t in enumerate(timestamps):
    if i % 2 == 0:  # reduce clutter by skipping some
        plt.text(x_pixels[i], y_pixels[i], f'{t:.1f}', fontsize=8)

plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.title('Player Trajectory Over Time')
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis()  # Invert Y axis if your image coordinate origin is top-left
plt.tight_layout()
plt.show()
