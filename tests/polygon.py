import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from typing import Union


def classifier(state: np.ndarray, c: float = 4e-1, mask: np.ndarray = None,
               ord: Union[int, str] = 2) -> np.ndarray:
    '''
    ord : {int, str}
    '''
    ord = int(ord) if ord.isdigit() else np.inf
    if isinstance(mask, np.ndarray):
        state = state[mask]
    return np.linalg.norm(state, ord=ord) < c


def confidence_region(states: np.ndarray, c: float = 4e-1, mask: np.ndarray = None,
                      ord: Union[int, float, str] = 2) -> np.ndarray:
    '''
    ord : {int, str: inf}
    '''
    return np.apply_along_axis(classifier, -1, states, c, mask, ord)


# Step 1: Load coordinates

index = -2
# u, v, w, x, y, z, p, q, r, psi, theta, phi
selected_states = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
path = "results_ilqr/stability_analysis/23_07_14_11_30/stability_region.npz"
array = np.load(path)
states = array['states']
mask = confidence_region(states[:, :, -1], ord='inf')
filtered_states = states[index][mask[index]]
filtered_start_states = filtered_states[:, 0]
points = filtered_start_states[:, selected_states]


# Step 2: Calculate the convex hull

# hull = ConvexHull(points)
# hull_vertices = points[hull.vertices]
x = points[:, 0]
y = points[:, 1]
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 4  # Example of a circular contour
contour = plt.contour(X, Y, Z, levels=[0])
try:
    contour_path = contour.collections[0].get_paths()[0]
except:
    breakpoint()
polygon = Polygon(contour_path.vertices)


# Step 3: Compute the centroid of the convex hull
centroid = polygon.centroid
print(f" ==> centroid: {centroid}")

# Step 4: Define a function to sample points inside the convex hull using random convex combination


def sample_points_within_contour(polygon, n_samples):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    while len(points) < n_samples:
        # Generate random points within the bounding box
        random_point = Point(
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y)
        )
        # Check if the point lies within the contour polygon
        if polygon.contains(random_point):
            points.append((random_point.x, random_point.y))

    return np.array(points)


# Sample 10 points from inside the convex hull
start = time.time()
samples = sample_points_within_contour(polygon, num_samples=1_000)
end = time.time()
print(f"  ==> Sampling time: {end - start}")

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot original points
ax.scatter(points[:, 0], points[:, 1], label='Points', color='blue')
# Plot convex hull
x, y = polygon.exterior.xy
ax.plot(x, y, 'k-', linewidth=2)  # Draw the outline of the polygo
# Plot centroid
ax.scatter(*polygon.centroid.xy, color='red',
           label='Centroid', marker='x', s=100)
# Plot sampled points
ax.scatter(samples[:, 0], samples[:, 1], color='green',
           label='Samples from Convex Hull', marker='o')

ax.set_title('Convex Hull and Centroid')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
ax.grid(True)
plt.show()
