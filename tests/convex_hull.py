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
selected_states = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
path = "results_ilqr/stability_analysis/23_07_14_11_30/stability_region.npz"
array = np.load(path)
states = array['states']
mask = confidence_region(states[:, :, -1], ord='inf', c=0.1)
filtered_states = states[index][mask[index]]
filtered_start_states = filtered_states[:, 0]
points = filtered_start_states[:, selected_states]


# Step 2: Calculate the convex hull

hull = ConvexHull(points)
hull_vertices = points[hull.vertices]
polygon = Polygon([points[vertex] for vertex in hull.vertices])


# Step 3: Compute the centroid of the convex hull
centroid = np.mean(hull_vertices, axis=0)
print(f" ==> centroid: {centroid}")

# Step 4: Define a function to sample points inside the convex hull using random convex combination


def sample_from_convex_hull(polygon, num_samples=100):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []

    while len(points) < num_samples:
        # Generate random points within the bounding box
        random_point = Point(
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y)
        )
        # Check if the point lies within the convex hull
        if polygon.contains(random_point):
            points.append((random_point.x, random_point.y))

    return np.array(points)


# Sample 10 points from inside the convex hull
start = time.time()
samples = sample_from_convex_hull(polygon, num_samples=1_000)
end = time.time()
print(f"  ==> Sampling time: {end - start}")

# Plotting
plt.figure(figsize=(8, 6))
# Plot original points
plt.scatter(points[:, 0], points[:, 1], label='Points', color='blue')
# Plot convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# Plot centroid
plt.scatter(*centroid, color='red', label='Centroid', marker='x', s=100)
# Plot sampled points
plt.scatter(samples[:, 0], samples[:, 1], color='green',
            label='Samples from Convex Hull', marker='o')

plt.title('Convex Hull and Centroid')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
