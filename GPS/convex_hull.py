import numpy as np
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


class ConvexSet:
    def __init__(self, coordinates: np.ndarray, mask: np.ndarray, n_x: int) -> None:
        self._hull = ConvexHull(coordinates)
        self._hull_vertices = coordinates[self._hull.vertices]
        self.polygon = Polygon([coordinates[vertex]
                               for vertex in self._hull.vertices])
        self.centroid = np.zeros(n_x)
        self.centroid[mask] = np.mean(self._hull_vertices, axis=0)
        self._mask = mask
        self._n_x = n_x

    def sample(self, num_samples: int = 1):
        min_x, min_y, max_x, max_y = self.polygon.bounds
        points = []

        while len(points) < num_samples:
            # Generate random points within the bounding box
            random_point = Point(
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y)
            )
            # Check if the point lies within the convex hull
            if self.polygon.contains(random_point):
                points.append((random_point.x, random_point.y))
        states = np.zeros((num_samples, self._n_x))
        states[: self._mask] = np.array(points)
        return states


class UniformSet:
    def __init__(self, low_range: np.ndarray, high_range: np.ndarray,
                 x0: np.ndarray = None) -> None:
        self.low_range = low_range
        self.high_range = high_range
        self.centroid = x0 if isinstance(
            x0, np.ndarray) else np.zeros(low_range.shape[0])

    def sample(self, num_samples: int = 1):
        size = (num_samples, self.low_range.shape[0])
        states = np.random.uniform(self.low_range, self.high_range, size)
        return states + self.centroid
