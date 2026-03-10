from typing import List


def optimize_route(distance_matrix: List[List[float]]) -> List[int]:
    """
    Simple route optimization using nearest neighbor algorithm.
    Returns the order of cities to visit.
    """

    if not distance_matrix:
        return []

    n = len(distance_matrix)
    visited = [False] * n

    route = [0]
    visited[0] = True

    for _ in range(n - 1):

        last_city = route[-1]

        nearest_city = None
        nearest_distance = float("inf")

        for city in range(n):
            if not visited[city] and distance_matrix[last_city][city] < nearest_distance:
                nearest_distance = distance_matrix[last_city][city]
                nearest_city = city

        if nearest_city is None:
            break

        route.append(nearest_city)
        visited[nearest_city] = True

    return route