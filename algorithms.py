import math
import heapq
import networkx as nx

coordinates = {
    "Baggage_Area":   (0, 0),
    "Checkpoint":     (1, 4),
    "Security":       (2, 1),
    "Food_Court":     (4, 2),
    "Terminal_Hall":  (5, 5),
    "Departure_Gate": (8, 6),
}

edges_with_costs = [
    ("Baggage_Area", "Checkpoint", 4.1),
    ("Baggage_Area", "Security", 2.2),
    ("Security", "Food_Court", 2.2),
    ("Checkpoint", "Terminal_Hall", 5.0),
    ("Food_Court", "Terminal_Hall", 3.2),
    ("Food_Court", "Departure_Gate", 6.0),
    ("Terminal_Hall", "Departure_Gate", 3.2),
]


def build_graph():
    G = nx.DiGraph()
    for node, pos in coordinates.items():
        G.add_node(node, pos=pos)
    for u, v, w in edges_with_costs:
        G.add_edge(u, v, weight=w)
    return G


def euclidean(a, b):
    (x1, y1), (x2, y2) = coordinates[a], coordinates[b]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def compute_heuristics(goal):
    return {n: round(euclidean(n, goal), 4) for n in coordinates}


def gbfs(graph, start, goal, h):
    counter = 0
    open_list = [(h[start], counter, start, [start], 0)]
    visited = set()
    expansion_order = []

    while open_list:
        _, _, current, path, cost = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            return path, cost, expansion_order

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                counter += 1
                edge_cost = graph[current][neighbor]["weight"]
                heapq.heappush(
                    open_list,
                    (h[neighbor], counter, neighbor,
                     path + [neighbor], cost + edge_cost)
                )
    return None, float("inf"), expansion_order


def astar(graph, start, goal, h):
    counter = 0
    open_list = [(h[start], counter, start, [start], 0)]
    visited = set()
    expansion_order = []
    best_g = {start: 0}

    while open_list:
        f, _, current, path, g = heapq.heappop(open_list)

        if current in visited:
            continue
        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            return path, g, expansion_order

        for neighbor in graph.neighbors(current):
            edge_cost = graph[current][neighbor]["weight"]
            new_g = g + edge_cost

            if neighbor not in best_g or new_g < best_g[neighbor]:
                best_g[neighbor] = new_g
                counter += 1
                new_f = new_g + h[neighbor]
                heapq.heappush(
                    open_list,
                    (new_f, counter, neighbor, path + [neighbor], new_g)
                )
    return None, float("inf"), expansion_order