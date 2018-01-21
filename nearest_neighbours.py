from typing import List

from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def nearest_neighbours_graph(points: np.array, query_points: np.array or None = None, n_neighbours=2) -> np.array:
    if query_points is None:
        query_points = points
    nbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree', n_jobs=-1).fit(points)
    return nbrs.kneighbors_graph(query_points).toarray()


def draw_nearest_neighbours_graph(nearest_neigbours_array: np.array, labels: List[str] = None):
    nx_graph = nx.from_numpy_matrix(nearest_neigbours_array)
    pos = None
    scale = 8
    if labels:
        label_mapping = {}
        for i in range(len(nx_graph.nodes())):
            label_mapping[i] = labels[i]
        nx_graph = nx.relabel_nodes(nx_graph, label_mapping)
        pos = nx.kamada_kawai_layout(nx_graph, scale=scale)
        nx.draw_networkx_labels(nx_graph, pos)

    if not pos:
        pos = nx.kamada_kawai_layout(nx_graph, scale=scale)
    nx.draw(nx_graph, pos)
    plt.show()


def get_nearest_neighbours(neighbours: np.array, labels: List[str] = None):
    return [labels[i] for i, neighbour in enumerate(neighbours)
            if neighbour == 1]
