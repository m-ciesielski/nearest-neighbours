from collections import OrderedDict
from typing import Dict, List

import numpy as np

from nearest_neighbours import nearest_neigbours_graph, draw_nearest_neighbours_graph


def load_embeddings(embeddings_file_path: str, limit=1000) -> Dict[str, List[float]]:
    embeddings = OrderedDict()
    with open(embeddings_file_path, mode='r', encoding='utf-8') as embeddings_file:
        for embedding_line in embeddings_file:
            label, vector_string = embedding_line.split(' ', maxsplit=1)
            embeddings[label] = [float(v) for v in vector_string.split(' ')]
            if len(embeddings) >= limit:
                break

    return embeddings

if __name__ == '__main__':

    embeddings = load_embeddings('glove.6B.50d.txt', limit=500)
    print(list(embeddings.items())[:10])

    # create neighbours graph
    vector_array = np.array([v for v in embeddings.values()])
    print(vector_array.shape)
    neighbours_graph = nearest_neigbours_graph(points=vector_array, n_neighbours=3)
    print(neighbours_graph.shape)
    draw_nearest_neighbours_graph(nearest_neigbours_array=neighbours_graph, labels=list(embeddings.keys()))

