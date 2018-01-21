import argparse

import numpy as np

from word_embeddings import load_embeddings
from nearest_neighbours import nearest_neighbours_graph, draw_nearest_neighbours_graph


def parse_args():
    parser = argparse.ArgumentParser(description='Draw a nearest neighbours graph for given word embeddings index.')
    parser.add_argument('-e', '--embeddings-path', required=True,
                        type=str, help='Path to CSV file containing word embeddings.')
    parser.add_argument('-l', '--limit', default=None,
                        type=int, help='Limit a number of words that will be loaded from word embeddings file.')
    parser.add_argument('-n', '--nearest-neighbours', default=3,
                        type=int, help='Nearest neighbours count.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    embeddings = load_embeddings(args.embeddings_path, limit=args.limit)

    # create neighbours graph
    vector_array = np.array([v for v in embeddings.values()])
    neighbours_graph = nearest_neighbours_graph(points=vector_array, n_neighbours=args.nearest_neighbours)
    draw_nearest_neighbours_graph(nearest_neigbours_array=neighbours_graph, labels=list(embeddings.keys()))

