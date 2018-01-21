import argparse
from collections import OrderedDict

import numpy as np

from word_embeddings import load_embeddings
from nearest_neighbours import nearest_neighbours_graph, get_nearest_neighbours


def parse_args():
    parser = argparse.ArgumentParser(description='Show nearest neighbours of given words in '
                                                 'given word embeddings index.')
    parser.add_argument('-e', '--embeddings-path', required=True,
                        type=str, help='Path to CSV file containing word embeddings.')
    parser.add_argument('-w', '--words', nargs='+', required=True,
                        type=str, help='List of words for which nearest neighbours will be printed.')
    parser.add_argument('-n', '--nearest-neighbours', default=3,
                        type=int, help='Nearest neighbours count.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    embeddings = load_embeddings(args.embeddings_path, limit=None)

    # TODO: simplify code
    # create neighbours graph
    vector_array = np.array([v for v in embeddings.values()])
    query_points_map = OrderedDict(
        [(label, vector)
         for label, vector in zip(embeddings.keys(), vector_array)
         if label in args.words])
    neighbours_graph = nearest_neighbours_graph(points=vector_array,
                                                query_points=np.array(list(query_points_map.values())),
                                                n_neighbours=args.nearest_neighbours)
    for i, word in enumerate(query_points_map.keys()):
        print(word)
        print('Najbliższi sąsiedzi: ')
        print(get_nearest_neighbours(neighbours_graph[i], labels=list(embeddings.keys())))

