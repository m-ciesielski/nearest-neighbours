from collections import OrderedDict
from typing import Dict, List


def load_embeddings(embeddings_file_path: str, limit: int or None = None) -> Dict[str, List[float]]:
    embeddings = OrderedDict()
    with open(embeddings_file_path, mode='r', encoding='utf-8') as embeddings_file:
        for embedding_line in embeddings_file:
            label, vector_string = embedding_line.split(' ', maxsplit=1)
            embeddings[label] = [float(v) for v in vector_string.split(' ')]
            if limit and len(embeddings) >= limit:
                break

    return embeddings
