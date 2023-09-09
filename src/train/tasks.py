from typing import Dict, Any

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender
from scipy.sparse import csr_matrix

from configs.model import bm25_config, tfidf_config, cosine_config, als_config


def train_implicit_models(spmat_norm: csr_matrix, spmat: csr_matrix) -> Dict[str, Any]:
    models = {
        "bm25": BM25Recommender(**bm25_config),
        "tfidf": TFIDFRecommender(**tfidf_config),
        "cosine": CosineRecommender(**cosine_config),
        "als": AlternatingLeastSquares(**als_config)
    }

    for model in ("tfidf", "cosine", "bm25"):
        models[model].fit(spmat_norm, show_progress=True)
    models["als"].fit(spmat)

    return models