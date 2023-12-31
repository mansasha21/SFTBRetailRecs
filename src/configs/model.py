RANDOM_STATE = 2105
N_CANDIDATES = 10


bm25_config = {
    "K": 10,
    "K1": 0.15,
    "B": 0.66
}

tfidf_config = {
    "K": 15
}

cosine_config = {
    "K": 15
}

als_config = {
    "factors": 5,
    "regularization": 0.01,
    "iterations": 100,
    "random_state": RANDOM_STATE,       
}

