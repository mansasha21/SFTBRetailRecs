import typing as tp

import polars as pl
from scipy.sparse import csr_matrix

from configs.model import N_CANDIDATES
from inference.utils import nearest_neighbours_inference, alternating_least_squares_inference
from data.utils import encoder2df


def get_candidates(
    models: tp.Dict, uim: csr_matrix, uim_norm: csr_matrix, user_indexes: tp.List[int]
) -> tp.Dict[str, pl.DataFrame]:

    bm25_recs = nearest_neighbours_inference(
        model=models["bm25"], uim=uim_norm, user_idxs=user_indexes, model_name="bm25", n_candidates_default=N_CANDIDATES
    )

    tfidf_recs = nearest_neighbours_inference(
        model=models["tfidf"], uim=uim_norm, user_idxs=user_indexes, model_name="tfidf", n_candidates_default=N_CANDIDATES
    )

    cosine_recs = nearest_neighbours_inference(
        model=models["cosine"], uim=uim_norm, user_idxs=user_indexes, model_name="cosine", n_candidates_default=N_CANDIDATES
    )

    als_recs = alternating_least_squares_inference(
        model=models["als"], uim=uim, user_idxs=user_indexes, n_candidates_default=N_CANDIDATES
    )

    return {"bm25": bm25_recs, "tfidf": tfidf_recs, "cosine": cosine_recs, "als": als_recs}


def union_candidates(candidates: tp.Sequence[pl.DataFrame]) -> pl.DataFrame:
    union_candidates = (
        pl.concat(candidates).select([pl.col("user_id"), pl.col("item_id")]).unique(maintain_order=False)
    )
    return union_candidates


def decode(table: pl.DataFrame, encoder: tp.Dict[int, int], key: str, enc_key: str) -> pl.DataFrame:
    encoder_table = encoder2df(encoder=encoder, key=key, enc_key=enc_key)
    table = table.join(encoder_table, on=[enc_key]).drop(enc_key)
    return table


def split_list_to_blocks(lst: tp.List[int], block_size: int):  # -> tp.Generator[tp.List[int]]:
    for i in range(0, len(lst), block_size):
        yield lst[i : i + block_size]


def union_candidates(candidates: tp.Sequence[pl.DataFrame]) -> pl.DataFrame:
    union_candidates = (
        pl.concat(candidates).select([pl.col("receipt_id"), pl.col("item_id"), pl.col("model_name")]).unique(maintain_order=False)
    )
    return union_candidates