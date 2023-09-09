import typing as tp

import numpy as np
import polars as pl

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from scipy.sparse import csr_matrix, find

from data.utils import encoder2df
from inference.utils import split_list_to_blocks


def alternating_least_squares_inference(
    model: AlternatingLeastSquares, uim: csr_matrix, user_idxs: tp.List[int], n_candidates_default: int = 100
) -> pl.DataFrame:

    n_items = uim.shape[1]
    n_candidates = min(n_candidates_default, n_items)
    recommend_items, _ = model.recommend(user_idxs, None, N=n_candidates, filter_already_liked_items=False)
    recommends = pl.DataFrame(
        [
            pl.Series(name="receipt_id_enc", values=np.repeat(user_idxs, n_candidates), dtype=pl.Int64),
            pl.Series(name="item_id_enc", values=recommend_items.flatten(), dtype=pl.Int64),
            pl.Series(name="model_name", values=['als'] * recommend_items.flatten().shape[0])
        ]
    )
    return recommends


def nearest_neighbours_inference(
    model: ItemItemRecommender, uim: csr_matrix, user_idxs: tp.List[int], model_name, n_candidates_default: int = 100
) -> pl.DataFrame:
    def compute(uim_sample: csr_matrix, user_idx: tp.List[int], model_sim: csr_matrix, n_cand: int):
        user_ids, item_ids, scores = find(uim_sample.dot(model_sim))
        recs = (
            pl.DataFrame(
                [
                    pl.Series(name="internal_receipt_id", values=user_ids, dtype=pl.Int64),
                    pl.Series(name="item_id_enc", values=item_ids, dtype=pl.Int64),
                    pl.Series(name="score", values=scores, dtype=pl.Float32),
                ]
            )
            .with_columns([pl.col("score").rank(method="ordinal", descending=True).over("internal_receipt_id").alias("rank")])
            .filter(pl.col("rank") <= n_cand)
            .join(
                pl.DataFrame(
                    [
                        pl.Series(name="receipt_id_enc", values=user_idx, dtype=pl.Int64),
                        pl.Series(name="internal_receipt_id", values=np.arange(len(user_idx)), dtype=pl.Int64),
                    ]
                ),
                on=["internal_receipt_id"],
            )
            .drop(["score", "rank", "internal_receipt_id"])
        )
        return recs

    recommends = [
        compute(uim[user_idx], user_idx, model.similarity, min(n_candidates_default, model.similarity.shape[1]))
        for user_idx in split_list_to_blocks(user_idxs, 1000)
    ]

    recommends_df: pl.DataFrame = pl.concat(recommends).select(
        [pl.col("receipt_id_enc").cast(pl.Int64), pl.col("item_id_enc").cast(pl.Int64)]
    ).with_columns([pl.lit(f"{model_name}").alias("model_name")])
    return recommends_df


def get_receipt_indexes(receipt_ids: pl.Series, encoder: tp.Dict[int, int]) -> tp.List[int]:
    receipt_indexes = []
    for receipt_id in receipt_ids.to_list():
        receipt_index = encoder.get(receipt_id)
        if receipt_index is not None:
            receipt_indexes.append(receipt_index)
    return receipt_indexes


def decode(table: pl.DataFrame, encoder: tp.Dict[int, int], key: str, enc_key: str) -> pl.DataFrame:
    encoder_table = encoder2df(encoder=encoder, key=key, enc_key=enc_key)
    table = table.join(encoder_table, on=[enc_key]).drop(enc_key)
    return table