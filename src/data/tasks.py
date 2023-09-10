from typing import Dict, Optional, Tuple

import polars as pl
from scipy.sparse import csr_matrix
from data.utils import prepare_rpi, create_sparse_matrices, get_pairs_with_context

def load_data(train_data_path: str, val_data_path: Optional[str] = None) -> Tuple[pl.DataFrame, csr_matrix, csr_matrix, Dict]:
    train_li = pl.read_csv(train_data_path, separator="\t")
    if val_data_path:
        val_li = pl.read_csv(val_data_path, separator="\t")
        full_li = pl.concat((train_li, val_li))
    else:
        full_li = train_li
    rpi = prepare_rpi(full_li)
    spmat_norm, spmat, encoders = create_sparse_matrices(rpi)
    return rpi, spmat, spmat_norm, encoders


def generate_features(train: pl.DataFrame, val: pl.DataFrame) -> pl.DataFrame:
    mapping, full_li = get_pairs_with_context(train, val)

    prices = full_li.unique(subset=["item_id", "price", "quantity"]).select(["item_id", "price"]).groupby("item_id").agg(pl.col("price").max())
    quantities = full_li.unique(subset=["item_id", "price", "quantity"]).select(["item_id", "quantity"]).groupby("item_id").agg(pl.col("quantity").sum())

    return mapping.explode("context").join(
            prices.rename({"price": "context_price"}),
            left_on="context",
            right_on="item_id"
        ).join(
            prices.rename({"price": "pos_price"}),
            left_on="positives",
            right_on="item_id"
        ).join(
            prices.rename({"price": "neg_price"}),
            left_on="negatives",
            right_on="item_id"
        ).join(
            quantities.rename({"quantity": "context_quantity"}),
            left_on="context",
            right_on="item_id"
        ).join(
            quantities.rename({"quantity": "pos_quantity"}),
            left_on="positives",
            right_on="item_id"
        ).join(
            quantities.rename({"quantity": "neg_quantity"}),
            left_on="negatives",
            right_on="item_id"
        ).groupby("receipt_id").agg(
            [pl.col("context")] + \
            [pl.col(col).first() for col in ["positives", "negatives"]] + \
            [pl.col(col).mean() for col in ["pos_price", "neg_price", "context_price"]] + \
            [pl.col(col).mean() for col in ["pos_quantity", "neg_quantity", "context_quantity"]]
        ), prices, quantities

def join_candidates_features(candidates: pl.DataFrame, prices: pl.DataFrame, quantities: pl.DataFrame) -> pl.DataFrame:
    return candidates.join(
            prices,
            how="left",
            on="item_id"
        ).join(
            quantities, 
            how="left",
            on="item_id"
        )

def join_context_features(context: pl.DataFrame, prices: pl.DataFrame, quantities: pl.DataFrame) -> pl.DataFrame:
    return context.explode("context").join(
            prices.rename({"price": "context_price"}),
            how="left",
            left_on="context",
            right_on="item_id"
        ).join(
            quantities.rename({"quantity": "context_quantity"}), 
            how="left",
            left_on="context",
            right_on="item_id"
        ).groupby("receipt_id").agg(
            [pl.col("context")] + \
            [pl.col(col).mean() for col in ["context_price", "context_quantity"]]
        )