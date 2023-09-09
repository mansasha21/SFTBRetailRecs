import typing as tp
from typing import Dict, Tuple, Any

import polars as pl
import numpy as np
from scipy.sparse import csr_matrix

from data.proc_text import (
    process_sentence,
    get_sentence_embedding
)


def df2encoder(df: pl.DataFrame, key: str) -> tp.Dict[int, int]:
    return {key_id: enc_key_id for enc_key_id, key_id in enumerate(df[key].unique())}


def encoder2df(encoder: tp.Dict[int, int], key: str, enc_key: str, dtype: type = pl.Int64) -> pl.DataFrame:
    return pl.DataFrame(data=list(encoder.items()), schema={key: dtype, enc_key: dtype}, orient="row")


def df2sparse(df: pl.DataFrame, row: str, col: str, score: str, normalize=False) -> csr_matrix:
    if normalize:
        row_sums = df.groupby(row).agg(pl.col(score).sum().alias("sum"))
        df = df.join(row_sums, on=row)
        vls = (df[score] / df["sum"]).to_numpy()
    else:
        vls = df[score].to_numpy()
    return csr_matrix((vls, (df[row].to_numpy(), df[col].to_numpy())), dtype=np.float32)


def prepare_rpi(line_itmes: pl.DataFrame) -> pl.DataFrame:
    df = (
        line_itmes
        .group_by(["receipt_id", "item_id"])
        .agg(pl.col("quantity").sum().alias("interaction_score"))
        .filter(~pl.col("interaction_score").is_null())
    )
    df = df.join(
        (df.group_by("receipt_id").agg([pl.col("interaction_score").sum().alias("denum")])),
        on=["receipt_id"],
    ).with_columns([(pl.col("interaction_score") / pl.col("denum")).alias("interaction_score_norm")])
    return df


def create_sparse_matrices(rpi: pl.DataFrame) -> Tuple[csr_matrix, csr_matrix, Dict[Any, Any]]:

    encoders = {"receipt_id": df2encoder(rpi, "receipt_id"), "item_id": df2encoder(rpi, "item_id")}
    rpi = rpi.join(encoder2df(encoder=encoders["receipt_id"], key="receipt_id", enc_key="receipt_id_enc"), on=["receipt_id"])
    rpi = rpi.join(
        encoder2df(encoder=encoders["item_id"], key="item_id", enc_key="item_id_enc"), on=["item_id"]
    )

    spmat_norm: csr_matrix = df2sparse(df=rpi, row="receipt_id_enc", col="item_id_enc", score="interaction_score_norm")
    spmat: csr_matrix = df2sparse(df=rpi, row="receipt_id_enc", col="item_id_enc", score="interaction_score")
    return spmat_norm, spmat, encoders


def get_cart_features(data: pl.DataFrame) -> pl.DataFrame:
    data["price"] = data["price"] * data["quantity"]
    receipt_info = data.groupby(by="receipt_id").agg(
        {
            "server_date": max,
            "local_date": max,
            "item_id": (list, len),
            "name": list,
            "price": [list, max, min, np.mean],
            "quantity": [list, max, min, np.mean]
        }
    )

    receipt_info.columns = [
        "server_date", "local_date", "receipt_items",
        "cnt_items", "names", "prices", "max_price", "min_price",
        "mean_price", "quantities", "max_quantity", "min_quantity",
        "mean_quantity"
    ]
    
    receipt_info["prices"] = np.array(receipt_info["prices"])
    receipt_info["quantities"] = np.array(receipt_info["quantities"])
    
    receipt_info["percentile90_price"] = [np.percentile(p, 90) for p in receipt_info.prices]
    receipt_info["percentile95_price"] = [np.percentile(p, 95) for p in receipt_info.prices]
    receipt_info["percentile99_price"] = [np.percentile(p, 99) for p in receipt_info.prices]
    
    receipt_info["percentile90_quantity"] = [np.percentile(q, 90) for q in receipt_info.quantities]
    receipt_info["percentile95_quantity"] = [np.percentile(q, 95) for q in receipt_info.quantities]
    receipt_info["percentile99_quantity"] = [np.percentile(q, 99) for q in receipt_info.quantities]
    
    receipt_info["item_emb"] = [np.array([get_sentence_embedding(item) for item in ri]) for ri in receipt_info.names]
    receipt_info["emb"] = [np.mean(item, axis=0) for item in receipt_info["item_emb"]]

    receipt_info["cart2item_dot"] = [[np.dot(cart, item) for item in items] for items, cart in zip(receipt_info["item_emb"], receipt_info["emb"])]

    return receipt_info.reset_index().drop(columns=["names", "prices", "quantities"])
