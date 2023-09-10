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
        row_sums = df.group_by(row).agg(pl.col(score).sum().alias("sum"))
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


def get_pairs_with_context(
    train: pl.DataFrame,
    val: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    def sample_neg(cart_items, items, probs):
        neg_item = np.random.choice(items, p=probs)
        while neg_item in cart_items:
            neg_item = np.random.choice(items, p=probs)
        return neg_item

    full_li = pl.concat((train, val))

    tmp = full_li["item_id"].value_counts().with_columns((pl.col("counts") / pl.col("counts").sum()).alias("prob"))
    items = tmp["item_id"].to_numpy()
    probs = tmp["prob"].to_numpy()

    r2i_desc = full_li.group_by(["receipt_id", "item_id"]).agg(pl.col("quantity").sum())

    non_empty_cart = r2i_desc.group_by(["receipt_id"]).agg(pl.col("item_id")).filter(pl.col("item_id").list.lengths() > 1)

    context = []
    positives = []
    negatives = []
    for cart_items in non_empty_cart["item_id"]:
        items_np = np.asarray(cart_items)
        mask = np.zeros(len(cart_items)).astype(bool)
        pos_idx = np.random.randint(0, items_np.shape[0])
        mask[pos_idx] = True

        context.append(items_np[~mask])
        positives.append(items_np[pos_idx])

        neg_item = sample_neg(cart_items, items, probs)
        negatives.append(neg_item)

    res = pl.DataFrame({
        "receipt_id": non_empty_cart["receipt_id"],
        "context": context,
        "positives": positives,
        "negatives": negatives
    })

    return res, full_li
