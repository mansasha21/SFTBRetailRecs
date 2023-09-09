import joblib 

from datetime import datetime
from typing import Dict, Optional

import polars as pl
from typer import Option, Typer
from configs.schema import DataSchema
from data.tasks import load_data
from train.tasks import train_implicit_models
from inference.tasks import get_candidates, union_candidates
from inference.utils import decode, get_receipt_indexes

cli = Typer()


@cli.command()
def load(
    train_data_path: str = Option(..., envvar="TRAIN_DATA_PATH"),
    val_data_path: Optional[str] = Option(default=None, envvar="VAL_DATA_PATH")
):
    schema = DataSchema()
    rpi, spmat, spmat_norm, encoders = load_data(train_data_path=train_data_path, val_data_path=val_data_path)
    rpi.write_csv(schema.target_paths["data.rpi"])
    jlb_items = (("spmat", spmat), ("spmat_norm", spmat_norm), ("encoders", encoders))
    for name, item in jlb_items:
        joblib.dump(item, schema.target_paths[f"models.{name}"])


@cli.command()
def train_candidate_models():
    schema = DataSchema()
    spmat_norm = joblib.load(schema.target_paths["models.spmat_norm"])
    spmat = joblib.load(schema.target_paths["models.spmat"])
    models = train_implicit_models(spmat_norm=spmat_norm, spmat=spmat)
    joblib.dump(models, schema.target_paths["models.implicit_models"])


@cli.command()
def inference_candidates(
    inference_data_path: str = Option(..., envvar="INFERENCE_DATA_PATH"),
):
    schema = DataSchema()
    spmat_norm = joblib.load(schema.target_paths["models.spmat_norm"])
    spmat = joblib.load(schema.target_paths["models.spmat"])
    models = joblib.load(schema.target_paths["models.implicit_models"])
    encoders = joblib.load(schema.target_paths["models.encoders"])

    df = (
        pl.read_csv(inference_data_path, separator="\t")
            .group_by(["receipt_id", "item_id"])
            .agg([pl.col("quantity").sum()])
            .select(["item_id", "receipt_id"])
            .group_by("receipt_id")
            .agg(pl.col("item_id"))
    )

    user_indexes = get_receipt_indexes(receipt_ids=df["receipt_id"], encoder=encoders["receipt_id"])

    candidates_by_model = get_candidates(
        models=models, uim=spmat, uim_norm=spmat_norm, user_indexes=user_indexes
    )
    for model_name, candidates in candidates_by_model.items():
        candidates = decode(table=candidates, encoder=encoders["receipt_id"], key="receipt_id", enc_key="receipt_id_enc")
        candidates = decode(
            table=candidates, encoder=encoders["item_id"], key="item_id", enc_key="item_id_enc"
        )
        candidates_by_model[model_name] = candidates
        candidates.write_csv(path=schema.target_paths[f"data.candidates_{model_name}"])

    candidates = union_candidates(list(candidates_by_model.values()))
    candidates.write_csv(schema.target_paths["data.candidates"])


@cli.command()
def candidates_join_features():
    pass


@cli.command()
def train_ranker():
    pass

@cli.command()
def make_recommendations():
    pass


@cli.command()
def evaluate_common_metrics():
    pass


@cli.command()
def evaluate_candidates_metrics():
    pass

if __name__ == "__main__":
    cli()
