import joblib 

from datetime import datetime
from typing import Dict, Optional

from tqdm import tqdm
from catboost import CatBoostRanker
import polars as pl
from typer import Option, Typer
from configs.schema import DataSchema
from data.tasks import load_data, join_candidates_features, join_context_features, generate_features
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
        candidates.write_csv(schema.target_paths[f"data.candidates_{model_name}"])

    candidates = union_candidates(list(candidates_by_model.values()))
    candidates.write_csv(schema.target_paths["data.candidates"])


@cli.command()
def train_ranker(
    train_data_path: str = Option(..., envvar="TRAIN_DATA_PATH"),
    val_data_path: Optional[str] = Option(default=None, envvar="VAL_DATA_PATH")
    ):

    schema = DataSchema()
    train_li = pl.read_csv(train_data_path, separator="\t")
    val_li = pl.read_csv(val_data_path, separator="\t")
    ds, prices, quantities = generate_features(train=train_li, val=val_li)

    pos_ds = ds.select(["positives", "receipt_id", "pos_price", "pos_quantity", "context_price", "context_quantity"]).with_columns(pl.lit(1).alias("target")).rename({
        "positives": "item_id",
        "pos_price": "price",
        "pos_quantity": "quantity",
    })
    neg_ds = ds.select(["negatives", "receipt_id", "neg_price", "neg_quantity", "context_price", "context_quantity"]).with_columns(pl.lit(0).alias("target")).rename({
        "negatives": "item_id",
        "neg_price": "price",
        "neg_quantity": "quantity",
    })
    ds = pl.concat((pos_ds, neg_ds)).sort("receipt_id")
    
    ranker = CatBoostRanker(verbose=250)
    ranker.fit(X=ds.select(["price", "quantity", "context_price", "context_quantity"]).to_pandas(), y=ds["target"].to_pandas(), group_id=ds["receipt_id"].to_pandas())
    joblib.dump(ranker, schema.target_paths["models.ranker"])
    prices.write_csv(schema.target_paths["data.prices"])
    quantities.write_csv(schema.target_paths["data.quantities"])


@cli.command()
def make_recommendations(
    val_data_path: Optional[str] = Option(default=None, envvar="VAL_DATA_PATH")
):
    schema = DataSchema()
    context_df = pl.read_csv(val_data_path, separator="\t").select(["receipt_id", "item_id"]).group_by("receipt_id").agg(pl.col("item_id").alias("context"))
    candidates = pl.read_csv(schema.target_paths["data.candidates"]).select(["receipt_id", "item_id"])
    prices = pl.read_csv(schema.target_paths["data.prices"])
    quantities = pl.read_csv(schema.target_paths["data.quantities"])
    ranker = joblib.load(schema.target_paths["models.ranker"])

    context_with_features = join_context_features(context=context_df, prices=prices, quantities=quantities)
    candidates_with_features = join_candidates_features(candidates=candidates, prices=prices, quantities=quantities)
    ds = context_with_features.join(candidates_with_features, on="receipt_id", how="left")
   
    predictions = {
        "receipt_id": [],
        "item_id": [],
        "score": []
    }
    for receipt_id in tqdm(ds["receipt_id"].unique()):
        receipt_items = ds.filter(pl.col("receipt_id") == receipt_id)
        score = ranker.predict(receipt_items.select(["price", "quantity", "context_price", "context_quantity"]).to_pandas())
        predictions["receipt_id"].append(receipt_id)
        predictions["item_id"].append(receipt_items["item_id"].to_list())
        predictions["score"].append(score)


    predictions = pl.DataFrame(predictions)

    pred_final = predictions.explode(["item_id", "score"]).sort(["receipt_id", "score"], descending=True).group_by("receipt_id").agg(pl.col("item_id").first())
    pred_final.write_csv(schema.target_paths["data.recommendations"])


@cli.command()
def evaluate_common_metrics(
    target_path: str = Option(..., envvar="TARGET_PATH"),
):
    schema = DataSchema()
    val_target = pl.read_csv(target_path, separator='\t')
    recs = pl.read_csv(schema.target_paths["data.recommendations"]).rename({"item_id":"pred_id"})
    res = val_target.join(recs, on="receipt_id", how="left").filter(pl.col("item_id") == pl.col("pred_id"))
    print("accuracy = ", res.shape[0] / val_target.shape[0])

if __name__ == "__main__":
    cli()