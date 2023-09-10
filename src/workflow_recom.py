from typer import Typer
from configs.schema import DataSchema

import polars as pl


def recom_by_receipt_id():
    schema = DataSchema()

    while True:
        receipt_id = input("Input receipt id: ['exit' to stop] ")

        if receipt_id == "exit": break
        receipt_id = int(receipt_id)

        recom: pl.DataFrame = pl.read_parquet(schema.target_paths["data.recommendations_10"])
    
        print(recom.filter(pl.col("receipt_id") == receipt_id))


if __name__ == "__main__":
    recom_by_receipt_id()