from datetime import datetime
from typing import Dict

import polars as pl
from typer import Option, Typer


cli = Typer()


@cli.command()
def load():
    pass


@cli.command()
def train_candidate_models():
    pass


@cli.command()
def inference_candidates():
    pass


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
