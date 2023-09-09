from dataclasses import dataclass
from datetime import datetime
from os.path import join


@dataclass
class DataSchema():
    cache_dir = "./cache/"
    export_dir = "./export/"
    def __init__(self):
        super().__init__()
        self.target_paths = {
            # cache
            "data.line_items": join(self.cache_dir, "line_items.csv"),
            "data.rpi": join(self.cache_dir, "rpi.csv"),
            "data.candidates": join(self.cache_dir, "candidates.pq"),
            "data.candidates_tfidf": join(self.cache_dir, "candidates_tfidf.pq"),
            "data.candidates_cosine": join(self.cache_dir, "candidates_cosine.pq"),
            "data.candidates_als": join(self.cache_dir, "candidates_als.pq"),
            "data.candidates_bm25": join(self.cache_dir, "candidates_bm25.pq"),
            "data.candidates_with_features": join(self.cache_dir, "candidates_with_features.pq"),
            "data.recommendations": join(self.cache_dir, "recommendations.pq"),

            # export
            "models.spmat_norm": join(self.export_dir, "spmat_norm.jlb"),
            "models.spmat": join(self.export_dir, "spmat.jlb"),
            "models.implicit_models": join(self.export_dir, "implicit_models.jlb"),
            "models.encoders": join(self.export_dir, "encoders.jlb"),
            "models.ranker": join(self.export_dir, "rank_model.jlb"),
            "metrics.common": join(self.export_dir, "common_metrics.jlb"),
            "metrics.candidates": join(self.export_dir, "candidates_metrics.jlb"),
        }