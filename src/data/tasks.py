from typing import Dict, Optional, Tuple

import polars as pl
from scipy.sparse import csr_matrix
from data.utils import prepare_rpi, create_sparse_matrices

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