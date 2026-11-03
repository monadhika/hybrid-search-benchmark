"""Embed WANDS products and queries with an off-the-shelf embedding model; cache to .npy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: " if "bge" in MODEL else ""

p = pd.read_csv(DATA / "product.csv", sep="\t").fillna("")
q = pd.read_csv(DATA / "query.csv", sep="\t")
docs = (p.product_name + ". " + p.product_class + ". " + p["category hierarchy"] + ". " + p.product_description).tolist()

m = SentenceTransformer(MODEL, device="mps")
m.max_seq_length = 256
tag = MODEL.split("/")[-1]
np.save(OUT / f"q_{tag}.npy", m.encode([QUERY_PREFIX + s for s in q["query"]], normalize_embeddings=True, batch_size=128))
np.save(OUT / f"d_{tag}.npy", m.encode(docs, normalize_embeddings=True, batch_size=128, show_progress_bar=True))
print("done", tag)
