import contextlib, io, runpy, sys, json
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("bench/evaluate.py", run_name="examples")
res, qids = g["res"], g["qids"]
rng = np.random.default_rng(1)
B = rng.integers(0, len(qids), (10000, len(qids)))
def diff(a, b):
    d = np.array([res[a][x] - res[b][x] for x in qids])
    bs = d[B].mean(1)
    return round(d.mean()*100, 2), round(np.percentile(bs, 2.5)*100, 2), round(np.percentile(bs, 97.5)*100, 2), int((d > 0.1).sum()), int((d < -0.1).sum())
for a, b in [("RRF (k=60)", "Vector only"), ("RRF (k=60)", "BM25 only"), ("Single-pass hybrid, one α", "RRF (k=60)"),
             ("Late weighted sum, one α", "RRF (k=60)"), ("Single-pass hybrid, one α", "Late weighted sum, one α"),
             ("Single-pass, α per query type", "Single-pass hybrid, one α"), ("Learned ranker on both legs", "Single-pass hybrid, one α"),
             ("Learned ranker on both legs", "RRF (k=60)")]:
    print(g["TAG"], f"{a} - {b}:", diff(a, b))
