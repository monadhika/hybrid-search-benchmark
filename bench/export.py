"""Run both embedding models and write all numbers (nDCG tables, helped/hurt counts, the Regner example) to one JSON."""
import contextlib, io, json, runpy, sys
import numpy as np
out_path = sys.argv[1] if len(sys.argv) > 1 else "results/results.json"
allres = {}
for tag in ["bge-base-en-v1.5", "all-MiniLM-L6-v2"]:
    sys.argv = ["evaluate.py", "data/wands", "bench", tag]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g = runpy.run_path("bench/evaluate.py", run_name="examples")
    r = json.loads(buf.getvalue())
    res, qids = g["res"], g["qids"]
    r["helped_hurt_vs_vector"] = {m: [int(sum(d[x] > res["Vector only"][x] + 0.1 for x in qids)),
                                      int(sum(d[x] < res["Vector only"][x] - 0.1 for x in qids))] for m, d in res.items()}
    r["helped_hurt_vs_bm25"] = {m: [int(sum(d[x] > res["BM25 only"][x] + 0.1 for x in qids)),
                                    int(sum(d[x] < res["BM25 only"][x] - 0.1 for x in qids))] for m, d in res.items()}
    r.pop("named_examples", None)
    if tag.startswith("bge"):
        x, p, rel = 97, g["p"], g["rel"]
        def lst(ids):
            return [[p.product_name.iat[int(d)], rel[x].get(int(d), 0)] for d in ids]
        fused = [int(d) for d in g["rrf"](x)]
        d0 = int(g["TOP_BM"][x][0])
        r["regner"] = {"query": g["qtext"][x], "bm25": lst(g["TOP_BM"][x][:3]), "vector": lst(g["TOP_VE"][x][:3]),
                       "rrf": lst(fused[:3]), "exact_rrf_rank": fused.index(d0) + 1,
                       "exact_vector_rank": int(list(np.argsort(-g["VE"][x])).index(d0) + 1)}
    allres[tag] = r
json.dump(allres, open(out_path, "w"), indent=1, ensure_ascii=False, default=lambda o: o.item())
print(json.dumps(allres["bge-base-en-v1.5"]["regner"], indent=1))
for t, r in allres.items():
    print(t, r["helped_hurt_vs_vector"], r["helped_hurt_vs_bm25"])
