"""Find real WANDS queries where BM25 ranks an Exact match first but RRF pushes it down."""
import contextlib, io, runpy, sys
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path("bench/evaluate.py", run_name="examples")
qids, rel, qtext, TOP_BM, TOP_VE, rrf, p, res = (g[k] for k in "qids rel qtext TOP_BM TOP_VE rrf p res".split())
out = []
for x in qids:
    d0 = int(TOP_BM[x][0])
    if rel[x].get(d0, 0) != 2:
        continue
    n_exact = sum(v == 2 for v in rel[x].values())
    fused = [int(d) for d in rrf(x)]
    pos = fused.index(d0) + 1
    vpos = [int(d) for d in TOP_VE[x]].index(d0) + 1 if d0 in [int(d) for d in TOP_VE[x]] else None
    if pos > 3:
        out.append((n_exact, pos, x, qtext[x], p.product_name.iat[d0], vpos,
                    [(p.product_name.iat[d], rel[x].get(d, 0), [int(e) for e in TOP_BM[x]].index(d) + 1 if d in [int(e) for e in TOP_BM[x]] else None,
                      [int(e) for e in TOP_VE[x]].index(d) + 1 if d in [int(e) for e in TOP_VE[x]] else None) for d in fused[:5]],
                    round(res["BM25 only"][x], 3), round(res["RRF (k=60)"][x], 3), round(res["Vector only"][x], 3)))
out.sort()
for o in out[:15]:
    print(o[:6], "ndcg bm25/rrf/vec", o[7:])
    for r in o[6]:
        print("     ", r)
