"""Hybrid search benchmark on WANDS (Wayfair ANnotation DataSet).

Compares BM25, vector search, RRF, late weighted fusion, single-pass hybrid scoring,
query-type weighting and a small learned ranker. nDCG@10 with Exact=2, Partial=1.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from sklearn.ensemble import HistGradientBoostingRegressor

DATA, BENCH = Path(sys.argv[1]), Path(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else "bge-base-en-v1.5"
K, RRF_K, FOLDS, SEED = 100, 60, 5, 0
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)

p = pd.read_csv(DATA / "product.csv", sep="\t").fillna("")
q = pd.read_csv(DATA / "query.csv", sep="\t")
lab = pd.read_csv(DATA / "label.csv", sep="\t")
GAIN = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
rel = defaultdict(dict)
for qid, pid, l in zip(lab.query_id, lab.product_id, lab.label):
    rel[qid][pid] = GAIN[l]
qids = q.query_id.tolist()
qtext = dict(zip(q.query_id, q["query"]))

# ---------- query slices ----------
words = {w.strip().lower() for w in open("/usr/share/dict/words")}
name_tokens = set(" ".join(p.product_name).lower().split())


def is_named(tok):
    if re.fullmatch(r"[a-z]*\d[a-z\d\-/.]*", tok) and re.search(r"[a-z]", tok):
        return True  # alphanumeric code like w003096809, 7qt
    if not tok.isalpha() or len(tok) < 3:
        return False
    return tok not in words and tok.rstrip("s") not in words and tok in name_tokens


SLICES = {
    "Named products and brands": lambda s: any(is_named(t) for t in s.lower().split()),
    "Numbers and sizes": lambda s: bool(re.search(r"\d", s)),
    "Short (1–2 words)": lambda s: len(s.split()) <= 2,
    "Long (5+ words)": lambda s: len(s.split()) >= 5,
}
slice_of = {name: [qid for qid in qids if f(qtext[qid])] for name, f in SLICES.items()}
qtype = {}
for qid in qids:
    s = qtext[qid]
    qtype[qid] = "named" if SLICES["Named products and brands"](s) else ("long" if len(s.split()) >= 5 else "other")

# ---------- retrievers: full-corpus scores for both legs ----------
stemmer = Stemmer.Stemmer("english")
corpus = (p.product_name + " " + p.product_class + " " + p["category hierarchy"] + " " + p.product_description).tolist()
retr = bm25s.BM25()
retr.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, show_progress=False), show_progress=False)
qtok = bm25s.tokenize([qtext[i] for i in qids], stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)
BM = {qid: retr.get_scores(toks) for qid, toks in zip(qids, qtok)}
QE, DE = np.load(BENCH / f"q_{TAG}.npy"), np.load(BENCH / f"d_{TAG}.npy")
VE = {qid: DE @ QE[i] for i, qid in enumerate(qids)}


def topk(scores, k=K):
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]


TOP_BM = {qid: topk(BM[qid]) for qid in qids}
TOP_VE = {qid: topk(VE[qid]) for qid in qids}


def ndcg(qid, ranked, k=10):
    r = rel[qid]
    dcg = sum((2 ** r.get(int(d), 0) - 1) / np.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(r.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# ---------- fusion methods ----------
def rrf(qid):
    s = defaultdict(float)
    for lst in (TOP_BM[qid], TOP_VE[qid]):
        for r, d in enumerate(lst):
            s[d] += 1 / (RRF_K + r + 1)
    return sorted(s, key=s.get, reverse=True)


def late(qid, a):
    """Late fusion: each leg only knows its own top-K; a document missing from a leg gets 0 there."""
    b = dict(zip(TOP_BM[qid], minmax(BM[qid][TOP_BM[qid]])))
    v = dict(zip(TOP_VE[qid], minmax(VE[qid][TOP_VE[qid]])))
    s = {d: a * v.get(d, 0) + (1 - a) * b.get(d, 0) for d in set(b) | set(v)}
    return sorted(s, key=s.get, reverse=True)


NB = {qid: BM[qid] / BM[qid].max() if BM[qid].max() > 0 else BM[qid] for qid in qids}
NV = {qid: minmax(VE[qid]) for qid in qids}


def single_pass(qid, a):
    """Single-pass hybrid: every document gets both scores in one scoring pass over the corpus."""
    return topk(a * NV[qid] + (1 - a) * NB[qid], 10)


rng = np.random.default_rng(SEED)
fold_of = dict(zip(qids, rng.permutation(np.arange(len(qids)) % FOLDS)))


def cv_alpha(fn, group=None):
    """Pick alpha on 4 folds, evaluate on the 5th. group(qid) -> key tunes a separate alpha per query type."""
    group = group or (lambda qid: "all")
    table = {a: {qid: ndcg(qid, fn(qid, a)) for qid in qids} for a in ALPHAS}
    out, chosen = {}, defaultdict(list)
    for f in range(FOLDS):
        train = [x for x in qids if fold_of[x] != f]
        for g in {group(x) for x in qids}:
            tr = [x for x in train if group(x) == g] or train
            best = max(ALPHAS, key=lambda a: np.mean([table[a][x] for x in tr]))
            chosen[g].append(float(best))
            for x in qids:
                if fold_of[x] == f and group(x) == g:
                    out[x] = table[best][x]
    return out, {g: v for g, v in chosen.items()}


# ---------- learned ranker over the union of both legs ----------
from collections import Counter
_df = Counter(t for n in p.product_name for t in set(n.lower().split()))
IDF = {t: np.log(len(p) / c) for t, c in _df.items()}


def features(qid):
    s = qtext[qid].lower()
    qt = set(s.split())
    cands = np.array(sorted(set(TOP_BM[qid]) | set(TOP_VE[qid])))
    rb = {d: r for r, d in enumerate(TOP_BM[qid])}
    rv = {d: r for r, d in enumerate(TOP_VE[qid])}
    rows = []
    for d in cands:
        name = set(p.product_name.iat[d].lower().split())
        cls = p.product_class.iat[d].lower()
        rows.append([
            NB[qid][d], NV[qid][d], BM[qid][d], VE[qid][d],
            rb.get(d, K), rv.get(d, K),
            len(qt & name) / len(qt), float(all(t in name for t in qt)), float(any(t in cls for t in qt)),
            sum(IDF.get(t, 0) for t in qt & name) / (sum(IDF.get(t, 10.7) for t in qt) or 1),
            max([IDF.get(t, 0) for t in qt & name] or [0]),
            len(qt), float(qtype[qid] == "named"), float(bool(re.search(r"\d", s))),
            np.log1p(p.rating_count.iat[d] or 0),
        ])
    return cands, np.array(rows)


FEAT = {qid: features(qid) for qid in qids}


def ranker_scores():
    out = {}
    for f in range(FOLDS):
        tr = [x for x in qids if fold_of[x] != f]
        X = np.vstack([FEAT[x][1] for x in tr])
        y = np.concatenate([[rel[x].get(int(d), 0) for d in FEAT[x][0]] for x in tr])
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED).fit(X, y)
        for x in qids:
            if fold_of[x] == f:
                c, F = FEAT[x]
                out[x] = ndcg(x, c[np.argsort(-m.predict(F))])
    return out


# ---------- run ----------
res = {
    "BM25 only": {x: ndcg(x, TOP_BM[x]) for x in qids},
    "Vector only": {x: ndcg(x, TOP_VE[x]) for x in qids},
    "RRF (k=60)": {x: ndcg(x, rrf(x)) for x in qids},
}
alphas = {}
res["Late weighted sum, one α"], alphas["late"] = cv_alpha(late)
res["Single-pass hybrid, one α"], alphas["single"] = cv_alpha(single_pass)
res["Single-pass, α per query type"], alphas["by_type"] = cv_alpha(single_pass, lambda x: qtype[x])
res["Learned ranker on both legs"] = ranker_scores()


def mean(d, ids):
    return float(np.mean([d[x] for x in ids]))


groups = {"All queries": qids, **slice_of}
table = {m: {g: round(mean(d, ids), 3) for g, ids in groups.items()} for m, d in res.items()}
hurt = {m: sum(d[x] < res["BM25 only"][x] - 0.1 for x in qids) for m, d in res.items()}
hurt_v = {m: sum(d[x] < res["Vector only"][x] - 0.1 for x in qids) for m, d in res.items()}


def recall_exact(lst, qid):
    ex = {d for d, g in rel[qid].items() if g == 2}
    return len(ex & {int(d) for d in lst}) / min(len(ex), K) if ex else None


rec = {}
for name, fn in {"BM25": lambda x: TOP_BM[x], "Vector": lambda x: TOP_VE[x],
                 "Union": lambda x: list(set(TOP_BM[x]) | set(TOP_VE[x]))}.items():
    vals = [recall_exact(fn(x), x) for x in qids]
    rec[name] = round(float(np.mean([v for v in vals if v is not None])), 3)
overlap = round(float(np.mean([len(set(TOP_BM[x]) & set(TOP_VE[x])) / K for x in qids])), 3)

# trap: BM25's #1 is an Exact match; where does it land after fusion?
trap_q = [x for x in qids if rel[x].get(int(TOP_BM[x][0]), 0) == 2]
trap = {}
for m, fn in {"RRF (k=60)": rrf, "Single-pass hybrid": lambda x: list(single_pass(x, np.median(alphas["single"]["all"])))}.items():
    pos = []
    for x in trap_q:
        lst = [int(d) for d in fn(x)]
        d0 = int(TOP_BM[x][0])
        pos.append(lst.index(d0) + 1 if d0 in lst else 999)
    trap[m] = {"queries": len(trap_q), "still #1": sum(p_ == 1 for p_ in pos), "out of top 3": sum(p_ > 3 for p_ in pos),
               "out of top 10": sum(p_ > 10 for p_ in pos)}

print(json.dumps({"model": TAG, "slice_sizes": {g: len(i) for g, i in groups.items()}, "ndcg@10": table,
                  "hurt_vs_bm25(>0.1)": hurt, "hurt_vs_vector(>0.1)": hurt_v, "exact_recall@100": rec,
                  "top100_overlap": overlap, "alphas": alphas, "trap": trap,
                  "named_examples": [qtext[x] for x in slice_of["Named products and brands"]][:40]}, indent=1, ensure_ascii=False, default=lambda o: o.item()))
