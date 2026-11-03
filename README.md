# Hybrid search benchmark on WANDS

How should you combine BM25 and vector search? This benchmark compares the common ways of doing it on [WANDS](https://github.com/wayfair/WANDS), Wayfair's public product search dataset: 480 real queries, 43,000 products and 233,000 human relevance labels.

It accompanies the article **[Hybrid search: how to combine BM25 and vector search without losing the best result](https://infersense-labs.ai/blog/hybrid-search-bm25-vector-search/)**.

## What it compares

| Method | What it does |
|---|---|
| BM25 only | Keyword search over product name, class, category and description (bm25s, English stemming and stopwords) |
| Vector only | Cosine similarity with an off-the-shelf embedding model |
| RRF (k = 60) | Reciprocal rank fusion of the two top-100 lists |
| Late weighted sum | Min-max normalised scores, one tuned α. A document missing from one list gets 0 for that side, which is all a late merge knows |
| Single-pass hybrid | The same weighted sum, but every product gets both scores |
| Single-pass, α per query type | A separate α for named or rare-word queries, long queries (5+ words) and the rest |
| Learned ranker | Gradient-boosted trees (pointwise) over the union of both top-100 lists, with score, rank and name-match features |

Two embedding models: `BAAI/bge-base-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`. The metric is nDCG@10 with Exact = 2, Partial = 1, Irrelevant = 0. Every α and the ranker are tuned with 5-fold cross-validation over queries, so no query is scored with settings tuned on itself.

## Results

nDCG@10 × 100, all 480 queries:

| Method | bge-base-en-v1.5 | all-MiniLM-L6-v2 |
|---|---|---|
| BM25 only | 68.9 | 68.9 |
| Vector only | 71.2 | 66.7 |
| RRF (k = 60) | 73.6 | 72.5 |
| Late weighted sum | 73.4 | 71.9 |
| Single-pass hybrid | 74.1 | 73.1 |
| Single-pass, α per query type | 74.1 | 73.1 |
| Learned ranker | 74.9 | 74.2 |

Main findings:

- Hybrid search beat both single retrievers with both models, by 2.4 (bge-base) and 3.6 (MiniLM) points over the better one.
- With the same formula, giving every candidate both scores beat merging two separately scored lists: +0.7 points with bge-base and +1.2 with MiniLM. Paired bootstrap 95% intervals were [0.2, 1.1] and [0.7, 1.8].
- The best α differed by model: 0.7 on the vector side with bge-base, 0.6 with MiniLM.
- A separate α per query type made no difference on WANDS, whose queries are mostly descriptive.
- Every method made dozens of individual queries clearly worse (by more than 0.1 nDCG@10) while improving more of them. Against vector search alone with bge-base: RRF 79 better and 46 worse; the learned ranker 106 better and 48 worse.

Per-slice results, helped/hurt counts, the tuned α values and the "regner power loom red" example are in [`results/results.json`](results/results.json).

**Caveats:** one dataset, one domain (furniture and home goods), and almost no SKU or model-number queries. The ranker is small and trained on 480 queries. Differences below about half a point are within noise. The learned ranker varies by about ±0.1 points between runs.

## Run it

Python 3.9 or later. Embedding 43,000 products takes a few minutes on an Apple Silicon GPU (`device="mps"` in `bench/embed.py`; change it to `"cuda"` or `"cpu"` as needed).

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./download_data.sh

python bench/embed.py data/wands bench                                         # bge-base-en-v1.5
python bench/embed.py data/wands bench sentence-transformers/all-MiniLM-L6-v2

python bench/evaluate.py data/wands bench                                      # results as JSON (bge-base)
python bench/evaluate.py data/wands bench all-MiniLM-L6-v2                     # results as JSON (MiniLM)
python bench/sig.py data/wands bench                                           # paired bootstrap intervals
python bench/examples.py data/wands bench                                      # queries where RRF buries BM25's exact #1
python bench/export.py results/results.json                                    # everything, both models
```

The query-type slices use `/usr/share/dict/words` to spot names and rare words, which exists on macOS and most Linux systems.

## Data and citation

WANDS is released by Wayfair under the MIT license: Chen, Y., Liu, S., Liu, Z., Sun, W., Baltrunas, L., and Schroeder, B. *WANDS: Dataset for Product Search Relevance Assessment.* ECIR 2022.

Benchmark code © 2026 Infersense Labs LLC, MIT license. Written by [Monadhika Sharma](https://infersense-labs.ai/about/).
