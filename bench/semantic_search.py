"""Sprint 14 exit criterion #1 — semantic search relevance benchmark.

Runs each query in the fixture through both `repo_query.semantic_search`
(hybrid BM25 + vector + RRF) and `repo_query.find_symbol` (Cypher
substring match) and reports hit@10 + MRR side-by-side.

Usage:
    NEO4J_URL=bolt+s://... NEO4J_PASSWORD=... OPENAI_API_KEY=... \
        python -m bench.semantic_search

Prereqs:
    - twai-swarm scanned with `--with-embeddings` so Function/Class
      nodes have the `embedding` property populated.
    - The full-text index `function_text` / `class_text` exists
      (created by loader.ensure_constraints automatically).

Pass criteria (per Sprint 11-14 plan exit criteria):
    - hybrid aggregate MRR > Cypher aggregate MRR
    - hybrid aggregate hit@10 >= Cypher aggregate hit@10

Exit codes:
    0  — both criteria met
    1  — hybrid did not beat Cypher on aggregate
    2  — runtime error (env, fixture, etc.)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from app import repo_query
from app.repo_indexer.loader import driver_from_env

REPO = "twai-swarm"
TOP_K = 10
FIXTURE = Path(__file__).parent / "fixtures" / "twai_swarm_semantic.yaml"


@dataclass
class QueryResult:
    query: str
    kind: str
    relevant_qns: set[str]
    top_choice: str
    hybrid_qns: list[str]
    cypher_qns: list[str]

    def hit_at_k(self, qns: list[str]) -> float:
        if not self.relevant_qns:
            return 0.0
        hits = sum(1 for q in qns[:TOP_K] if q in self.relevant_qns)
        return hits / len(self.relevant_qns)

    def mrr(self, qns: list[str]) -> float:
        for rank, qn in enumerate(qns[:TOP_K], start=1):
            if qn == self.top_choice:
                return 1.0 / rank
        return 0.0


def load_fixture() -> list[dict]:
    if not FIXTURE.exists():
        print(f"FATAL: fixture not found: {FIXTURE}", file=sys.stderr)
        sys.exit(2)
    with FIXTURE.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list) or not data:
        print("FATAL: fixture is empty or malformed", file=sys.stderr)
        sys.exit(2)
    return data


def run_query(driver, query: str) -> tuple[list[str], list[str]]:
    hybrid = repo_query.semantic_search(driver, REPO, query, k=TOP_K)
    cypher = repo_query.find_symbol(driver, REPO, query, limit=TOP_K)
    return (
        [h.qualified_name for h in hybrid],
        [c.qualified_name for c in cypher],
    )


def main() -> int:
    for var in ("NEO4J_URL", "NEO4J_PASSWORD"):
        if not os.getenv(var):
            print(f"FATAL: ${var} not set", file=sys.stderr)
            return 2

    fixture = load_fixture()
    results: list[QueryResult] = []

    print(f"Running {len(fixture)} queries against repo={REPO} (k={TOP_K})...")
    print()

    with driver_from_env() as driver:
        for entry in fixture:
            q = entry["query"]
            try:
                hybrid_qns, cypher_qns = run_query(driver, q)
            except Exception as exc:
                print(f"  query failed: {q!r}: {exc}", file=sys.stderr)
                hybrid_qns, cypher_qns = [], []

            results.append(QueryResult(
                query=q,
                kind=entry.get("kind", "unknown"),
                relevant_qns=set(entry.get("relevant_qns", [])),
                top_choice=entry.get("top_choice", ""),
                hybrid_qns=hybrid_qns,
                cypher_qns=cypher_qns,
            ))

    # Per-query table
    print(f"{'kind':<11} {'hit@10':>14} {'mrr':>14}    query")
    print(f"{'':<11} {'hyb':>6} {'cyp':>6}  {'hyb':>6} {'cyp':>6}")
    print("-" * 100)
    h_hit_total = c_hit_total = h_mrr_total = c_mrr_total = 0.0
    for r in results:
        h_hit = r.hit_at_k(r.hybrid_qns)
        c_hit = r.hit_at_k(r.cypher_qns)
        h_mrr = r.mrr(r.hybrid_qns)
        c_mrr = r.mrr(r.cypher_qns)
        h_hit_total += h_hit
        c_hit_total += c_hit
        h_mrr_total += h_mrr
        c_mrr_total += c_mrr
        marker = ""
        if h_mrr > c_mrr:
            marker = "  ^"
        elif h_mrr < c_mrr:
            marker = "  v"
        print(f"{r.kind:<11} {h_hit:>6.2f} {c_hit:>6.2f}  {h_mrr:>6.2f} {c_mrr:>6.2f}    {r.query[:60]}{marker}")
    print("-" * 100)

    n = len(results)
    h_hit_avg = h_hit_total / n
    c_hit_avg = c_hit_total / n
    h_mrr_avg = h_mrr_total / n
    c_mrr_avg = c_mrr_total / n

    # Aggregate by kind for diagnostic context.
    by_kind: dict[str, list[QueryResult]] = {}
    for r in results:
        by_kind.setdefault(r.kind, []).append(r)
    print()
    print("Aggregate (overall):")
    print(f"  hybrid: hit@10={h_hit_avg:.3f}  MRR={h_mrr_avg:.3f}")
    print(f"  cypher: hit@10={c_hit_avg:.3f}  MRR={c_mrr_avg:.3f}")
    print()
    print("Aggregate (by kind):")
    for kind, rs in sorted(by_kind.items()):
        h_hit_k = sum(r.hit_at_k(r.hybrid_qns) for r in rs) / len(rs)
        c_hit_k = sum(r.hit_at_k(r.cypher_qns) for r in rs) / len(rs)
        h_mrr_k = sum(r.mrr(r.hybrid_qns) for r in rs) / len(rs)
        c_mrr_k = sum(r.mrr(r.cypher_qns) for r in rs) / len(rs)
        print(f"  {kind:<11} (n={len(rs):>2}): "
              f"hyb hit@10={h_hit_k:.3f} MRR={h_mrr_k:.3f} | "
              f"cyp hit@10={c_hit_k:.3f} MRR={c_mrr_k:.3f}")
    print()

    # Pass / fail evaluation against the plan's exit criteria.
    mrr_pass = h_mrr_avg > c_mrr_avg
    hit_pass = h_hit_avg >= c_hit_avg
    print("Exit criteria:")
    print(f"  hybrid MRR > cypher MRR    : {'PASS' if mrr_pass else 'FAIL'}  ({h_mrr_avg:.3f} vs {c_mrr_avg:.3f})")
    print(f"  hybrid hit@10 >= cypher    : {'PASS' if hit_pass else 'FAIL'}  ({h_hit_avg:.3f} vs {c_hit_avg:.3f})")

    return 0 if (mrr_pass and hit_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
