"""Sprint 13a — CommunityDetectPhase tests against synthetic IndexBatches.

Like test_finalize.py, we hand-build IndexBatches rather than parsing
source — keeps the algorithm under test free of tree-sitter + Neo4j
dependencies. Skips if `python-louvain` (the community detection runtime
dep) isn't installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# python-louvain is the runtime requirement; rustworkx is for the future
# preferred path but the phase falls back transparently when it lacks
# `community.louvain_communities`. We just need the louvain package to
# run any community-detection test at all.
pytest.importorskip("community")  # python-louvain installs as `community`
pytest.importorskip("networkx")   # transitive dep of python-louvain

from app.repo_indexer.actions import (  # noqa: E402
    CallEdge,
    ClassNode,
    FunctionNode,
    IndexBatch,
    RepoNode,
)
from app.repo_indexer.phases.community_detect import (  # noqa: E402
    CommunityDetectPhase,
    _cohesion,
    _label_for,
    _tokens_for_qn,
)
from app.repo_indexer.runner import PhaseContext  # noqa: E402


REPO = RepoNode(name="r", url="", commit_sha="", tenant_id="t1")


def _ctx(batch: IndexBatch) -> PhaseContext:
    """Bare minimum context — phase only reads `repo`, `batch`, `progress`."""
    return PhaseContext(
        repo=batch.repo,
        repo_root=Path("."),
        languages=("python",),
        batch=batch,
        progress=lambda _msg: None,
    )


def _fn(qn: str, file_path: str = "x.py") -> FunctionNode:
    return FunctionNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=1, line_end=2,
    )


def _cls(qn: str, file_path: str = "x.py") -> ClassNode:
    return ClassNode(
        repo="r", qualified_name=qn, name=qn.split(".")[-1],
        file_path=file_path, line_start=1, line_end=2,
    )


def _call(caller: str, callee: str) -> CallEdge:
    return CallEdge(repo="r", caller_qn=caller, callee_qn=callee, line=1)


# ---------------------------------------------------------------------------
# 1. Empty batch — phase does nothing.
# ---------------------------------------------------------------------------

def test_empty_batch_noop():
    batch = IndexBatch(repo=REPO)
    CommunityDetectPhase().run(_ctx(batch))
    assert batch.communities == []
    assert batch.member_of == []


# ---------------------------------------------------------------------------
# 2. Two clearly-separated clusters.
# ---------------------------------------------------------------------------

def test_two_clear_clusters():
    """Two 4-node fully-connected cliques joined by one bridge edge —
    Louvain should return exactly two communities of size 4 each."""
    batch = IndexBatch(repo=REPO)
    cluster_a = ["app.a.alpha", "app.a.beta", "app.a.gamma", "app.a.delta"]
    cluster_b = ["app.b.one", "app.b.two", "app.b.three", "app.b.four"]
    for qn in cluster_a + cluster_b:
        batch.functions.append(_fn(qn))
    # Dense intra-cluster edges
    for c in (cluster_a, cluster_b):
        for i, src in enumerate(c):
            for tgt in c[i + 1:]:
                batch.calls.append(_call(src, tgt))
    # One bridge edge
    batch.calls.append(_call(cluster_a[0], cluster_b[0]))

    CommunityDetectPhase().run(_ctx(batch))

    assert len(batch.communities) == 2
    sizes = sorted(c.size for c in batch.communities)
    assert sizes == [4, 4]


# ---------------------------------------------------------------------------
# 3. Label heuristic picks the most-common token.
# ---------------------------------------------------------------------------

def test_label_uses_common_token():
    """Pure-function test of `_label_for`. Three function qns all sharing
    the token `auth` should produce label `auth`, regardless of how
    Louvain would have clustered them — _label_for is independent of
    the partition algorithm.

    (Originally this test ran the full phase and relied on Louvain
    clustering 3 nodes in a triangle into one community. After Sprint
    13a tuning bumped the resolution to 2.0, modularity prefers
    splitting tiny synthetic graphs into singletons, so the assertion
    no longer survives an end-to-end run. Testing the helper directly
    is more honest anyway.)
    """
    members = ["app.a.auth_login", "app.b.auth_logout", "app.c.auth_check"]
    label = _label_for(sorted(members), used_labels=set(), cluster_index=0)
    assert label == "auth"


# ---------------------------------------------------------------------------
# 4. Disambiguation — two clusters that would tie on top label.
# ---------------------------------------------------------------------------

def test_label_disambiguation():
    """Two clusters whose top-frequency token is identical. Second one
    must get a different label per the disambiguation rule.

    The token heuristic looks at the LAST TWO dotted segments — so we
    construct qns where the second-to-last segment is the same in every
    member of a cluster (`handler`), and the last segment varies. This
    guarantees `handler` is the top token in both clusters."""
    batch = IndexBatch(repo=REPO)
    # Both clusters share the same second-to-last segment "handler".
    # Last segments differ within and across clusters.
    cluster_1 = [
        "app.handler.alpha",
        "app.handler.beta",
        "app.handler.gamma",
    ]
    cluster_2 = [
        "app.handler.delta",
        "app.handler.epsilon",
        "app.handler.zeta",
    ]
    for qn in cluster_1 + cluster_2:
        batch.functions.append(_fn(qn))
    # Triangle inside each cluster, no cross edges.
    for cluster in (cluster_1, cluster_2):
        batch.calls.append(_call(cluster[0], cluster[1]))
        batch.calls.append(_call(cluster[1], cluster[2]))
        batch.calls.append(_call(cluster[2], cluster[0]))

    CommunityDetectPhase().run(_ctx(batch))

    labels = {c.label for c in batch.communities}
    assert len(labels) == 2, f"labels collided: {labels}"
    # Both clusters had `handler` as the top token; the first emitted
    # keeps it, the second gets a disambiguated form.
    assert "handler" in labels
    other = next(lbl for lbl in labels if lbl != "handler")
    assert other.startswith("handler_") and len(other) > len("handler_")


# ---------------------------------------------------------------------------
# 5. Cohesion clamping — singletons get 0.0, fully-connected get 1.0.
# ---------------------------------------------------------------------------

def test_cohesion_clamped():
    """Pure-function test of `_cohesion`. Singleton (no edges touching the
    member) returns 0.0; fully-connected community (every edge intra)
    returns 1.0.

    (Originally this test ran the full phase. After 13a tuning bumped
    resolution to 2.0, Louvain splits the triangle fixture into three
    singletons — the cohesion math still works correctly, but the
    triangle community no longer exists in the partition. Testing the
    cohesion helper directly with a synthetic graph keeps the math
    coverage without coupling to Louvain's micro-graph behavior.)
    """
    import networkx as nx

    triangle = {"app.t.foo_alpha", "app.t.foo_beta", "app.t.foo_gamma"}
    singleton = {"app.iso.lonely_function"}

    g = nx.Graph()
    for n in triangle | singleton:
        g.add_node(n)
    # Triangle edges only — singleton has no edges.
    nodes = sorted(triangle)
    g.add_edge(nodes[0], nodes[1])
    g.add_edge(nodes[1], nodes[2])
    g.add_edge(nodes[2], nodes[0])

    assert _cohesion(g, triangle) == 1.0
    assert _cohesion(g, singleton) == 0.0


# ---------------------------------------------------------------------------
# 6. Every Function and Class lands in exactly one MemberOfEdge.
# ---------------------------------------------------------------------------

def test_member_of_edges_one_per_function():
    batch = IndexBatch(repo=REPO)
    fn_qns = [f"app.m.fn_{i}" for i in range(6)]
    cls_qns = [f"app.m.Cls_{i}" for i in range(3)]
    for qn in fn_qns:
        batch.functions.append(_fn(qn))
    for qn in cls_qns:
        batch.classes.append(_cls(qn))
    # A few calls so Louvain has structure to chew on.
    batch.calls.append(_call(fn_qns[0], fn_qns[1]))
    batch.calls.append(_call(fn_qns[2], fn_qns[3]))
    batch.calls.append(_call(fn_qns[4], fn_qns[5]))

    CommunityDetectPhase().run(_ctx(batch))

    members_seen = [edge.member_qn for edge in batch.member_of]
    assert sorted(members_seen) == sorted(fn_qns + cls_qns)
    # No duplicates.
    assert len(members_seen) == len(set(members_seen))


# ---------------------------------------------------------------------------
# 7. Determinism: same input twice produces identical labels + memberships.
# ---------------------------------------------------------------------------

def test_deterministic_across_runs():
    def _build() -> IndexBatch:
        b = IndexBatch(repo=REPO)
        cluster_a = ["app.a.alpha", "app.a.beta", "app.a.gamma", "app.a.delta"]
        cluster_b = ["app.b.one", "app.b.two", "app.b.three", "app.b.four"]
        cluster_c = ["app.c.uno", "app.c.dos", "app.c.tres", "app.c.quatro"]
        for qn in cluster_a + cluster_b + cluster_c:
            b.functions.append(_fn(qn))
        for cluster in (cluster_a, cluster_b, cluster_c):
            for i, src in enumerate(cluster):
                for tgt in cluster[i + 1:]:
                    b.calls.append(_call(src, tgt))
        # Sparse cross-cluster bridges
        b.calls.append(_call(cluster_a[0], cluster_b[0]))
        b.calls.append(_call(cluster_b[0], cluster_c[0]))
        return b

    batch_1 = _build()
    batch_2 = _build()
    CommunityDetectPhase().run(_ctx(batch_1))
    CommunityDetectPhase().run(_ctx(batch_2))

    labels_1 = sorted(c.label for c in batch_1.communities)
    labels_2 = sorted(c.label for c in batch_2.communities)
    assert labels_1 == labels_2

    member_map_1 = {edge.member_qn: edge.community_label for edge in batch_1.member_of}
    member_map_2 = {edge.member_qn: edge.community_label for edge in batch_2.member_of}
    assert member_map_1 == member_map_2


# ---------------------------------------------------------------------------
# 8. Symbol-targeted CALLS don't pull external Symbols into communities.
# ---------------------------------------------------------------------------

def test_skip_symbol_targets():
    """A CALLS edge to a Symbol qn (not in batch.functions/classes) must
    NOT create a graph node — Symbol qns aren't members we own, and
    pulling them in would inflate cluster sizes + create spurious bridges."""
    batch = IndexBatch(repo=REPO)
    fn_a = "app.svc.handler_one"
    fn_b = "app.svc.handler_two"
    batch.functions.append(_fn(fn_a))
    batch.functions.append(_fn(fn_b))
    batch.calls.append(_call(fn_a, fn_b))
    # External symbol — should NOT become a graph node.
    batch.calls.append(_call(fn_a, "external.lib.do_thing"))
    batch.calls.append(_call(fn_b, "external.lib.do_thing"))

    CommunityDetectPhase().run(_ctx(batch))

    member_qns = {edge.member_qn for edge in batch.member_of}
    assert "external.lib.do_thing" not in member_qns
    assert member_qns == {fn_a, fn_b}
    # Total community size must equal owned-member count, not symbol count.
    assert sum(c.size for c in batch.communities) == 2


# ---------------------------------------------------------------------------
# 9. Pure-function unit tests for the label heuristic — fast feedback loop.
# ---------------------------------------------------------------------------

def test_tokens_for_qn_basic():
    # Last two dotted segments only ("repo_indexer", "foo_bar"), each split
    # on `_`/`-`, alpha-only + lowercase.
    assert _tokens_for_qn("app.repo_indexer.foo_bar") == ["repo", "indexer", "foo", "bar"]
    # Digits get stripped from each piece (alpha-only); empty pieces drop.
    assert _tokens_for_qn("pkg.x123_y") == ["pkg", "x", "y"]
    # Single-segment qn: don't slice past the start.
    assert _tokens_for_qn("alone") == ["alone"]


def test_label_for_fallback_on_empty():
    # qn with literally no alpha characters at all → fallback to cluster_<index>.
    assert _label_for(["123.456"], used_labels=set(), cluster_index=7) == "cluster_7"
    # qn with at least one alpha token → that token is the label, not fallback.
    assert _label_for(["x.123"], used_labels=set(), cluster_index=7) == "x"
