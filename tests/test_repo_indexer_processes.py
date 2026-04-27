"""Sprint 13b — ProcessExtractPhase tests against synthetic IndexBatches.

Mirrors test_repo_indexer_communities.py: hand-built IndexBatches, no
tree-sitter, no Neo4j. Skips if rustworkx (used for the call-graph) isn't
installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# rustworkx is the only graph dep we need; the phase doesn't use networkx
# or python-louvain. Skip the whole module if it's missing rather than
# breaking pytest collection.
pytest.importorskip("rustworkx")

from app.repo_indexer.actions import (  # noqa: E402
    CallEdge,
    FunctionNode,
    IndexBatch,
    MemberOfEdge,
    RepoNode,
)
from app.repo_indexer.phases.process_extract import (  # noqa: E402
    MAX_DEPTH,
    MAX_PROCESSES,
    ProcessExtractPhase,
)
from app.repo_indexer.runner import PhaseContext  # noqa: E402


REPO = RepoNode(name="r", url="", commit_sha="", tenant_id="t1")


def _ctx(batch: IndexBatch) -> PhaseContext:
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


def _call(caller: str, callee: str) -> CallEdge:
    return CallEdge(repo="r", caller_qn=caller, callee_qn=callee, line=1)


def _member(qn: str, label: str) -> MemberOfEdge:
    return MemberOfEdge(repo="r", tenant_id="t1", member_qn=qn, community_label=label)


def _populate_simple_chain(batch: IndexBatch, qns: list[str], labels: list[str]) -> None:
    """Helper: add functions + linear A->B->C calls + community labels."""
    for qn in qns:
        batch.functions.append(_fn(qn))
    for i in range(len(qns) - 1):
        batch.calls.append(_call(qns[i], qns[i + 1]))
    for qn, label in zip(qns, labels):
        batch.member_of.append(_member(qn, label))


# ---------------------------------------------------------------------------
# 1. Empty batch — phase does nothing.
# ---------------------------------------------------------------------------

def test_empty_batch_noop():
    batch = IndexBatch(repo=REPO)
    ProcessExtractPhase().run(_ctx(batch))
    assert batch.processes == []
    assert batch.step_in_process == []


# ---------------------------------------------------------------------------
# 2. Functions + calls but no communities (13a didn't run) — no-op.
# ---------------------------------------------------------------------------

def test_no_communities_noop():
    batch = IndexBatch(repo=REPO)
    batch.functions.append(_fn("a.foo"))
    batch.functions.append(_fn("a.bar"))
    batch.calls.append(_call("a.foo", "a.bar"))
    # Deliberately no MemberOfEdge entries.

    ProcessExtractPhase().run(_ctx(batch))
    assert batch.processes == []
    assert batch.step_in_process == []


# ---------------------------------------------------------------------------
# 3. 3 functions in 3 different communities, A -> B -> C → one process.
# ---------------------------------------------------------------------------

def test_simple_three_community_chain():
    batch = IndexBatch(repo=REPO)
    qns = ["app.entry.alpha", "app.mid.beta", "app.tail.gamma"]
    labels = ["entry", "mid", "tail"]
    _populate_simple_chain(batch, qns, labels)

    ProcessExtractPhase().run(_ctx(batch))

    # The DFS will emit both the length-2 prefix (alpha->beta) and the
    # length-3 full chain (alpha->beta->gamma). Both score positively.
    # The full chain scores higher (3 * 2 = 6 > 2 * 1 = 2), so it ranks
    # first. Both should appear; assert the top one is the full chain.
    assert len(batch.processes) >= 1
    top = batch.processes[0]
    assert top.name == "alpha -> gamma"

    # The full-chain process has 3 steps in order [alpha, beta, gamma].
    full_steps = [
        e for e in batch.step_in_process if e.process_name == "alpha -> gamma"
    ]
    full_steps.sort(key=lambda e: e.step)
    assert [e.step for e in full_steps] == [0, 1, 2]
    assert [e.member_qn for e in full_steps] == qns


# ---------------------------------------------------------------------------
# 4. Same chain but all in one community — dropped (cross_count = 0).
# ---------------------------------------------------------------------------

def test_within_community_chain_dropped():
    batch = IndexBatch(repo=REPO)
    qns = ["app.x.alpha", "app.x.beta", "app.x.gamma"]
    labels = ["solo", "solo", "solo"]
    _populate_simple_chain(batch, qns, labels)

    ProcessExtractPhase().run(_ctx(batch))

    assert batch.processes == []
    assert batch.step_in_process == []


# ---------------------------------------------------------------------------
# 5. Two separate cross-community chains, two entry points → two processes.
# ---------------------------------------------------------------------------

def test_multiple_entry_points():
    batch = IndexBatch(repo=REPO)
    chain_a = ["pkg.a.start_a", "pkg.b.middle_a", "pkg.c.end_a"]
    chain_b = ["pkg.d.start_b", "pkg.e.middle_b", "pkg.f.end_b"]
    for qn in chain_a + chain_b:
        batch.functions.append(_fn(qn))
    for i in range(2):
        batch.calls.append(_call(chain_a[i], chain_a[i + 1]))
        batch.calls.append(_call(chain_b[i], chain_b[i + 1]))
    for qn in chain_a:
        batch.member_of.append(_member(qn, qn.split(".")[1]))
    for qn in chain_b:
        batch.member_of.append(_member(qn, qn.split(".")[1]))

    ProcessExtractPhase().run(_ctx(batch))

    full_chain_processes = [p for p in batch.processes if "->" in p.name and " #" not in p.name]
    names = {p.name for p in full_chain_processes}
    assert "start_a -> end_a" in names
    assert "start_b -> end_b" in names


# ---------------------------------------------------------------------------
# 6. Cycle — DFS doesn't infinite-loop; nothing crosses a boundary internally.
# ---------------------------------------------------------------------------

def test_cycle_handled():
    batch = IndexBatch(repo=REPO)
    batch.functions.append(_fn("a.foo"))
    batch.functions.append(_fn("a.bar"))
    batch.calls.append(_call("a.foo", "a.bar"))
    batch.calls.append(_call("a.bar", "a.foo"))
    batch.member_of.append(_member("a.foo", "comm_a"))
    batch.member_of.append(_member("a.bar", "comm_b"))

    # Should complete without recursion error. Both functions are in
    # cycles so the entry-point fallback (smallest in-degree) kicks in;
    # in-degree is 1 for both, so both become entries. The chain
    # foo -> bar (and bar -> foo) crosses a community boundary so we get
    # a length-2 process — that's fine, the assertion is that DFS
    # terminates.
    ProcessExtractPhase().run(_ctx(batch))
    # No crash. Process list is allowed to be empty or contain truncated
    # 2-step chains; the contract is "doesn't infinite-loop".
    for p in batch.processes:
        # Every emitted process has at least 2 steps (chain enumeration
        # rejects length < 2 chains).
        steps = [e for e in batch.step_in_process if e.process_name == p.name]
        assert len(steps) >= 2


# ---------------------------------------------------------------------------
# 7. 12-function linear chain → truncated at MAX_DEPTH=8.
# ---------------------------------------------------------------------------

def test_max_depth_respected():
    batch = IndexBatch(repo=REPO)
    qns = [f"comm_{i}.fn_{i}" for i in range(12)]
    labels = [f"comm_{i}" for i in range(12)]
    _populate_simple_chain(batch, qns, labels)

    ProcessExtractPhase().run(_ctx(batch))

    # The longest chain captured must be exactly MAX_DEPTH steps.
    step_counts_per_process: dict[str, int] = {}
    for edge in batch.step_in_process:
        step_counts_per_process[edge.process_name] = (
            step_counts_per_process.get(edge.process_name, 0) + 1
        )
    longest = max(step_counts_per_process.values())
    assert longest == MAX_DEPTH


# ---------------------------------------------------------------------------
# 8. 60 separate cross-community chains → top-50 cap.
# ---------------------------------------------------------------------------

def test_max_processes_cap():
    batch = IndexBatch(repo=REPO)
    # Build 60 disjoint length-3 chains, each crossing 3 communities.
    for i in range(60):
        # Use zero-padded indices so name ordering matches numeric ordering
        # — makes the determinism assertion below cleaner.
        idx = f"{i:03d}"
        qns = [f"chain_{idx}.entry.fn_a", f"chain_{idx}.mid.fn_b", f"chain_{idx}.tail.fn_c"]
        labels = [f"entry_{idx}", f"mid_{idx}", f"tail_{idx}"]
        _populate_simple_chain(batch, qns, labels)

    ProcessExtractPhase().run(_ctx(batch))
    assert len(batch.processes) == MAX_PROCESSES


# ---------------------------------------------------------------------------
# 9. Step indices are 0-based and ordered caller-to-callee.
# ---------------------------------------------------------------------------

def test_step_indices_zero_based_and_ordered():
    batch = IndexBatch(repo=REPO)
    qns = ["c1.first", "c2.second", "c3.third", "c4.fourth"]
    labels = ["c1", "c2", "c3", "c4"]
    _populate_simple_chain(batch, qns, labels)

    ProcessExtractPhase().run(_ctx(batch))

    assert any(p.name == "first -> fourth" for p in batch.processes)
    full_steps = [
        e for e in batch.step_in_process if e.process_name == "first -> fourth"
    ]
    full_steps.sort(key=lambda e: e.step)
    assert [e.step for e in full_steps] == [0, 1, 2, 3]
    assert [e.member_qn for e in full_steps] == qns


# ---------------------------------------------------------------------------
# 10. Determinism — same input twice yields identical processes + edges.
# ---------------------------------------------------------------------------

def test_deterministic_across_runs():
    def _build() -> IndexBatch:
        b = IndexBatch(repo=REPO)
        # Branching tree with multiple cross-community chains.
        edges = [
            ("a.entry", "b.middle"),
            ("a.entry", "c.alt"),
            ("b.middle", "d.tail"),
            ("c.alt", "d.tail"),
            ("c.alt", "e.other"),
        ]
        qns = sorted({n for e in edges for n in e})
        for qn in qns:
            b.functions.append(_fn(qn))
            b.member_of.append(_member(qn, qn.split(".")[0]))
        for src, tgt in edges:
            b.calls.append(_call(src, tgt))
        return b

    b1, b2 = _build(), _build()
    ProcessExtractPhase().run(_ctx(b1))
    ProcessExtractPhase().run(_ctx(b2))

    names_1 = [p.name for p in b1.processes]
    names_2 = [p.name for p in b2.processes]
    assert names_1 == names_2

    edges_1 = [(e.process_name, e.step, e.member_qn) for e in b1.step_in_process]
    edges_2 = [(e.process_name, e.step, e.member_qn) for e in b2.step_in_process]
    assert edges_1 == edges_2


# ---------------------------------------------------------------------------
# 11. Name disambiguation — collision gets "#2" suffix.
# ---------------------------------------------------------------------------

def test_name_disambiguation():
    batch = IndexBatch(repo=REPO)
    # Two chains where first.short_name = "start" and last.short_name = "end".
    # We give them IDENTICAL chain lengths and cross_community_counts so
    # the score tie-break falls to chain-tuple sort (which is alphabetical
    # by qn). The chain emitted FIRST keeps the bare name; the SECOND gets
    # "#2".
    chain_1 = ["pkg_a.entry.start", "pkg_a.mid.middle", "pkg_a.tail.end"]
    chain_2 = ["pkg_b.entry.start", "pkg_b.mid.middle", "pkg_b.tail.end"]
    for qn in chain_1 + chain_2:
        batch.functions.append(_fn(qn))
    for i in range(2):
        batch.calls.append(_call(chain_1[i], chain_1[i + 1]))
        batch.calls.append(_call(chain_2[i], chain_2[i + 1]))
    for qn in chain_1 + chain_2:
        # Use the second-to-last segment as the community label so each
        # chain crosses 3 distinct communities.
        batch.member_of.append(_member(qn, qn.split(".")[1]))

    ProcessExtractPhase().run(_ctx(batch))

    # Find the two top-scoring (length-3) processes.
    full = [p for p in batch.processes if p.summary.count(",") == 2]
    assert len(full) >= 2
    names = {p.name for p in full}
    assert "start -> end" in names
    assert "start -> end #2" in names


# ---------------------------------------------------------------------------
# 12. CALLS to Symbol qns (not in functions) are skipped.
# ---------------------------------------------------------------------------

def test_skip_symbol_callees():
    batch = IndexBatch(repo=REPO)
    qns = ["c1.foo", "c2.bar"]
    labels = ["c1", "c2"]
    for qn in qns:
        batch.functions.append(_fn(qn))
    for qn, label in zip(qns, labels):
        batch.member_of.append(_member(qn, label))
    batch.calls.append(_call("c1.foo", "c2.bar"))
    # External symbol — not in batch.functions; should be ignored.
    batch.calls.append(_call("c1.foo", "external.lib.do_thing"))
    batch.calls.append(_call("c2.bar", "another.external.helper"))

    ProcessExtractPhase().run(_ctx(batch))

    # No process should reference the external symbols.
    member_qns = {e.member_qn for e in batch.step_in_process}
    assert "external.lib.do_thing" not in member_qns
    assert "another.external.helper" not in member_qns
    # The internal foo -> bar chain is the only valid one and should appear.
    assert any(p.name == "foo -> bar" for p in batch.processes)
