"""Process extraction phase — find call chains that cross community boundaries.

Sprint 13b. Runs AFTER CommunityDetectPhase (13a). The output is a set of
`Process` nodes labelling "execution flows" through the codebase — chains
of CALLS edges that traverse 2+ distinct communities. This is the
abstraction GitNexus surfaces via its MCP `query` tool: instead of raw
symbols, the agent gets ranked flows ("RepoTaskWorkflow.run -> resolve_batch")
each with its ordered step list.

Implementation notes
--------------------

* **Algorithm:** forward DFS from every entry point (functions called by
  no other function). Each chain is scored as `length * cross_community_count`
  where cross_community_count is `len(distinct community labels in chain) - 1`.
  Chains entirely within one community score 0 and are dropped — they're
  internal helpers, not flows.

* **Why DFS not BFS:** we want full chains (the agent asks "what does this
  flow do end-to-end?"), not shortest paths. DFS up to MAX_DEPTH=8 caps the
  combinatorial explosion while still surfacing the typical
  workflow-run -> activity -> sub-call shape.

* **Determinism:** every list operation that feeds into output is sorted.
  Entry points sorted alphabetically; DFS recurses into successors in
  alphabetical order; candidate chains sorted by (-score, name, full_path).
  Same input batch ⇒ identical processes + step orderings, both for caching
  and so the Coder gets reproducible answers across re-indexes.

* **Cycle protection:** per-DFS visited set. A function can't appear twice
  in the same chain. (Tarjan-style SCC handling isn't needed here — we're
  enumerating simple paths, not contracting components.)

* **Reversibility:** no-op when `ctx.batch.member_of` is empty (i.e. 13a
  didn't run, or had nothing to do). Same one-line removal pattern as 13a.
"""
from __future__ import annotations

import time

from ..actions import ProcessNode, StepInProcessEdge
from ..runner import PhaseContext


# Cap chain length to keep DFS tractable. 8 is empirically deep enough to
# capture workflow.run -> activity -> sub-step patterns without exploding
# on densely-connected helper layers.
MAX_DEPTH = 8

# Cap output count — top 50 by (length * cross_community_count) score, per
# the Sprint 13 plan. Keeps the Coder's process list scannable.
MAX_PROCESSES = 50

# Truncate process summaries to this many step short-names. Beyond ~6 the
# summary becomes unreadable in a tool result line.
SUMMARY_MAX_STEPS = 6
SUMMARY_TRUNCATE = "..."

# Sentinel community label for functions not in any community. Functions
# without a real label must NOT count toward cross-boundary scoring (they'd
# spuriously inflate the score of chains touching orphan helpers).
_ORPHAN_LABEL = "<orphan>"


def _short_name(qn: str) -> str:
    """Last dotted segment — what shows up in process names + summaries."""
    return qn.rsplit(".", 1)[-1]


def _build_call_graph(ctx: PhaseContext):
    """Build a directed rustworkx PyDiGraph of resolved Function -> Function calls.

    Nodes are Function qns only. Class qns aren't included — STEP_IN_PROCESS
    points at Functions per the schema, and CALLS edges between classes
    don't exist anyway. Calls into Symbol nodes (externals) are skipped:
    flows end at the boundary of code we own.

    Returns (graph, qn_to_idx, idx_to_qn). Indices are needed because
    rustworkx works on integer indices, not arbitrary keys.
    """
    import rustworkx as rx

    function_qns: set[str] = {fn.qualified_name for fn in ctx.batch.functions}

    graph = rx.PyDiGraph()
    qn_to_idx: dict[str, int] = {}
    # Insert nodes in sorted order so node indices are deterministic — not
    # strictly required (we sort qns at every traversal point) but makes
    # debugging dumps reproducible.
    for qn in sorted(function_qns):
        qn_to_idx[qn] = graph.add_node(qn)
    idx_to_qn = {idx: qn for qn, idx in qn_to_idx.items()}

    # CALLS — both endpoints must be in function_qns. Symbol callees are
    # external by definition and don't form flows. Deduplicate parallel
    # edges (same caller + callee from multiple call sites) — rustworkx
    # would store them as parallel edges and inflate successor enumeration.
    seen_edges: set[tuple[int, int]] = set()
    for call in ctx.batch.calls:
        if call.caller_qn in qn_to_idx and call.callee_qn in qn_to_idx:
            src = qn_to_idx[call.caller_qn]
            tgt = qn_to_idx[call.callee_qn]
            if src == tgt:
                continue  # self-call; would force cycle handling for no benefit
            edge = (src, tgt)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            graph.add_edge(src, tgt, None)

    return graph, qn_to_idx, idx_to_qn


def _entry_points(graph, idx_to_qn: dict[int, str]) -> list[int]:
    """Function indices to start DFS from.

    Primary: functions with `out_degree > 0` (they call something) and
    `in_degree == 0` (no one in the resolved graph calls them) — these are
    the entries to flows: workflow runs, CLI handlers, FastAPI endpoints.

    Tie-break: if no such functions exist (every function has someone
    calling it — usually means the whole graph is in a cycle, rare but
    happens on metaprogramming-heavy repos), fall back to functions with
    the smallest in-degree among those with out-degree > 0.

    Returns indices sorted by qn ASC for deterministic iteration.
    """
    entries: list[int] = []
    for idx in graph.node_indices():
        if graph.out_degree(idx) == 0:
            continue  # leaves can't start a chain
        if graph.in_degree(idx) == 0:
            entries.append(idx)

    if not entries:
        # Fallback: smallest in-degree among nodes with out-degree > 0.
        # All-cycle case. Pick the minimum non-zero in-degree.
        candidates: list[tuple[int, int]] = []
        for idx in graph.node_indices():
            if graph.out_degree(idx) == 0:
                continue
            candidates.append((graph.in_degree(idx), idx))
        if not candidates:
            return []
        min_in = min(d for d, _ in candidates)
        entries = [idx for d, idx in candidates if d == min_in]

    return sorted(entries, key=lambda i: idx_to_qn[i])


def _community_lookup(ctx: PhaseContext) -> dict[str, str]:
    """member_qn -> community_label, with `_ORPHAN_LABEL` for unlabelled.

    Functions not in any MemberOfEdge get the orphan sentinel. They never
    contribute to cross-boundary count: a chain Helper -> Orphan -> Helper
    where both helpers share a community is still single-community by
    distinct-label count == 1 (orphan + helper-community = 2 → -1 = 1
    cross), but we strip the orphan label before counting to avoid that.
    See `_score_chain` for the strip.
    """
    out: dict[str, str] = {}
    for edge in ctx.batch.member_of:
        out[edge.member_qn] = edge.community_label
    return out


def _enumerate_chains(
    graph,
    idx_to_qn: dict[int, str],
    entries: list[int],
) -> list[tuple[str, ...]]:
    """All forward DFS chains from each entry point, capped at MAX_DEPTH.

    For each entry, we enumerate every simple path of length 2..MAX_DEPTH.
    Length-1 chains (single function, no calls) are skipped — they can't
    cross community boundaries by definition.

    Returns a list of qn tuples — caller-to-callee order, no duplicates.
    """
    chains: set[tuple[str, ...]] = set()

    def dfs(node: int, path: list[int], visited: set[int]) -> None:
        if len(path) >= MAX_DEPTH:
            # Emit the chain at max depth (it's the longest we'll capture)
            # and stop recursing — going deeper would just truncate the
            # same way next time.
            chains.add(tuple(idx_to_qn[i] for i in path))
            return
        # Successors sorted by qn for deterministic chain enumeration.
        succ_indices = sorted(graph.successor_indices(node), key=lambda i: idx_to_qn[i])
        if not succ_indices:
            # Dead-end — emit if we've accumulated a real chain.
            if len(path) >= 2:
                chains.add(tuple(idx_to_qn[i] for i in path))
            return

        # Emit intermediate chains too — a 5-step path's prefixes might
        # score higher (different cross-community counts). But cap at
        # length >= 2 (single-step "chains" can't cross a boundary).
        if len(path) >= 2:
            chains.add(tuple(idx_to_qn[i] for i in path))

        for nxt in succ_indices:
            if nxt in visited:
                continue  # cycle protection — don't re-enter in this DFS
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt, path, visited)
            path.pop()
            visited.remove(nxt)

    for entry in entries:
        dfs(entry, [entry], {entry})

    return sorted(chains)


def _score_chain(
    chain: tuple[str, ...],
    member_of: dict[str, str],
) -> tuple[int, int, int]:
    """Score for a chain: (score, length, cross_community_count).

    cross_community_count = (distinct community labels in chain) - 1, with
    the `_ORPHAN_LABEL` sentinel stripped first so unlabelled functions
    don't spuriously bump the count.

    score = length * cross_community_count. A chain entirely within one
    community (or all-orphan) scores 0 and is dropped by the caller.
    """
    labels = {member_of.get(qn, _ORPHAN_LABEL) for qn in chain}
    labels.discard(_ORPHAN_LABEL)
    distinct = len(labels)
    cross = max(0, distinct - 1)
    return (len(chain) * cross, len(chain), cross)


def _format_summary(chain: tuple[str, ...]) -> str:
    """Comma-separated short-names of the first SUMMARY_MAX_STEPS members.

    Truncated with `...` when the chain has more steps than the cap.
    """
    short_names = [_short_name(qn) for qn in chain]
    if len(short_names) <= SUMMARY_MAX_STEPS:
        return ", ".join(short_names)
    return ", ".join(short_names[:SUMMARY_MAX_STEPS]) + SUMMARY_TRUNCATE


def _format_name(chain: tuple[str, ...]) -> str:
    """`<first.short_name> -> <last.short_name>`. Disambiguation suffix
    is added by the caller after collision detection."""
    return f"{_short_name(chain[0])} -> {_short_name(chain[-1])}"


class ProcessExtractPhase:
    """Find cross-community CALL chains and emit Process + STEP_IN_PROCESS rows."""

    name = "process_extract"

    def run(self, ctx: PhaseContext) -> None:
        # No functions OR no community assignments → nothing to extract.
        # The community phase populates `member_of`; if it didn't run (e.g.
        # the indexer was invoked without 13a, or the resolved graph had
        # zero functions), processes are meaningless.
        if not ctx.batch.functions or not ctx.batch.member_of:
            return

        start = time.monotonic()
        graph, qn_to_idx, idx_to_qn = _build_call_graph(ctx)
        if graph.num_nodes() == 0:
            return

        entries = _entry_points(graph, idx_to_qn)
        if not entries:
            return

        member_of = _community_lookup(ctx)
        chains = _enumerate_chains(graph, idx_to_qn, entries)

        # Score every chain. Keep only those that cross a boundary
        # (cross_community_count >= 1, i.e. score > 0).
        scored: list[tuple[tuple[int, int, int], tuple[str, ...]]] = []
        for chain in chains:
            metrics = _score_chain(chain, member_of)
            if metrics[0] > 0:
                scored.append((metrics, chain))

        if not scored:
            ctx.progress(f"[indexer] extracted 0 processes in {time.monotonic() - start:.2f}s")
            return

        # Sort by (score DESC, chain ASC) for determinism. We sort the chain
        # tuple itself (not just name) so two chains with identical first/last
        # short names but different middles order stably.
        scored.sort(key=lambda item: (-item[0][0], item[1]))

        # Take top N. Disambiguation: if two chains would produce identical
        # names, append "#2" / "#3" / ... in score order. The highest-scoring
        # chain keeps the bare name.
        top = scored[:MAX_PROCESSES]
        used_names: dict[str, int] = {}
        tenant_id = ctx.repo.tenant_id
        emitted = 0
        for _metrics, chain in top:
            base_name = _format_name(chain)
            count = used_names.get(base_name, 0) + 1
            used_names[base_name] = count
            name = base_name if count == 1 else f"{base_name} #{count}"
            summary = _format_summary(chain)

            ctx.batch.processes.append(ProcessNode(
                repo=ctx.repo.name,
                tenant_id=tenant_id,
                name=name,
                summary=summary,
            ))
            for step_idx, member_qn in enumerate(chain):
                ctx.batch.step_in_process.append(StepInProcessEdge(
                    repo=ctx.repo.name,
                    tenant_id=tenant_id,
                    process_name=name,
                    member_qn=member_qn,
                    step=step_idx,
                ))
            emitted += 1

        elapsed = time.monotonic() - start
        ctx.progress(f"[indexer] extracted {emitted} processes in {elapsed:.2f}s")
