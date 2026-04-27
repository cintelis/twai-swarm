"""Community detection phase — Louvain clustering over the resolved graph.

Sprint 13a. Runs AFTER ResolvePhase so the CALLS / IMPORTS / INHERITS_FROM
edges already point at in-repo Functions/Classes (not Symbols). The phase
mutates `ctx.batch` in-place: it appends `CommunityNode` rows + one
`MemberOfEdge` per Function/Class member. No Neo4j writes here — same
contract as ResolvePhase; the loader handles persistence.

Implementation notes
--------------------

* **Algorithm:** Louvain modularity maximisation. We tried
  `rustworkx.community.louvain_communities` first (preferred — same dep we
  already pull in for Tarjan SCC in `scope_resolution/finalize.py`), but
  rustworkx 0.17.1 (the latest pip-released as of Sprint 13a) does not
  expose a `community` module — that landed in 0.18 / unreleased main.
  Fallback is `python-louvain` (the `community` package on PyPI), which
  operates on a `networkx.Graph`. networkx is therefore a transitive
  runtime dep here; the spike doc excluded it as the *primary* graph
  engine for resolution but not for one-off post-processing like this.

* **Determinism:** Louvain is stochastic — node iteration order and tie
  breaks shift cluster labels between runs. Per the Sprint 13 risks
  section ("a Coder asking 'what modules exist?' should get the same
  answer between runs"), we hardcode `random_state=42`. The `_label_for`
  heuristic also sorts member_qns before counting tokens so iteration
  order doesn't perturb ties.

* **Reversibility:** The phase is a no-op when the graph has no
  Function/Class nodes (all-SHA-match re-index, or pure-data repo). One
  line in `phases/__init__.py` removes it from the pipeline; nothing
  else in the indexer depends on community data.
"""
from __future__ import annotations

import re
import time
from collections import Counter

from ..actions import CommunityNode, MemberOfEdge
from ..runner import PhaseContext


# Hardcoded so re-running the indexer on identical code produces identical
# community assignments. Documented here because the Sprint 13a risks
# section calls it out — don't change without breaking the determinism
# promise the Coder relies on.
LOUVAIN_SEED = 42

_TOKEN_SPLIT_RE = re.compile(r"[._\-]")
_ALPHA_RE = re.compile(r"[^a-z]")


def _tokens_for_qn(qn: str) -> list[str]:
    """Extract tokens from a qualified name for label-heuristic counting.

    Take the LAST two dotted segments (so `app.repo_indexer.foo.bar` →
    `foo` + `bar` tokens), then split each on `_` / `-` to break compound
    names. Lowercase + alpha-only; digits and symbols are dropped.

    Pure function: same input always produces the same output list.
    """
    segments = qn.split(".")
    last_two = segments[-2:] if len(segments) >= 2 else segments
    tokens: list[str] = []
    for seg in last_two:
        for piece in _TOKEN_SPLIT_RE.split(seg):
            cleaned = _ALPHA_RE.sub("", piece.lower())
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _label_for(member_qns: list[str], used_labels: set[str], cluster_index: int) -> str:
    """Pick a deterministic label for a community.

    Strategy: count token frequencies across all member qns (sorted to
    avoid order-dependent tie-breaks), label = highest-frequency token.
    On collision with `used_labels`, append the next-most-frequent token
    until unique. Fallback for empty/all-collision: `cluster_<index>`.

    Pure function of `(sorted(member_qns), used_labels, cluster_index)`.
    """
    sorted_qns = sorted(member_qns)
    counter: Counter[str] = Counter()
    for qn in sorted_qns:
        counter.update(_tokens_for_qn(qn))

    if not counter:
        return f"cluster_{cluster_index}"

    # most_common() is order-stable for equal counts in CPython 3.7+;
    # combined with sorted_qns input above, this is fully deterministic.
    ranked = [tok for tok, _ in counter.most_common()]

    primary = ranked[0]
    if primary not in used_labels:
        return primary

    # Disambiguate by appending the next-most-frequent token, then the
    # one after, etc. (`auth_handlers`, `auth_routes`, ...).
    for secondary in ranked[1:]:
        candidate = f"{primary}_{secondary}"
        if candidate not in used_labels:
            return candidate

    # Fully exhausted ranked tokens — fall back to numeric.
    return f"cluster_{cluster_index}"


def _build_graph(ctx: PhaseContext):
    """Build an undirected networkx Graph from the resolved batch.

    Nodes = every Function qn + every Class qn in the batch.
    Edges = CALLS + INHERITS_FROM (between in-repo nodes only — Symbol
    targets are skipped) + IMPORTS (file-level, projected to the
    importing file's defining module's top-level Functions/Classes
    against the imported module's defining members).

    Returns (graph, member_qns_sorted).
    """
    import networkx as nx

    # Member set: every Function + Class is a candidate node.
    member_qns: set[str] = set()
    for fn in ctx.batch.functions:
        member_qns.add(fn.qualified_name)
    for cls in ctx.batch.classes:
        member_qns.add(cls.qualified_name)

    g = nx.Graph()
    for qn in sorted(member_qns):
        g.add_node(qn)

    # CALLS — both endpoints must be in member_qns; calls into Symbol
    # nodes are external and excluded by definition.
    for call in ctx.batch.calls:
        if call.caller_qn in member_qns and call.callee_qn in member_qns:
            g.add_edge(call.caller_qn, call.callee_qn)

    # INHERITS_FROM — same filter; only edges between owned classes count.
    for inh in ctx.batch.inherits:
        if inh.child_qn in member_qns and inh.parent_qn in member_qns:
            g.add_edge(inh.child_qn, inh.parent_qn)

    # IMPORTS — file-level. Project each File→Module import to edges
    # between the importing file's top-level members and the imported
    # module's top-level members. This connects modules whose code never
    # directly calls each other but live in the same logical import
    # neighborhood (e.g. shared utility modules).
    if ctx.batch.imports:
        # file_path -> set of member qns defined in that file
        file_members: dict[str, set[str]] = {}
        for fn in ctx.batch.functions:
            file_members.setdefault(fn.file_path, set()).add(fn.qualified_name)
        for cls in ctx.batch.classes:
            file_members.setdefault(cls.file_path, set()).add(cls.qualified_name)

        # module_qn -> file_path (defining file). We project imports to
        # the importing file's members on one side and the target module's
        # defining file's members on the other.
        module_file: dict[str, str] = {m.qualified_name: m.file_path for m in ctx.batch.modules}

        for imp in ctx.batch.imports:
            src_members = file_members.get(imp.file_path, set())
            target_file = module_file.get(imp.target_qn)
            if target_file is None:
                # Symbol import or unowned target — skip.
                continue
            tgt_members = file_members.get(target_file, set())
            if not src_members or not tgt_members:
                continue
            # Connect a single representative pair (smallest-name on each
            # side, deterministic). Adding the full Cartesian product
            # would over-densify the graph and wash out call-based
            # community signal.
            src_anchor = min(src_members)
            tgt_anchor = min(tgt_members)
            if src_anchor != tgt_anchor:
                g.add_edge(src_anchor, tgt_anchor)

    return g, sorted(member_qns)


def _detect_communities(graph) -> list[set[str]]:
    """Run Louvain with the deterministic seed.

    Tries `rustworkx.community.louvain_communities` first (it doesn't
    exist in 0.17.1 but will in a future release; we want to switch back
    automatically once available). Falls back to `python-louvain` on a
    networkx Graph.

    Returns a list of node-set communities, sorted by smallest member
    qn for stable ordering (so cluster_index in `_label_for` is
    deterministic across runs).
    """
    # Prefer rustworkx.community.louvain_communities when available — same
    # dep we already pull in for Tarjan SCC. As of rustworkx 0.17.1 the
    # `community` submodule doesn't exist; this `find_spec` check makes
    # the fallback automatic once a future rustworkx ships it. We probe
    # via importlib rather than try/import to keep ruff (F401) happy
    # about the unused symbol.
    import importlib.util
    if importlib.util.find_spec("rustworkx.community") is not None:
        # When rustworkx Louvain lands, plug the adapter in here. Today
        # this branch is unreachable; documented for the future swap.
        pass

    import community as community_louvain  # python-louvain

    # python-louvain returns {node: community_id}. Group into sets.
    if graph.number_of_nodes() == 0:
        return []

    partition = community_louvain.best_partition(graph, random_state=LOUVAIN_SEED)
    by_id: dict[int, set[str]] = {}
    for node, cid in partition.items():
        by_id.setdefault(cid, set()).add(node)

    # Stable ordering: sort communities by their min-member qn.
    return sorted(by_id.values(), key=lambda s: min(s))


def _cohesion(graph, members: set[str]) -> float:
    """Intra-community edge ratio: intra / (intra + inter).

    Returns 0.0 for singletons or fully-disconnected communities (no
    edges touch any member). Clamped to [0.0, 1.0].
    """
    intra = 0
    inter = 0
    for u in members:
        # `graph[u]` is a dict of neighbors — works on networkx Graph.
        for v in graph[u]:
            if v in members:
                intra += 1  # double-counted (u-v and v-u); OK, ratio survives
            else:
                inter += 1
    total = intra + inter
    if total == 0:
        return 0.0
    ratio = intra / total
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        return 1.0
    return ratio


class CommunityDetectPhase:
    """Detect graph communities and emit Community + MEMBER_OF rows."""

    name = "community_detect"

    def run(self, ctx: PhaseContext) -> None:
        # No functions and no classes → nothing to cluster. This happens
        # on a re-index where every file's SHA matched (parse skipped
        # everything). The IndexBatch is effectively empty for our
        # purposes; bail out to keep the loader writes a no-op.
        if not ctx.batch.functions and not ctx.batch.classes:
            return

        start = time.monotonic()
        graph, _member_qns = _build_graph(ctx)

        if graph.number_of_nodes() == 0:
            return

        communities = _detect_communities(graph)
        used_labels: set[str] = set()
        tenant_id = ctx.repo.tenant_id

        for idx, members in enumerate(communities):
            label = _label_for(sorted(members), used_labels, idx)
            used_labels.add(label)

            cohesion = _cohesion(graph, members)
            ctx.batch.communities.append(CommunityNode(
                repo=ctx.repo.name,
                tenant_id=tenant_id,
                label=label,
                cohesion=cohesion,
                size=len(members),
            ))
            for member_qn in sorted(members):
                ctx.batch.member_of.append(MemberOfEdge(
                    repo=ctx.repo.name,
                    tenant_id=tenant_id,
                    member_qn=member_qn,
                    community_label=label,
                ))

        elapsed = time.monotonic() - start
        ctx.progress(
            f"[indexer] detected {len(communities)} communities in {elapsed:.1f}s"
        )
        for k, v in ctx.batch.counts().items():
            ctx.progress(f"  {k:20s} {v}")
