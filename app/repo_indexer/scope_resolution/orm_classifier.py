"""Sprint 15b.2 — ORM call-site classifier.

Runs at finalize time AFTER call resolution. Joins
`batch.orm_call_hints` against the rewritten `batch.calls` by
`(caller_qn, line, leaf)`, resolves the access target to a TableNode
via the `class_qn → TableNode` index, and emits TableAccessEdge
records (READS / WRITES) on `batch.table_accesses`.

The classifier needs:
    - `batch.tables`               (declared table list, for class_qn → table mapping)
    - `batch.orm_call_hints`       (extractor-side argument captures)
    - `batch.calls`                (ALREADY rewritten by step 9)
    - resolver closures            (`_resolve_type_name`, `_resolve_type_chain`,
                                    `local_var_index.find` + `scope_tree`,
                                    `func_scope_id`, `caller_class_qn`, `class_scope_id`,
                                    `dispatch_index.parent_of` — for Django MRO check)

Recognises three receiver kinds:
    - **sa_session**: receiver type is sqlalchemy.orm.Session (or a
      common alias). Table comes from the first positional arg's
      identifier (or, for `execute(select(X))`, the inner-call arg).
    - **django_manager**: receiver chain head is a class with a
      Django `models.Model` ancestor; the chain has `.objects.<verb>`
      shape OR is a chained QuerySet leaf. Table = the head class.
    - **django_instance**: receiver is a variable typed to a Django
      Model class; method is one of `save` / `delete` /
      `refresh_from_db`. Table = the variable's class.

Emits TableAccessEdge records keyed `(caller_qn, table_name, op, line)`.
"""
from __future__ import annotations

from ..actions import (
    IndexBatch,
    OrmCallHint,
    TableAccessEdge,
)
from ..domain_extractors.orm_python import (
    DJANGO_BOTH_LEAF_METHODS,
    DJANGO_INSTANCE_READ_METHODS,
    DJANGO_INSTANCE_WRITE_METHODS,
    DJANGO_READ_LEAF_METHODS,
    DJANGO_WRITE_LEAF_METHODS,
    SA_INNER_CONSTRUCTS,
    SA_SESSION_LEAF_METHODS,
)


# Class qns we treat as a SQLAlchemy Session. The literal-name set covers
# the canonical patterns (Session, scoped_session, AsyncSession). Caller
# may add aliases observed in real-world repos here without affecting the
# classifier shape.
SA_SESSION_TYPE_NAMES = frozenset({
    "Session",
    "AsyncSession",
    "scoped_session",
    "sessionmaker",
    # Common dotted re-exports that sometimes survive resolution
    "sqlalchemy.orm.Session",
    "sqlalchemy.orm.scoped_session",
    "sqlalchemy.orm.sessionmaker",
    "sqlalchemy.ext.asyncio.AsyncSession",
})

# Django Model ancestor qns. We check the bare leaf name AND the dotted
# `models.Model` form because tree-sitter inheritance is recorded
# textually. Ancestor walk via `parent_of` will also follow chains like
# `class User(AbstractUser): ...` → `AbstractUser` → `Model`.
DJANGO_MODEL_TERMINAL_BARE_NAMES = frozenset({"Model", "AbstractUser", "AbstractBaseUser"})


def classify_orm_calls(
    batch: IndexBatch,
    *,
    qn_index,
    parent_relation,
    local_var_index,
    scope_tree,
    func_scope_id,
    func_file_path,
    caller_class_qn,
    class_scope_id,
    resolve_type_name,
    resolve_type_chain,
) -> None:
    """Emit TableAccessEdge records onto `batch.table_accesses` for every
    ORM call hint the classifier can resolve.

    Mutates `batch.table_accesses` in place. Other batch fields untouched.

    Pre-conditions:
        - `batch.calls` has already been rewritten by `finalize_batch`
        - `batch.orm_call_hints` contains the extractor-side captures
        - `batch.tables` contains TableNode records for declared models

    `qn_index` must support `.lookup(qn) → Declaration | None`.
    `parent_relation` is `dict[child_qn → list[parent_qn]]` for Django
    ancestor walking.
    """
    if not batch.orm_call_hints or not batch.tables:
        return

    # Build indexes.
    # 1. class_qn → TableNode (for Django + SA declarative + SA typed).
    table_by_class_qn: dict[str, str] = {}
    table_by_bare_class_name: dict[str, str] = {}
    for t in batch.tables:
        if t.model_qn:
            table_by_class_qn[t.model_qn] = t.name
            bare = t.model_qn.rsplit(".", 1)[-1]
            # Don't clobber if the bare name is ambiguous; prefer first.
            table_by_bare_class_name.setdefault(bare, t.name)

    # Fast index of "this class qn was declared as a Django model" — we
    # tag it from the dialect on the TableNode rather than walking the
    # InheritsEdge graph, since `models.Model` is external and gets
    # filtered out of `parent_relation` (only resolved in-repo parents
    # survive there). Bare-name fallback covers the common case where
    # the resolver hasn't seen the model definition cross-file.
    django_model_qns: set[str] = set()
    django_model_bare_names: set[str] = set()
    for t in batch.tables:
        if t.dialect == "django" and t.model_qn:
            django_model_qns.add(t.model_qn)
            django_model_bare_names.add(t.model_qn.rsplit(".", 1)[-1])

    # 2. Fast lookup for hints by (caller_qn, line, leaf).
    hints_by_key: dict[tuple[str, int, str], OrmCallHint] = {}
    for hint in batch.orm_call_hints:
        hints_by_key[(hint.caller_qn, hint.line, hint.leaf)] = hint

    # 3. Build inherits map (child_qn → list[parent_qn]) including
    # external parents (parent_relation filters those out, but Django's
    # `models.Model` IS external, so we need the raw textual edges).
    raw_parent_relation: dict[str, list[str]] = {}
    for edge in batch.inherits:
        raw_parent_relation.setdefault(edge.child_qn, []).append(edge.parent_qn)

    # 4. Cache for `is_django_model_class(class_qn)`.
    django_model_cache: dict[str, bool] = {}

    def _is_django_model_class(class_qn: str) -> bool:
        """True iff `class_qn` was declared as a Django model.

        Fast-path: TableNode with `dialect="django"` already records this
        — when the model class is declared in the repo, we have a
        TableNode for it. For inherited cases (`class Admin(User)` where
        `User` is the django model), walk `raw_parent_relation` to find a
        Django ancestor by bare name (`Model`, `AbstractUser`, etc.).
        Memoised.
        """
        cached = django_model_cache.get(class_qn)
        if cached is not None:
            return cached
        if class_qn in django_model_qns:
            django_model_cache[class_qn] = True
            return True
        bare = class_qn.rsplit(".", 1)[-1]
        if bare in django_model_bare_names:
            django_model_cache[class_qn] = True
            return True
        # Walk inherits transitively (raw, so external `models.Model`
        # parents are visible).
        seen: set[str] = set()
        stack: list[str] = [class_qn]
        result = False
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            cur_bare = cur.rsplit(".", 1)[-1]
            if cur_bare in DJANGO_MODEL_TERMINAL_BARE_NAMES:
                result = True
                break
            if cur in django_model_qns or cur_bare in django_model_bare_names:
                result = True
                break
            parents = raw_parent_relation.get(cur, [])
            for p in parents:
                stack.append(p)
        django_model_cache[class_qn] = result
        return result

    def _is_sa_session_class(class_qn: str) -> bool:
        bare = class_qn.rsplit(".", 1)[-1]
        return class_qn in SA_SESSION_TYPE_NAMES or bare in SA_SESSION_TYPE_NAMES

    def _resolve_var_to_class(caller_qn: str, var_name: str) -> str | None:
        """Resolve a local-variable name in a function to its class qn.

        Walks the LocalVarTypeIndex (same machinery `_resolve_var_binding`
        in finalize.py uses).
        """
        caller_scope = func_scope_id.get(caller_qn)
        if caller_scope is None:
            return None
        ref = local_var_index.find(caller_scope, var_name, scope_tree)
        if ref is None:
            return None
        return resolve_type_chain(ref.raw_name, ref.declared_at_scope)

    def _resolve_chain_head_to_class(caller_qn: str, chain_head: str) -> str | None:
        """Resolve `User` in `User.objects.filter(...)`. Same as
        `_resolve_type_name` against the caller's file_path.
        """
        file_path = func_file_path.get(caller_qn)
        if file_path is None:
            return None
        return resolve_type_name(file_path, chain_head)

    # Pre-compute callee_qn lookup. After finalize, edges that resolved
    # have `callee_qn = Class.method`; unresolved retain raw text.
    # Index by (caller_qn, line) → list of edges, since multiple calls
    # can share a line (chained method calls on the same statement).
    edges_by_key: dict[tuple[str, int], list] = {}
    for edge in batch.calls:
        edges_by_key.setdefault((edge.caller_qn, edge.line), []).append(edge)

    seen_emits: set[tuple[str, str, str, int]] = set()

    def _emit(caller_qn: str, table_name: str, op: str, line: int) -> None:
        key = (caller_qn, table_name, op, line)
        if key in seen_emits:
            return
        seen_emits.add(key)
        batch.table_accesses.append(TableAccessEdge(
            repo=batch.repo.name,
            tenant_id=batch.repo.tenant_id,
            function_qn=caller_qn,
            table_name=table_name,
            op_kind=op,
            line=line,
        ))

    for hint in batch.orm_call_hints:
        leaf = hint.leaf
        caller_qn = hint.caller_qn
        line = hint.line

        # Fetch the matching CallEdge (post-resolution).
        edges = edges_by_key.get((caller_qn, line), [])
        # Find the edge whose callee leaf matches this hint's leaf.
        matched_edge = None
        for e in edges:
            edge_leaf = e.callee_qn.rsplit(".", 1)[-1]
            if edge_leaf == leaf:
                matched_edge = e
                break
        if matched_edge is None:
            continue
        callee_qn = matched_edge.callee_qn

        # Dispatch order: detect `.objects.` shape FIRST so Django leaves
        # (`delete`, `get`) that overlap with SA Session methods route
        # to the Django classifier. Then SA Session methods. Then Django
        # instance methods.
        is_django_manager_shape = ".objects." in callee_qn

        if is_django_manager_shape and (
            leaf in DJANGO_READ_LEAF_METHODS
            or leaf in DJANGO_WRITE_LEAF_METHODS
            or leaf in DJANGO_BOTH_LEAF_METHODS
        ):
            _classify_django_manager_call(
                hint, caller_qn, leaf, line, callee_qn,
                _is_django_model_class, _resolve_chain_head_to_class,
                table_by_class_qn, table_by_bare_class_name, _emit,
            )
            continue

        if leaf in SA_SESSION_LEAF_METHODS:
            _classify_sa_session_call(
                hint, caller_qn, leaf, line, callee_qn,
                _is_sa_session_class, _resolve_chain_head_to_class,
                _resolve_var_to_class, table_by_class_qn,
                table_by_bare_class_name, _emit,
            )
            continue

        # Plain `Model.objects.<verb>` without inner shape detection
        # (covers QuerySet chain leaves with sequential calls — but our
        # extractor currently emits one hint per call, so this is the
        # primary Django manager path when shape didn't match above).
        if (leaf in DJANGO_READ_LEAF_METHODS
                or leaf in DJANGO_WRITE_LEAF_METHODS
                or leaf in DJANGO_BOTH_LEAF_METHODS):
            _classify_django_manager_call(
                hint, caller_qn, leaf, line, callee_qn,
                _is_django_model_class, _resolve_chain_head_to_class,
                table_by_class_qn, table_by_bare_class_name, _emit,
            )
            continue

        if (leaf in DJANGO_INSTANCE_WRITE_METHODS
                or leaf in DJANGO_INSTANCE_READ_METHODS):
            _classify_django_instance_call(
                hint, caller_qn, leaf, line, callee_qn,
                _is_django_model_class, _resolve_var_to_class,
                _resolve_chain_head_to_class, caller_class_qn,
                table_by_class_qn, table_by_bare_class_name, _emit,
            )
            continue


def _classify_sa_session_call(
    hint: OrmCallHint,
    caller_qn: str,
    leaf: str,
    line: int,
    callee_qn: str,
    is_sa_session_class,
    resolve_chain_head_to_class,
    resolve_var_to_class,
    table_by_class_qn: dict[str, str],
    table_by_bare_class_name: dict[str, str],
    emit,
) -> None:
    """Classify a SQLAlchemy Session method call.

    Receiver type confirmation is best-effort — we accept the call when
    the leaf method matches AND we can resolve the table from the
    argument hint. The callee_qn shape (`Class.method` vs raw
    `session.query`) is what the classifier checks for receiver
    awareness; the literal-name `is_sa_session_class` short-circuits
    the common case where 14g binds `session: Session = SessionLocal()`.
    """
    # Determine op kind from leaf.
    if leaf in {"query", "get", "scalars"}:
        op = "read"
    elif leaf == "execute":
        # Inner construct decides op.
        if hint.inner_call_fn == "select":
            op = "read"
        elif hint.inner_call_fn in {"insert", "update", "delete"}:
            op = "write"
        else:
            return  # text(), literal_column, etc. — drop
    elif leaf in {"add", "add_all", "merge", "delete",
                  "bulk_save_objects", "bulk_insert_mappings",
                  "bulk_update_mappings"}:
        op = "write"
    else:
        return

    # Receiver type sanity check: if callee_qn was rewritten to a known
    # in-repo Class.method (compound resolver landed), confirm Class is
    # an SA Session type. When the call stays raw (`session.query`,
    # `db.add`, etc.) we DON'T strictly check the receiver — the table
    # resolution from the argument hint is the discriminator (drop
    # cleanly when arg_head doesn't map to a TableNode).
    parts = callee_qn.split(".")
    if len(parts) > 2:
        # Multi-segment resolved chain (e.g. `pkg.Class.method`). The
        # receiver IS the resolved class qn — verify.
        recv = ".".join(parts[:-1])
        if not is_sa_session_class(recv):
            return

    # Table identity:
    #   - leaf in {query, get, scalars}, no inner: arg_head is the class identifier
    #   - leaf == execute with inner select/insert/update/delete: inner_call_arg_head
    #   - leaf in {add, merge, delete, add_all}: arg_head is a variable; resolve via local_var_index
    #   - leaf in bulk_save_objects: arg_head is a variable holding a list
    target_class_qn: str | None = None
    if leaf in {"query", "get", "scalars"}:
        head_ident = hint.arg_head
        if not head_ident:
            return
        # First identifier in the chain — `User` in `User` or `User.id`
        first = head_ident.split(".", 1)[0]
        target_class_qn = resolve_chain_head_to_class(caller_qn, first)
    elif leaf == "execute":
        if hint.inner_call_fn not in SA_INNER_CONSTRUCTS:
            return
        inner_head = hint.inner_call_arg_head
        if not inner_head:
            return
        first = inner_head.split(".", 1)[0]
        target_class_qn = resolve_chain_head_to_class(caller_qn, first)
    elif leaf in {"add", "merge", "delete", "add_all"}:
        head_ident = hint.arg_head
        if not head_ident:
            return
        first = head_ident.split(".", 1)[0]
        # Try local-var typing first (most common: `u = User(); s.add(u)`),
        # then fall back to direct class-name resolution (rare).
        target_class_qn = resolve_var_to_class(caller_qn, first)
        if target_class_qn is None:
            target_class_qn = resolve_chain_head_to_class(caller_qn, first)
    elif leaf == "bulk_save_objects":
        head_ident = hint.arg_head
        if not head_ident:
            return
        first = head_ident.split(".", 1)[0]
        target_class_qn = resolve_var_to_class(caller_qn, first)
    elif leaf in {"bulk_insert_mappings", "bulk_update_mappings"}:
        # First positional is the model class itself; second is dicts.
        # arg_head captured the first positional — that IS the model.
        head_ident = hint.arg_head
        if not head_ident:
            return
        first = head_ident.split(".", 1)[0]
        target_class_qn = resolve_chain_head_to_class(caller_qn, first)

    table_name = _table_for_class(
        target_class_qn, hint.arg_head,
        table_by_class_qn, table_by_bare_class_name,
    )
    if table_name is None:
        return
    emit(caller_qn, table_name, op, line)


def _classify_django_manager_call(
    hint: OrmCallHint,
    caller_qn: str,
    leaf: str,
    line: int,
    callee_qn: str,
    is_django_model_class,
    resolve_chain_head_to_class,
    table_by_class_qn: dict[str, str],
    table_by_bare_class_name: dict[str, str],
    emit,
) -> None:
    """Classify `Model.objects.<verb>(...)` and chained QuerySet leaves.

    The receiver chain in `callee_qn` is the unresolved raw form when
    Django (since the resolver doesn't follow Manager → QuerySet). Pull
    the head identifier (the class name) and resolve.
    """
    # Determine op.
    if leaf in DJANGO_READ_LEAF_METHODS:
        ops: list[str] = ["read"]
    elif leaf in DJANGO_WRITE_LEAF_METHODS:
        ops = ["write"]
    elif leaf in DJANGO_BOTH_LEAF_METHODS:
        ops = ["read", "write"]
    else:
        return

    # Extract the head identifier from the callee chain.
    # callee_qn likely looks like `User.objects.filter` or
    # `User.objects.create` (bare). Pre-resolution rewrite only attempts
    # on resolvable receivers; Django Manager calls fall through.
    parts = callee_qn.split(".")
    if len(parts) < 3:
        # `something.filter` — chained off a non-named receiver. Skip
        # (caught by the chain-collapse rule in spec section C).
        return
    if parts[1] != "objects":
        # `User.active.filter` — custom manager; defer per spec.
        return
    head = parts[0]

    target_class_qn = resolve_chain_head_to_class(caller_qn, head)
    if target_class_qn is None:
        return
    if not is_django_model_class(target_class_qn):
        return

    table_name = _table_for_class(
        target_class_qn, head,
        table_by_class_qn, table_by_bare_class_name,
    )
    if table_name is None:
        return

    for op in ops:
        emit(caller_qn, table_name, op, line)


def _classify_django_instance_call(
    hint: OrmCallHint,
    caller_qn: str,
    leaf: str,
    line: int,
    callee_qn: str,
    is_django_model_class,
    resolve_var_to_class,
    resolve_chain_head_to_class,
    caller_class_qn: dict[str, str],
    table_by_class_qn: dict[str, str],
    table_by_bare_class_name: dict[str, str],
    emit,
) -> None:
    """Classify `instance.save()` / `instance.delete()` / `.refresh_from_db()`.

    Receiver must resolve to a Django Model class. `self.save()` inside
    a `Model.save` override is suppressed to avoid double-counting the
    framework's write.
    """
    if leaf in DJANGO_INSTANCE_WRITE_METHODS:
        op = "write"
    elif leaf in DJANGO_INSTANCE_READ_METHODS:
        op = "read"
    else:
        return

    # callee_qn after resolution is `Class.method` for self/var-typed
    # receivers, or raw `var.save` for unresolved.
    if "." not in callee_qn:
        return
    recv_text, method = callee_qn.rsplit(".", 1)
    if method != leaf:
        return

    # Direct Class.method — receiver is a class qn.
    if "." in recv_text or recv_text in {"self", "this"}:
        # If receiver is `Class.method` after rewrite, recv_text could
        # itself be a dotted class qn.
        if recv_text in {"self", "this"}:
            target_class_qn = caller_class_qn.get(caller_qn)
        else:
            # Try whole as class qn (post-resolve case).
            target_class_qn = recv_text
            # Validate it actually exists in our model index.
            if target_class_qn not in table_by_class_qn:
                # Fall back to bare resolution.
                bare = recv_text.rsplit(".", 1)[-1]
                fallback = table_by_bare_class_name.get(bare)
                if fallback is None:
                    return
    else:
        # Unresolved — recv_text is a bare local-var name.
        target_class_qn = resolve_var_to_class(caller_qn, recv_text)
        if target_class_qn is None:
            return

    if target_class_qn is None:
        return
    if not is_django_model_class(target_class_qn):
        # Could be a non-Model class with a `save` method. Drop.
        return

    # Suppress `super().save()` and self-recursive `save()` inside an override.
    enclosing_class = caller_class_qn.get(caller_qn)
    if (enclosing_class is not None
            and enclosing_class == target_class_qn
            and caller_qn.endswith(f".{leaf}")):
        return

    table_name = _table_for_class(
        target_class_qn, recv_text.rsplit(".", 1)[-1],
        table_by_class_qn, table_by_bare_class_name,
    )
    if table_name is None:
        return
    emit(caller_qn, table_name, op, line)


def _table_for_class(
    target_class_qn: str | None,
    bare_fallback: str,
    table_by_class_qn: dict[str, str],
    table_by_bare_class_name: dict[str, str],
) -> str | None:
    """Resolve a class qn (or bare name fallback) to a table name."""
    if target_class_qn is not None:
        hit = table_by_class_qn.get(target_class_qn)
        if hit is not None:
            return hit
        # Bare-name fallback for cases where the class qn missed (e.g.
        # the model is in another file we couldn't resolve through 12c).
        bare = target_class_qn.rsplit(".", 1)[-1]
        hit = table_by_bare_class_name.get(bare)
        if hit is not None:
            return hit
    if bare_fallback:
        hit = table_by_bare_class_name.get(bare_fallback)
        if hit is not None:
            return hit
    return None


__all__ = ["classify_orm_calls"]
