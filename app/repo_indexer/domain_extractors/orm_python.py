"""Sprint 15b.1 — ORM declaration extraction for Python sources.

Recognises ORM model classes and emits TableNode + ColumnNode records:

    SQLAlchemy declarative:
        class User(Base):
            __tablename__ = "users"
            id = Column(Integer, primary_key=True)
            name = Column(String(50), nullable=False)

    SQLAlchemy 2.0 typed:
        class User(Base):
            __tablename__ = "users"
            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str] = mapped_column(String(50))

    Django:
        class User(models.Model):
            name = models.CharField(max_length=50)
            email = models.EmailField(unique=True)

    SQLAlchemy classical (module-level):
        users = Table("users", metadata,
                      Column("id", Integer, primary_key=True))

The class-walk hook is invoked from `extractor_python._emit_class`; the
classical Table() pass walks module-level assignments separately.

Returns:
    extract_orm_class_decls(class_node, …) → (TableNode|None, list[ColumnNode])
    extract_orm_classical_tables(root, …)  → (list[TableNode], list[ColumnNode])

Detection heuristics:
    - SQLAlchemy declarative: class has any superclass whose bare name
      is a registered Base alias OR contains `__tablename__` literal.
    - SQLAlchemy 2.0 typed: same as declarative + presence of
      `: Mapped[...]` annotated assignments.
    - Django: superclass dotted text contains `models.Model` or bare
      `Model` AND a `models.<*>Field` assignment is found.

Dialect resolution:
    - If `mapped_column(...)` appears: 'sqlalchemy_typed'
    - Else if `Column(...)` appears with a `__tablename__`: 'sqlalchemy_declarative'
    - Else if a `models.<*>Field` appears: 'django'

Conservative — when no ORM signal is recognised, return None.
"""
from __future__ import annotations

from typing import Any

from ..actions import ColumnNode, TableNode

# Class names we treat as a SQLAlchemy declarative `Base`. Pre-14g this
# was the only signal; with 14g typeBindings we COULD additionally check
# the receiver's binding chain — but the literal-name check covers all
# the canonical patterns (`declarative_base()` always returns a class
# named `Base` by convention, and Django's `models.Model` is fixed).
SA_DECLARATIVE_BASE_NAMES = frozenset({
    "Base", "DeclarativeBase", "Model", "ModelBase",
})

DJANGO_MODEL_BASES = frozenset({
    "Model", "models.Model",
    "AbstractUser", "AbstractBaseUser",
    "models.AbstractUser", "models.AbstractBaseUser",
})

# Common Django field constructor names — used for dialect detection
# and column extraction.
DJANGO_FIELD_TYPES = frozenset({
    "CharField", "TextField", "IntegerField", "BigIntegerField",
    "SmallIntegerField", "PositiveIntegerField", "PositiveSmallIntegerField",
    "FloatField", "DecimalField", "BooleanField", "NullBooleanField",
    "DateField", "DateTimeField", "TimeField", "DurationField",
    "EmailField", "URLField", "SlugField", "UUIDField",
    "FileField", "ImageField", "FilePathField",
    "ForeignKey", "OneToOneField", "ManyToManyField",
    "AutoField", "BigAutoField", "SmallAutoField",
    "JSONField", "BinaryField", "GenericIPAddressField",
    "IPAddressField",
})


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _string_literal_text(source: bytes, string_node: Any) -> str | None:
    if string_node is None or string_node.type != "string":
        return None
    parts: list[str] = []
    for c in string_node.children:
        if c.type in ("string_start", "string_end"):
            continue
        if c.type == "interpolation":
            return None
        if c.type == "string_content":
            parts.append(_node_text(source, c))
        else:
            parts.append(_node_text(source, c))
    return "".join(parts)


def _flatten_attribute(source: bytes, n: Any) -> str | None:
    """`a.b.c` → "a.b.c"; identifiers pass through; everything else → None."""
    if n is None:
        return None
    if n.type == "identifier":
        return _node_text(source, n)
    if n.type == "attribute":
        obj = n.child_by_field_name("object")
        attr = n.child_by_field_name("attribute")
        base = _flatten_attribute(source, obj) if obj is not None else None
        if base is None or attr is None:
            return None
        return f"{base}.{_node_text(source, attr)}"
    return None


def _bool_kwarg(source: bytes, args_node: Any, name: str) -> bool | None:
    """Return True/False if `name=True`/`name=False` is present, else None."""
    if args_node is None:
        return None
    for c in args_node.children:
        if c.type != "keyword_argument":
            continue
        kw_name = c.child_by_field_name("name")
        kw_value = c.child_by_field_name("value")
        if kw_name is None or kw_value is None:
            continue
        if _node_text(source, kw_name) != name:
            continue
        v = _node_text(source, kw_value).strip()
        if v == "True":
            return True
        if v == "False":
            return False
        return None
    return None


def _string_kwarg(source: bytes, args_node: Any, name: str) -> str | None:
    if args_node is None:
        return None
    for c in args_node.children:
        if c.type != "keyword_argument":
            continue
        kw_name = c.child_by_field_name("name")
        kw_value = c.child_by_field_name("value")
        if kw_name is None or kw_value is None:
            continue
        if _node_text(source, kw_name) != name:
            continue
        if kw_value.type != "string":
            return None
        return _string_literal_text(source, kw_value)
    return None


def _first_positional_string(source: bytes, args_node: Any) -> str | None:
    if args_node is None:
        return None
    for c in args_node.children:
        if c.type in ("(", ")", ","):
            continue
        if c.type == "keyword_argument":
            continue
        if c.type == "string":
            return _string_literal_text(source, c)
        return None
    return None


def _first_positional_arg_text(source: bytes, args_node: Any) -> str | None:
    """Return the source text of the first positional arg (any shape)."""
    if args_node is None:
        return None
    for c in args_node.children:
        if c.type in ("(", ")", ","):
            continue
        if c.type == "keyword_argument":
            continue
        return _node_text(source, c)
    return None


def _foreign_key_target(source: bytes, args_node: Any) -> str:
    """Find the first `ForeignKey("table.col")` arg (positional or
    inside a nested call) and return the bare `table` portion."""
    if args_node is None:
        return ""
    for c in args_node.children:
        target = ""
        if c.type == "call":
            fn = c.child_by_field_name("function")
            inner_args = c.child_by_field_name("arguments")
            if fn is None:
                continue
            fn_name = _flatten_attribute(source, fn)
            if fn_name is not None and fn_name.split(".")[-1] == "ForeignKey":
                literal = _first_positional_string(source, inner_args)
                if literal:
                    target = literal.split(".", 1)[0]
        elif c.type == "keyword_argument":
            value = c.child_by_field_name("value")
            if value is not None and value.type == "call":
                fn = value.child_by_field_name("function")
                inner_args = value.child_by_field_name("arguments")
                if fn is None:
                    continue
                fn_name = _flatten_attribute(source, fn)
                if fn_name is not None and fn_name.split(".")[-1] == "ForeignKey":
                    literal = _first_positional_string(source, inner_args)
                    if literal:
                        target = literal.split(".", 1)[0]
        if target:
            return target
    return ""


def _collect_superclass_names(source: bytes, class_node: Any) -> list[str]:
    """Return the list of superclass dotted names for a class."""
    out: list[str] = []
    superclasses = class_node.child_by_field_name("superclasses")
    if superclasses is None:
        return out
    for sub in superclasses.children:
        if sub.type in ("identifier", "attribute"):
            out.append(_node_text(source, sub))
    return out


def _is_django_model_class(superclass_names: list[str]) -> bool:
    """A class with any superclass dotted-name in DJANGO_MODEL_BASES."""
    for s in superclass_names:
        if s in DJANGO_MODEL_BASES:
            return True
        # Trailing "Model" or "models.Model" — match both forms.
        if s.endswith(".Model") or s.endswith(".AbstractUser") or s.endswith(".AbstractBaseUser"):
            return True
    return False


def _is_sa_declarative_class(superclass_names: list[str]) -> bool:
    """A class whose superclass bare name is in SA_DECLARATIVE_BASE_NAMES.

    Excludes Django pattern (handled separately) — caller must already
    have rejected `models.Model`-style superclasses.
    """
    for s in superclass_names:
        bare = s.rsplit(".", 1)[-1]
        if bare in SA_DECLARATIVE_BASE_NAMES and bare != "Model":
            return True
    return False


def _find_tablename_literal(source: bytes, body_node: Any) -> str | None:
    """Walk the class body for `__tablename__ = "users"` and return "users"."""
    if body_node is None:
        return None
    for c in body_node.children:
        if c.type == "expression_statement":
            for sub in c.children:
                if sub.type != "assignment":
                    continue
                left = sub.child_by_field_name("left")
                right = sub.child_by_field_name("right")
                if left is None or right is None:
                    continue
                if left.type != "identifier":
                    continue
                if _node_text(source, left) != "__tablename__":
                    continue
                if right.type != "string":
                    continue
                return _string_literal_text(source, right)
    return None


def _find_django_meta_db_table(source: bytes, body_node: Any) -> str | None:
    """Walk for `class Meta: db_table = "users"` (Django convention)."""
    if body_node is None:
        return None
    for c in body_node.children:
        if c.type != "class_definition":
            continue
        name_node = c.child_by_field_name("name")
        if name_node is None or _node_text(source, name_node) != "Meta":
            continue
        meta_body = c.child_by_field_name("body")
        if meta_body is None:
            continue
        for mc in meta_body.children:
            if mc.type != "expression_statement":
                continue
            for sub in mc.children:
                if sub.type != "assignment":
                    continue
                left = sub.child_by_field_name("left")
                right = sub.child_by_field_name("right")
                if left is None or right is None:
                    continue
                if left.type != "identifier":
                    continue
                if _node_text(source, left) != "db_table":
                    continue
                if right.type != "string":
                    continue
                return _string_literal_text(source, right)
    return None


def _camel_to_snake(name: str) -> str:
    """Django's default `db_table = "<app>_<modelname_snake>"`. We don't
    know the app label, so emit just the snake-cased model name when no
    Meta.db_table is set. Caller can always join with the model_qn for
    disambiguation.
    """
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _extract_sa_column_assignment(
    source: bytes, assign_node: Any,
) -> tuple[str, str, dict] | None:
    """SQLAlchemy declarative + 2.0 typed columns.

    Recognises:
        id = Column(Integer, primary_key=True)
        id: Mapped[int] = mapped_column(Integer, primary_key=True)

    Returns (col_name, type_raw, kwargs_dict) or None if not a recognised
    SA column declaration.

    `kwargs_dict` contains: primary_key, nullable, indexed, foreign_key_table.
    """
    left = assign_node.child_by_field_name("left")
    right = assign_node.child_by_field_name("right")
    if right is None or right.type != "call":
        return None

    # Bare `id = Column(...)`
    if left is not None and left.type == "identifier":
        col_name = _node_text(source, left)
    else:
        return None

    fn = right.child_by_field_name("function")
    args = right.child_by_field_name("arguments")
    if fn is None:
        return None
    fn_name = _flatten_attribute(source, fn)
    if fn_name is None:
        return None
    leaf = fn_name.split(".")[-1]
    if leaf not in {"Column", "mapped_column"}:
        return None

    # type_raw: first positional arg if not a string (string positional
    # is the column name override in SA — `Column("col_name", Integer)`).
    type_raw = ""
    if args is not None:
        positional_count = 0
        for c in args.children:
            if c.type in ("(", ")", ","):
                continue
            if c.type == "keyword_argument":
                continue
            positional_count += 1
            text = _node_text(source, c)
            # Skip column-name override (string literal in first slot)
            if positional_count == 1 and c.type == "string":
                # use as the actual column name override; type is the next positional
                col_name_override = _string_literal_text(source, c)
                if col_name_override:
                    col_name = col_name_override
                continue
            # First non-string positional is the type
            type_raw = text
            break

    pk = _bool_kwarg(source, args, "primary_key") or False
    nullable_val = _bool_kwarg(source, args, "nullable")
    nullable = True if nullable_val is None else nullable_val
    indexed = _bool_kwarg(source, args, "index") or False
    fk_table = _foreign_key_target(source, args)

    return (col_name, type_raw, {
        "primary_key": pk,
        "nullable": nullable,
        "indexed": indexed,
        "foreign_key_table": fk_table,
    })


def _extract_sa_typed_assignment(
    source: bytes, assign_node: Any,
) -> tuple[str, str, dict] | None:
    """SQLAlchemy 2.0 typed-column shape:

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        name: Mapped[str] = mapped_column()

    The annotation lives on the `assignment` node's "type" field for
    annotated-assignment grammar — but tree-sitter-python parses this
    as `assignment` with a child `type` field carrying the `: Mapped[X]`
    portion. The RHS call (mapped_column) handling matches
    `_extract_sa_column_assignment`.
    """
    left = assign_node.child_by_field_name("left")
    type_node = assign_node.child_by_field_name("type")
    right = assign_node.child_by_field_name("right")
    if left is None or left.type != "identifier":
        return None
    if type_node is None:
        return None
    col_name = _node_text(source, left)

    # Type-raw preference: the inner type from `Mapped[X]` annotation.
    type_text = _node_text(source, type_node)
    type_raw = type_text
    if "[" in type_text and type_text.endswith("]"):
        bracket = type_text.index("[")
        wrapper = type_text[:bracket].strip()
        inner = type_text[bracket + 1:-1].strip()
        if wrapper == "Mapped":
            type_raw = inner

    if right is None or right.type != "call":
        # `id: Mapped[int]` with no RHS — bare typed annotation, still
        # a column. Best-effort: emit with default kwargs.
        return (col_name, type_raw, {
            "primary_key": False, "nullable": True,
            "indexed": False, "foreign_key_table": "",
        })

    fn = right.child_by_field_name("function")
    args = right.child_by_field_name("arguments")
    if fn is None:
        return None
    fn_name = _flatten_attribute(source, fn)
    if fn_name is None:
        return None
    leaf = fn_name.split(".")[-1]
    if leaf not in {"Column", "mapped_column"}:
        return None

    pk = _bool_kwarg(source, args, "primary_key") or False
    nullable_val = _bool_kwarg(source, args, "nullable")
    nullable = True if nullable_val is None else nullable_val
    indexed = _bool_kwarg(source, args, "index") or False
    fk_table = _foreign_key_target(source, args)

    return (col_name, type_raw, {
        "primary_key": pk,
        "nullable": nullable,
        "indexed": indexed,
        "foreign_key_table": fk_table,
    })


def _extract_django_field_assignment(
    source: bytes, assign_node: Any,
) -> tuple[str, str, dict] | None:
    """Django field shape:

        name = models.CharField(max_length=50)
        email = models.EmailField(unique=True)
        owner = models.ForeignKey(User, on_delete=models.CASCADE)

    Returns (col_name, "models.<FieldName>" type_raw, kwargs_dict) or None.
    """
    left = assign_node.child_by_field_name("left")
    right = assign_node.child_by_field_name("right")
    if left is None or left.type != "identifier":
        return None
    if right is None or right.type != "call":
        return None
    fn = right.child_by_field_name("function")
    args = right.child_by_field_name("arguments")
    if fn is None:
        return None
    fn_name = _flatten_attribute(source, fn)
    if fn_name is None:
        return None
    leaf = fn_name.split(".")[-1]
    if leaf not in DJANGO_FIELD_TYPES:
        return None

    col_name = _node_text(source, left)
    type_raw = fn_name

    # Django convention: `null=True` ⇒ nullable; `db_index=True` ⇒ indexed;
    # `primary_key=True`. ForeignKey first positional arg is the related
    # model's bare name — we treat this as the foreign_key_table guess.
    pk = _bool_kwarg(source, args, "primary_key") or False
    null_val = _bool_kwarg(source, args, "null")
    # Django default null=False (db column is NOT NULL).
    nullable = bool(null_val) if null_val is not None else False
    indexed = _bool_kwarg(source, args, "db_index") or False

    fk_table = ""
    if leaf in {"ForeignKey", "OneToOneField", "ManyToManyField"}:
        first_pos = _first_positional_arg_text(source, args)
        if first_pos:
            # Strip surrounding quotes if it was a string ("User"); leave
            # bare name otherwise. Best-effort — the model name doesn't
            # always equal the table name in Django, but it's the closest
            # signal we have at extract time.
            fk_table = first_pos.strip('"').strip("'")

    return (col_name, type_raw, {
        "primary_key": pk,
        "nullable": nullable,
        "indexed": indexed,
        "foreign_key_table": fk_table,
    })


def extract_orm_class_decls(
    source: bytes,
    class_node: Any,
    class_qn: str,
    file_path: str,
    repo_name: str,
    tenant_id: str,
) -> tuple[TableNode | None, list[ColumnNode]]:
    """Inspect a class for ORM declaration patterns.

    Returns (TableNode|None, list[ColumnNode]). None when the class
    doesn't match any recognised ORM dialect — the caller should then
    skip this class for ORM purposes.
    """
    body = class_node.child_by_field_name("body")
    if body is None:
        return None, []

    superclasses = _collect_superclass_names(source, class_node)
    line_start = class_node.start_point[0] + 1
    name_node = class_node.child_by_field_name("name")
    if name_node is None:
        return None, []
    class_bare_name = _node_text(source, name_node)

    is_django = _is_django_model_class(superclasses)
    is_sa = _is_sa_declarative_class(superclasses)

    if not is_django and not is_sa:
        return None, []

    if is_django:
        # Try Meta.db_table, fall back to snake_case of class name.
        table_name = _find_django_meta_db_table(source, body)
        if not table_name:
            table_name = _camel_to_snake(class_bare_name)
        columns: list[ColumnNode] = []
        for c in body.children:
            if c.type != "expression_statement":
                continue
            for sub in c.children:
                if sub.type != "assignment":
                    continue
                result = _extract_django_field_assignment(source, sub)
                if result is None:
                    continue
                col_name, type_raw, kwargs = result
                columns.append(ColumnNode(
                    repo=repo_name,
                    tenant_id=tenant_id,
                    table_name=table_name,
                    name=col_name,
                    type_raw=type_raw,
                    primary_key=kwargs["primary_key"],
                    nullable=kwargs["nullable"],
                    indexed=kwargs["indexed"],
                    foreign_key_table=kwargs["foreign_key_table"],
                    file_path=file_path,
                    line_start=sub.start_point[0] + 1,
                ))
        return (
            TableNode(
                repo=repo_name,
                tenant_id=tenant_id,
                name=table_name,
                model_qn=class_qn,
                dialect="django",
                schema="",
                file_path=file_path,
                line_start=line_start,
            ),
            columns,
        )

    # SQLAlchemy declarative or 2.0 typed
    table_name = _find_tablename_literal(source, body)
    if not table_name:
        # No __tablename__ — could be an abstract base. Skip.
        return None, []

    columns_sa: list[ColumnNode] = []
    saw_mapped_column = False
    for c in body.children:
        if c.type != "expression_statement":
            continue
        for sub in c.children:
            if sub.type != "assignment":
                continue
            # Try typed (SA 2.0) first — has a type annotation.
            type_node = sub.child_by_field_name("type")
            if type_node is not None:
                result = _extract_sa_typed_assignment(source, sub)
            else:
                result = _extract_sa_column_assignment(source, sub)
            if result is None:
                continue
            col_name, type_raw, kwargs = result
            # Track whether mapped_column was used for dialect classification.
            right = sub.child_by_field_name("right")
            if right is not None and right.type == "call":
                fn = right.child_by_field_name("function")
                if fn is not None:
                    fn_name = _flatten_attribute(source, fn)
                    if fn_name and fn_name.split(".")[-1] == "mapped_column":
                        saw_mapped_column = True
            columns_sa.append(ColumnNode(
                repo=repo_name,
                tenant_id=tenant_id,
                table_name=table_name,
                name=col_name,
                type_raw=type_raw,
                primary_key=kwargs["primary_key"],
                nullable=kwargs["nullable"],
                indexed=kwargs["indexed"],
                foreign_key_table=kwargs["foreign_key_table"],
                file_path=file_path,
                line_start=sub.start_point[0] + 1,
            ))

    dialect = "sqlalchemy_typed" if saw_mapped_column else "sqlalchemy_declarative"
    return (
        TableNode(
            repo=repo_name,
            tenant_id=tenant_id,
            name=table_name,
            model_qn=class_qn,
            dialect=dialect,
            schema="",
            file_path=file_path,
            line_start=line_start,
        ),
        columns_sa,
    )


def extract_orm_classical_tables(
    source: bytes,
    root: Any,
    module_qn: str,
    file_path: str,
    repo_name: str,
    tenant_id: str,
) -> tuple[list[TableNode], list[ColumnNode]]:
    """Walk module-level for `users = Table("users", metadata, Column(...), …)`.

    Classical-style SQLAlchemy declarations with no class. Skips entries
    that aren't bare-name LHS assignments to a `Table(...)` call.

    Each Column inside the Table args becomes a ColumnNode owned by the
    table's name. Same kwargs heuristic as `_extract_sa_column_assignment`.
    """
    tables: list[TableNode] = []
    columns: list[ColumnNode] = []

    for child in root.children:
        if child.type != "expression_statement":
            continue
        for sub in child.children:
            if sub.type != "assignment":
                continue
            left = sub.child_by_field_name("left")
            right = sub.child_by_field_name("right")
            if left is None or left.type != "identifier":
                continue
            if right is None or right.type != "call":
                continue
            fn = right.child_by_field_name("function")
            args = right.child_by_field_name("arguments")
            if fn is None:
                continue
            fn_name = _flatten_attribute(source, fn)
            if fn_name is None or fn_name.split(".")[-1] != "Table":
                continue
            table_name = _first_positional_string(source, args)
            if not table_name:
                continue
            line_start = sub.start_point[0] + 1
            tables.append(TableNode(
                repo=repo_name,
                tenant_id=tenant_id,
                name=table_name,
                model_qn="",  # classical Table() has no class
                dialect="sqlalchemy_classical",
                schema="",
                file_path=file_path,
                line_start=line_start,
            ))
            # Walk the Table()'s positional args for Column(...) entries.
            if args is not None:
                for arg_child in args.children:
                    if arg_child.type != "call":
                        continue
                    inner_fn = arg_child.child_by_field_name("function")
                    inner_args = arg_child.child_by_field_name("arguments")
                    if inner_fn is None:
                        continue
                    inner_fn_name = _flatten_attribute(source, inner_fn)
                    if inner_fn_name is None or inner_fn_name.split(".")[-1] != "Column":
                        continue
                    col_name = _first_positional_string(source, inner_args)
                    if not col_name:
                        continue
                    # Type is the next positional after the name string.
                    type_raw = ""
                    seen_name = False
                    if inner_args is not None:
                        for ic in inner_args.children:
                            if ic.type in ("(", ")", ","):
                                continue
                            if ic.type == "keyword_argument":
                                continue
                            if not seen_name and ic.type == "string":
                                seen_name = True
                                continue
                            type_raw = _node_text(source, ic)
                            break
                    pk = _bool_kwarg(source, inner_args, "primary_key") or False
                    nullable_val = _bool_kwarg(source, inner_args, "nullable")
                    nullable = True if nullable_val is None else nullable_val
                    indexed = _bool_kwarg(source, inner_args, "index") or False
                    fk_table = _foreign_key_target(source, inner_args)
                    columns.append(ColumnNode(
                        repo=repo_name,
                        tenant_id=tenant_id,
                        table_name=table_name,
                        name=col_name,
                        type_raw=type_raw,
                        primary_key=pk,
                        nullable=nullable,
                        indexed=indexed,
                        foreign_key_table=fk_table,
                        file_path=file_path,
                        line_start=arg_child.start_point[0] + 1,
                    ))

    # Module-level placeholder: module_qn is currently unused, kept for
    # potential future qn-prefixing.
    _ = module_qn
    return tables, columns


# ─── Sprint 15b.2 — extractor-side ORM call hint capture ────────────────────

# SA / generic ORM leaf method names we want to capture argument hints
# for. The classifier uses these to identify ORM calls and resolve the
# target table.
SA_SESSION_LEAF_METHODS = frozenset({
    "query", "get", "scalars", "execute",
    "add", "add_all", "merge", "delete",
    "bulk_save_objects", "bulk_insert_mappings", "bulk_update_mappings",
})

# Django QuerySet / Manager READ leaf methods
DJANGO_READ_LEAF_METHODS = frozenset({
    "all", "filter", "exclude", "get", "first", "last", "count",
    "exists", "values", "values_list", "only", "defer", "order_by",
    "distinct", "annotate", "aggregate", "in_bulk", "iterator",
    "raw", "none", "reverse", "union", "difference", "intersection",
    "select_related", "prefetch_related",
})

# Django QuerySet / Manager WRITE leaf methods
DJANGO_WRITE_LEAF_METHODS = frozenset({
    "create", "bulk_create", "update", "bulk_update",
    "raw_update",
})

# Django dual-op leaf methods (emit READ + WRITE)
DJANGO_BOTH_LEAF_METHODS = frozenset({
    "get_or_create", "update_or_create",
})

# Django instance methods. Receiver must resolve to a Django Model class.
DJANGO_INSTANCE_WRITE_METHODS = frozenset({"save", "delete"})
DJANGO_INSTANCE_READ_METHODS = frozenset({"refresh_from_db"})

# All ORM leaf methods we want to capture hints for. Used to gate the
# OrmCallHint emit at extraction time.
ORM_LEAF_METHODS = (
    SA_SESSION_LEAF_METHODS
    | DJANGO_READ_LEAF_METHODS
    | DJANGO_WRITE_LEAF_METHODS
    | DJANGO_BOTH_LEAF_METHODS
    | DJANGO_INSTANCE_WRITE_METHODS
    | DJANGO_INSTANCE_READ_METHODS
)

# SA construct names that wrap a class inside `session.execute(...)`.
SA_INNER_CONSTRUCTS = frozenset({"select", "insert", "update", "delete"})


def capture_orm_call_hint(
    source: bytes,
    call_node: Any,
    leaf: str,
    args_node: Any,
) -> tuple[str, str, str] | None:
    """Inspect a `call` AST node whose leaf method is an ORM-recognised
    name; capture (arg_head, inner_call_fn, inner_call_arg_head) for
    the finalize-time classifier.

    Returns None when nothing useful is extractable (skips emit).
    """
    if args_node is None:
        return None

    # First positional arg
    first_pos: Any = None
    for c in args_node.children:
        if c.type in ("(", ")", ","):
            continue
        if c.type == "keyword_argument":
            continue
        first_pos = c
        break

    if first_pos is None:
        # No positional arg — for many SA methods this still matters
        # (e.g. `session.execute()` is malformed but `session.commit()`
        # has no args). Emit with empty arg_head so the classifier can
        # still run boundary checks.
        return ("", "", "")

    # Inner call: `execute(select(User))` shape
    if first_pos.type == "call":
        inner_fn = first_pos.child_by_field_name("function")
        inner_args = first_pos.child_by_field_name("arguments")
        inner_fn_name = _flatten_attribute(source, inner_fn) if inner_fn is not None else None
        if inner_fn_name is not None:
            leaf_inner = inner_fn_name.split(".")[-1]
            inner_arg_head = ""
            if inner_args is not None:
                for ic in inner_args.children:
                    if ic.type in ("(", ")", ","):
                        continue
                    if ic.type == "keyword_argument":
                        continue
                    flat = _flatten_attribute(source, ic)
                    if flat is not None:
                        inner_arg_head = flat
                    break
            return (_flatten_attribute(source, first_pos) or "", leaf_inner, inner_arg_head)

    # Bare positional identifier or attribute chain
    if first_pos.type == "list":
        # `bulk_save_objects([u1, u2])` — peek at first list element
        for lc in first_pos.children:
            if lc.type in ("[", "]", ","):
                continue
            flat = _flatten_attribute(source, lc)
            if flat is not None:
                return (flat, "", "")
            # First non-flat element — stop, no useful hint
            return ("", "", "")
        return ("", "", "")

    flat = _flatten_attribute(source, first_pos)
    if flat is not None:
        return (flat, "", "")

    # Untyped first arg (string literal, dict, etc.) — emit no hint
    return ("", "", "")


__all__ = [
    "extract_orm_class_decls",
    "extract_orm_classical_tables",
    "capture_orm_call_hint",
    "ORM_LEAF_METHODS",
    "SA_SESSION_LEAF_METHODS",
    "DJANGO_READ_LEAF_METHODS",
    "DJANGO_WRITE_LEAF_METHODS",
    "DJANGO_BOTH_LEAF_METHODS",
    "DJANGO_INSTANCE_WRITE_METHODS",
    "DJANGO_INSTANCE_READ_METHODS",
    "SA_INNER_CONSTRUCTS",
]
