"""Sprint 15b — ORM extraction tests for Python.

Two layers:
  1. Declaration tests — `extract_python_file(...,extract_orm=True)`
     against synthetic SQLAlchemy / Django sources, asserting TableNode
     + ColumnNode emit shape.
  2. Call-site classifier tests — full pipeline through `finalize_batch`,
     asserting TableAccessEdge READS/WRITES emit.
"""
from __future__ import annotations

import pytest

try:
    import tree_sitter_python as _tspy  # noqa: F401
    from tree_sitter import Language, Parser
    HAS_TS = True
except Exception:
    HAS_TS = False


from app.repo_indexer.actions import RepoNode  # noqa: E402
from app.repo_indexer.extractor_python import extract_python_file  # noqa: E402
from app.repo_indexer.scope_resolution.finalize import finalize_batch  # noqa: E402

REPO = RepoNode(name="r", url="", commit_sha="")


@pytest.fixture
def parser():
    if not HAS_TS:
        pytest.skip("tree-sitter-python not installed")
    import tree_sitter_python as tspython
    return Parser(Language(tspython.language()))


# ─── 15b.1 — Declaration extraction ─────────────────────────────────────────

def test_orm_disabled_by_default(parser):
    """--with-orm off ⇒ no Table/Column nodes even for clear ORM patterns."""
    src = (
        b"from sqlalchemy.orm import declarative_base\n"
        b"from sqlalchemy import Column, Integer, String\n"
        b"Base = declarative_base()\n"
        b"class User(Base):\n"
        b"    __tablename__ = 'users'\n"
        b"    id = Column(Integer, primary_key=True)\n"
    )
    batch = extract_python_file(REPO, "models.py", src, "sha", parser)
    assert batch.tables == []
    assert batch.columns == []


def test_sqlalchemy_declarative_extraction(parser):
    """Classic declarative pattern: __tablename__ + Column() declarations."""
    src = (
        b"from sqlalchemy import Column, Integer, String, ForeignKey\n"
        b"class User(Base):\n"
        b"    __tablename__ = 'users'\n"
        b"    id = Column(Integer, primary_key=True)\n"
        b"    name = Column(String(50), nullable=False)\n"
        b"    org_id = Column(Integer, ForeignKey('orgs.id'))\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    assert len(batch.tables) == 1
    t = batch.tables[0]
    assert t.name == "users"
    assert t.dialect == "sqlalchemy_declarative"
    assert t.model_qn == "models.User"

    cols_by_name = {c.name: c for c in batch.columns}
    assert set(cols_by_name) == {"id", "name", "org_id"}
    assert cols_by_name["id"].primary_key is True
    assert cols_by_name["name"].nullable is False
    assert cols_by_name["org_id"].foreign_key_table == "orgs"


def test_sqlalchemy_typed_extraction(parser):
    """SA 2.0 typed columns: `id: Mapped[int] = mapped_column(...)`."""
    src = (
        b"from sqlalchemy.orm import Mapped, mapped_column\n"
        b"class User(Base):\n"
        b"    __tablename__ = 'users'\n"
        b"    id: Mapped[int] = mapped_column(primary_key=True)\n"
        b"    name: Mapped[str] = mapped_column()\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    assert len(batch.tables) == 1
    assert batch.tables[0].dialect == "sqlalchemy_typed"
    cols_by_name = {c.name: c for c in batch.columns}
    assert cols_by_name["id"].type_raw == "int"
    assert cols_by_name["id"].primary_key is True


def test_sqlalchemy_classical_table(parser):
    """Classical-style `users = Table("users", metadata, Column(...))`."""
    src = (
        b"from sqlalchemy import Table, Column, Integer, String, MetaData\n"
        b"metadata = MetaData()\n"
        b"users = Table(\n"
        b"    'users', metadata,\n"
        b"    Column('id', Integer, primary_key=True),\n"
        b"    Column('name', String(50)),\n"
        b")\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    assert len(batch.tables) == 1
    t = batch.tables[0]
    assert t.name == "users"
    assert t.dialect == "sqlalchemy_classical"
    assert t.model_qn == ""
    cols_by_name = {c.name: c for c in batch.columns}
    assert set(cols_by_name) == {"id", "name"}
    assert cols_by_name["id"].primary_key is True


def test_django_model_extraction(parser):
    """Django Model with field declarations."""
    src = (
        b"from django.db import models\n"
        b"class User(models.Model):\n"
        b"    name = models.CharField(max_length=50)\n"
        b"    email = models.EmailField(unique=True)\n"
        b"    class Meta:\n"
        b"        db_table = 'users'\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    assert len(batch.tables) == 1
    t = batch.tables[0]
    assert t.name == "users"
    assert t.dialect == "django"
    assert t.model_qn == "models.User"
    cols_by_name = {c.name: c for c in batch.columns}
    assert set(cols_by_name) == {"name", "email"}
    assert cols_by_name["name"].type_raw == "models.CharField"


def test_django_model_default_table_name(parser):
    """Django without Meta.db_table → snake_case of class name."""
    src = (
        b"from django.db import models\n"
        b"class UserProfile(models.Model):\n"
        b"    name = models.CharField(max_length=50)\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    assert len(batch.tables) == 1
    assert batch.tables[0].name == "user_profile"


def test_non_orm_class_skipped(parser):
    """A regular class with no ORM signals shouldn't emit Table/Column."""
    src = (
        b"class Service:\n"
        b"    def __init__(self):\n"
        b"        self.x = 1\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    assert batch.tables == []
    assert batch.columns == []


def test_django_foreign_key_target(parser):
    """ForeignKey first positional arg becomes foreign_key_table."""
    src = (
        b"from django.db import models\n"
        b"class Post(models.Model):\n"
        b"    author = models.ForeignKey('User', on_delete=models.CASCADE)\n"
        b"    class Meta:\n"
        b"        db_table = 'posts'\n"
    )
    batch = extract_python_file(
        REPO, "models.py", src, "sha", parser, extract_orm=True,
    )
    cols = {c.name: c for c in batch.columns}
    assert cols["author"].foreign_key_table == "User"


# ─── 15b.2 — Call-site classifier ───────────────────────────────────────────

def test_django_objects_filter_emits_read(parser):
    """`User.objects.filter(...)` → READ User."""
    src = (
        b"from django.db import models\n"
        b"class User(models.Model):\n"
        b"    class Meta:\n"
        b"        db_table = 'users'\n"
        b"def get_active():\n"
        b"    return User.objects.filter(active=True)\n"
    )
    batch = extract_python_file(
        REPO, "app.py", src, "sha", parser, extract_orm=True,
    )
    assert len(batch.tables) == 1
    assert batch.orm_call_hints, "should have captured filter() call hint"
    finalize_batch(batch)
    reads = [e for e in batch.table_accesses if e.op_kind == "read"]
    assert len(reads) >= 1
    assert reads[0].table_name == "users"
    assert reads[0].function_qn == "app.get_active"


def test_django_objects_create_emits_write(parser):
    """`User.objects.create(...)` → WRITE User."""
    src = (
        b"from django.db import models\n"
        b"class User(models.Model):\n"
        b"    class Meta:\n"
        b"        db_table = 'users'\n"
        b"def create_user(name):\n"
        b"    return User.objects.create(name=name)\n"
    )
    batch = extract_python_file(
        REPO, "app.py", src, "sha", parser, extract_orm=True,
    )
    finalize_batch(batch)
    writes = [e for e in batch.table_accesses if e.op_kind == "write"]
    assert len(writes) >= 1
    assert writes[0].table_name == "users"


def test_django_get_or_create_emits_both(parser):
    """`User.objects.get_or_create(...)` → READ + WRITE User (two edges)."""
    src = (
        b"from django.db import models\n"
        b"class User(models.Model):\n"
        b"    class Meta:\n"
        b"        db_table = 'users'\n"
        b"def upsert(name):\n"
        b"    return User.objects.get_or_create(name=name)\n"
    )
    batch = extract_python_file(
        REPO, "app.py", src, "sha", parser, extract_orm=True,
    )
    finalize_batch(batch)
    ops = sorted(e.op_kind for e in batch.table_accesses)
    assert "read" in ops
    assert "write" in ops


def test_sqlalchemy_session_query_emits_read(parser):
    """`session.query(User)` → READ User."""
    src = (
        b"from sqlalchemy.orm import Session\n"
        b"from sqlalchemy import Column, Integer\n"
        b"class User(Base):\n"
        b"    __tablename__ = 'users'\n"
        b"    id = Column(Integer, primary_key=True)\n"
        b"def list_users(session: Session):\n"
        b"    return session.query(User).all()\n"
    )
    batch = extract_python_file(
        REPO, "app.py", src, "sha", parser, extract_orm=True,
    )
    finalize_batch(batch)
    reads = [e for e in batch.table_accesses if e.op_kind == "read"]
    assert any(r.table_name == "users" for r in reads), \
        f"expected READ users, got {batch.table_accesses}"


def test_sqlalchemy_session_add_emits_write(parser):
    """`session.add(u)` where `u: User` is a local var → WRITE User."""
    src = (
        b"from sqlalchemy.orm import Session\n"
        b"from sqlalchemy import Column, Integer\n"
        b"class User(Base):\n"
        b"    __tablename__ = 'users'\n"
        b"    id = Column(Integer, primary_key=True)\n"
        b"def create_user(session: Session, name):\n"
        b"    u = User()\n"
        b"    session.add(u)\n"
    )
    batch = extract_python_file(
        REPO, "app.py", src, "sha", parser, extract_orm=True,
    )
    finalize_batch(batch)
    writes = [e for e in batch.table_accesses if e.op_kind == "write"]
    assert any(w.table_name == "users" for w in writes), \
        f"expected WRITE users, got {batch.table_accesses}"


def test_no_table_for_unknown_class(parser):
    """`User.objects.filter(...)` where User isn't a known model → no edge."""
    src = (
        b"def f():\n"
        b"    return SomeUnknown.objects.filter(x=1)\n"
    )
    batch = extract_python_file(
        REPO, "app.py", src, "sha", parser, extract_orm=True,
    )
    finalize_batch(batch)
    assert batch.table_accesses == []
