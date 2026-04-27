"""Package-level contract: every promised export is reachable from the root."""
from __future__ import annotations


def test_all_exports_importable_from_package_root():
    """Single import statement should pull every public name."""
    from app.repo_indexer.scope_resolution import (  # noqa: F401
        Declaration,
        ModuleScopeIndex,
        Position,
        PositionIndex,
        QualifiedNameIndex,
        Range,
        ScopeId,
        ScopeTree,
        ScopeTreeInvariantError,
        build_module_scope_index,
        build_position_index,
        build_qualified_name_index,
        build_scope_tree,
        format_range,
        range_strictly_contains,
        ranges_overlap,
        start_is_at_or_before,
    )


def test_dunder_all_matches_actual_exports():
    """`__all__` should advertise exactly the names we exposed above."""
    from app.repo_indexer import scope_resolution as sr

    expected = {
        "Position",
        "Range",
        "ScopeId",
        "Declaration",
        "ScopeTree",
        "ScopeTreeInvariantError",
        "build_scope_tree",
        "ranges_overlap",
        "range_strictly_contains",
        "start_is_at_or_before",
        "format_range",
        "PositionIndex",
        "build_position_index",
        "ModuleScopeIndex",
        "build_module_scope_index",
        "QualifiedNameIndex",
        "build_qualified_name_index",
    }
    assert set(sr.__all__) == expected


def test_no_tree_sitter_required():
    """Importing the package must not pull in tree-sitter.

    The package contract is "consumes pre-extracted data, doesn't parse" —
    so even if tree-sitter were absent from the environment, this import
    chain should succeed. We can't easily simulate tree-sitter being
    missing in the test environment, so we just assert the package
    namespace doesn't reference it.
    """
    import sys

    # Force-import (in case it's cached lazily).
    from app.repo_indexer import scope_resolution  # noqa: F401

    # Look at every submodule we ship; none of them should have imported
    # tree_sitter at import time.
    submods = [
        "app.repo_indexer.scope_resolution",
        "app.repo_indexer.scope_resolution.types",
        "app.repo_indexer.scope_resolution.scope_tree",
        "app.repo_indexer.scope_resolution.position_index",
        "app.repo_indexer.scope_resolution.module_scope_index",
        "app.repo_indexer.scope_resolution.qualified_name_index",
    ]
    for name in submods:
        mod = sys.modules.get(name)
        assert mod is not None, f"{name} should be loaded"
        # No attribute named tree_sitter should be reachable on the module.
        assert not hasattr(mod, "tree_sitter")


def test_no_app_repo_indexer_runtime_dependency():
    """The package should not import from app.repo_indexer.* runtime modules.

    Sprint 12a contract: standalone, peer to the indexer, not a child.
    Allowed: app.repo_indexer.scope_resolution.* (siblings inside the
    package). Forbidden: walker, resolver, loader, extractors, runner,
    phases, __main__, actions.
    """
    import sys

    forbidden = {
        "app.repo_indexer.walker",
        "app.repo_indexer.resolver",
        "app.repo_indexer.loader",
        "app.repo_indexer.extractor_python",
        "app.repo_indexer.extractor_typescript",
        "app.repo_indexer.runner",
        "app.repo_indexer.__main__",
        "app.repo_indexer.actions",
    }

    # Snapshot the modules already loaded BEFORE we touch scope_resolution.
    # Then re-import a fresh copy and check no forbidden module appears.
    pre = set(sys.modules.keys())
    # Drop any scope_resolution modules so re-import is meaningful.
    for name in list(sys.modules.keys()):
        if name.startswith("app.repo_indexer.scope_resolution"):
            del sys.modules[name]

    from app.repo_indexer import scope_resolution  # noqa: F401

    # Find modules newly added by our import.
    post = set(sys.modules.keys())
    added = post - pre
    leaked = added & forbidden
    assert not leaked, (
        f"scope_resolution leaked dependency on {leaked}; "
        "package must stay standalone"
    )
