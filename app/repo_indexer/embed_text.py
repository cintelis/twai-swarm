"""Per-symbol embedding-text generation.

Sprint 14a. Mirror of GitNexus's `text-generator.ts`: fold the structural
context we already extracted (qualified name, params, container, docstring)
into a single deterministic string per Function / Class. That string is
what `phases/embed.py` hands to `app.embeddings.embed_text`.

Pure functions on dataclasses — no I/O, no tree-sitter, no Neo4j. Trivially
unit-testable; deterministic across runs (so a re-index of unchanged code
produces byte-identical embedding inputs, which is the precondition for
cache-hit-style optimization later).

Why this format
---------------
The retrieval-time query at 14b will be free-text English ("find auth code",
"where do we parse JSON"). That's matched against the embedding of the
*symbol's purpose*, which is best surfaced by:

1. **kind + qualified name** — gives the embedding both the lexical name
   ("login_handler") and the structural location ("app.auth.login_handler").
2. **params** — function signatures carry intent (`def authenticate(user, password)`
   embeds very differently from `def authenticate()`).
3. **container** — the parent class qn (for methods) or file path (for
   top-level fns). Surfaces co-location signal: methods on `AuthService`
   should cluster with each other in vector space.
4. **docstring** — the cleanest natural-language signal we have. Truncated
   at DOCSTRING_MAX_CHARS to avoid letting one verbose docstring dominate
   the embedding budget.
"""
from __future__ import annotations

from .actions import ClassNode, FunctionNode

# Truncate docstrings before they hit the embedding input. The embedder
# itself truncates at MAX_INPUT_CHARS (8000), but we want each section to
# stay roughly proportional — a 5000-char docstring would crowd out the
# rest of the structural context.
DOCSTRING_MAX_CHARS = 500


def _kind_for(fn: FunctionNode) -> str:
    """Return the human-readable function kind label.

    `method` wins over `async function` when both apply — async methods
    are still methods first, and the container line already carries the
    class context.
    """
    if fn.is_method:
        return "method"
    if fn.is_async:
        return "async function"
    return "function"


def _truncate_docstring(docstring: str) -> str:
    """Trim docstrings at DOCSTRING_MAX_CHARS.

    No ellipsis appended — embedders care about content, not punctuation,
    and the truncation is deterministic on the input.
    """
    if not docstring:
        return ""
    if len(docstring) <= DOCSTRING_MAX_CHARS:
        return docstring
    return docstring[:DOCSTRING_MAX_CHARS]


def embedding_text_for_function(fn: FunctionNode) -> str:
    """Per-symbol embedding text for a Function.

    Format (deterministic, single string, sections joined by `\\n`):

        {kind} {qualified_name}
        params: {p1}, {p2}, ...        # only when params are non-empty
        in {container}                 # parent_class_qn if method, else file_path
        {docstring}                    # truncated to DOCSTRING_MAX_CHARS, only when non-empty

    Sections that would be empty are omitted entirely so the embedder
    doesn't see dangling labels.
    """
    lines = [f"{_kind_for(fn)} {fn.qualified_name}"]

    if fn.params:
        lines.append(f"params: {', '.join(fn.params)}")

    container = fn.parent_class_qn if (fn.is_method and fn.parent_class_qn) else fn.file_path
    lines.append(f"in {container}")

    docstring = _truncate_docstring(fn.docstring)
    if docstring:
        lines.append(docstring)

    return "\n".join(lines)


def embedding_text_for_class(cls: ClassNode) -> str:
    """Per-symbol embedding text for a Class.

    Format (deterministic, single string, sections joined by `\\n`):

        class {qualified_name}
        in {file_path}
        {docstring}                    # truncated, only when non-empty

    No params line — classes don't have a flat param list (the __init__
    method gets its own embedding via embedding_text_for_function).
    """
    lines = [
        f"class {cls.qualified_name}",
        f"in {cls.file_path}",
    ]
    docstring = _truncate_docstring(cls.docstring)
    if docstring:
        lines.append(docstring)
    return "\n".join(lines)
