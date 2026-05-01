"""Sprint 17f — HTTP route extraction for Spring (Java) controllers.

Recognises Spring MVC / WebFlux annotation patterns on Java classes and
methods, emits RouteNode + RouteEdge records into the IndexBatch.

Driven by the `annotations: tuple[str, ...]` field on ClassNode and
FunctionNode (added in 17a). The Java extractor captures annotations as
verbatim source strings (`'@RestController'`, `'@RequestMapping("/api")'`,
`'@GetMapping("/users")'`); this module parses the argument list out of
those strings at consumption time.

Patterns covered:

    Class-level marker         @RestController, @Controller
    Class-level prefix         @RequestMapping("/api")
    Verb-specific shorthand    @GetMapping("/users"), @PostMapping(...),
                               @PutMapping(...), @DeleteMapping(...),
                               @PatchMapping(...)
    Long form                  @RequestMapping(method = RequestMethod.GET,
                                               value = "/x")
    Multi-method long form     @RequestMapping(value = "/x",
                                               method = {GET, POST})
                               → emits one RouteNode per method.
    Spring path variables      `/users/{id}` — preserved by normalize_path.

NOT covered (out per Sprint 17f plan):

    - Functional route definitions (`RouterFunctions.route()` API)
    - Spring Cloud Gateway YAML routes
    - Servlet `web.xml` mappings
    - WebFlux `@MessageMapping` / RSocket `@MessageMapping` (HTTP only)
    - `consumes=` / `produces=` content-type filters (could land in
      framework_specifics later if cheap; skipped for v1)
    - Path-variable type inference (`@PathVariable Long id` → Long)

Strict v1 controller gating:

    A class is treated as a Spring controller ONLY when it carries
    `@RestController` or `@Controller`. Bare `@RequestMapping` on a
    `@Service` is NOT enough to enable route extraction. This avoids
    over-emitting routes from non-controller classes that happen to
    use `@RequestMapping` for path-prefix scoping in tests / utilities.
    Documented departure from "scan everything"; revisit if real Spring
    repos demand looser detection.

Wiring:

    The Java extractor (`extractor_java.py::extract_java_file`) accepts
    an `extract_routes: bool = False` flag. When True, it collects
    (class_qn, class_annotations) and (method_qn, method_annotations,
    line_start) tuples during its normal AST walk, then runs this
    module's `extract_routes_for_controller` as a post-pass once the
    full class+method tree is built. Post-pass keeps `_emit_function`
    free of cross-class state and matches how routes_typescript does
    a separate top-level walk after the main extractor finishes.
"""
from __future__ import annotations

import logging
from typing import Any

from ..actions import RouteEdge, RouteNode
from .routes_common import normalize_path


_LOG = logging.getLogger(__name__)


# Marker annotations: any of these on a class enables route extraction.
SPRING_CONTROLLER_MARKERS: frozenset[str] = frozenset({
    "RestController",
    "Controller",
})

# Verb-shorthand mapping: annotation name → HTTP method.
SPRING_VERB_MAPPINGS: dict[str, str] = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
}

# Long-form annotation: `@RequestMapping(method = RequestMethod.GET, ...)`.
# Methods come from the `method=` arg; defaults to ALL methods if absent
# (per Spring docs). For v1 we treat absent-method as a single GET — the
# common-case happy path. If a real repo needs the all-methods fan-out
# we'll revisit; emitting 7 routes per `@RequestMapping("/x")` is noisy.
SPRING_REQUEST_MAPPING_NAME = "RequestMapping"


def _annotation_name(annotation_text: str) -> str:
    """Extract the annotation identifier from a raw `@Foo(...)` string.

    `@RequestMapping("/api")` → `"RequestMapping"`
    `@GetMapping`             → `"GetMapping"`
    `@RestController`         → `"RestController"`
    `@org.springframework.web.bind.annotation.GetMapping("/x")`
                              → `"GetMapping"` (last dotted segment)
    """
    if not annotation_text or not annotation_text.startswith("@"):
        return ""
    # Strip the `@`. Then take everything before the first `(`.
    body = annotation_text[1:].lstrip()
    paren = body.find("(")
    name = body[:paren] if paren >= 0 else body
    name = name.strip()
    # Allow fully-qualified annotation names — keep just the last segment.
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    return name


def _annotation_args_text(annotation_text: str) -> str:
    """Return the parenthesised argument text from `@Foo(...)` minus
    the outer parens. Returns `""` for marker annotations with no args
    (`@Override`) or malformed inputs.

    `@RequestMapping("/api")`             → `'"/api"'`
    `@GetMapping(value = "/x")`           → `'value = "/x"'`
    `@GetMapping`                         → `''`
    `@SuppressWarnings("unchecked")`      → `'"unchecked"'`
    """
    if not annotation_text:
        return ""
    open_paren = annotation_text.find("(")
    if open_paren < 0:
        return ""
    # Find the MATCHING close paren — annotations may contain nested
    # parens (rare but legal: `@Foo(bar = baz(1))`). Scan with a depth
    # counter, respecting quoted strings so a `)` inside `"..."` doesn't
    # trip us up.
    depth = 0
    in_string = False
    string_quote = ""
    end_idx = -1
    for i in range(open_paren, len(annotation_text)):
        ch = annotation_text[i]
        if in_string:
            if ch == "\\":
                # Skip the next char — escape sequence inside a string.
                continue
            if ch == string_quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx < 0:
        return ""
    return annotation_text[open_paren + 1 : end_idx].strip()


def _split_top_level_commas(args_text: str) -> list[str]:
    """Split `args_text` on commas that are NOT inside `{...}`,
    `(...)`, or quoted strings. Used to break up a `key = value, …`
    argument list into individual `key = value` pieces.

    `'value = "/x", method = {GET, POST}'`
        → `['value = "/x"', 'method = {GET, POST}']`
    `'"/x"'`
        → `['"/x"']`
    """
    out: list[str] = []
    if not args_text:
        return out
    depth = 0
    in_string = False
    string_quote = ""
    start = 0
    for i, ch in enumerate(args_text):
        if in_string:
            if ch == "\\":
                continue
            if ch == string_quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            string_quote = ch
            continue
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            piece = args_text[start:i].strip()
            if piece:
                out.append(piece)
            start = i + 1
    tail = args_text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _unquote_string(text: str) -> str | None:
    """If `text` is a quoted string literal (`"..."` or `'...'`),
    return its inner content with simple escape handling. Otherwise None.

    `'"/api"'`     → `'/api'`
    `'"a\\"b"'`    → `'a"b'`
    `'foo'`        → None    (unquoted ident)
    """
    if not text or len(text) < 2:
        return None
    q = text[0]
    if q not in ('"', "'") or text[-1] != q:
        return None
    inner = text[1:-1]
    # Decode the most common escapes. Don't try to be clever about
    # unicode escapes — Spring annotation paths don't contain them in
    # any real-world code.
    out_chars: list[str] = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt in ('"', "'", "\\"):
                out_chars.append(nxt)
                i += 2
                continue
            if nxt == "n":
                out_chars.append("\n")
                i += 2
                continue
            if nxt == "t":
                out_chars.append("\t")
                i += 2
                continue
            # Unknown escape — pass through the backslash + next char.
            out_chars.append(ch)
            i += 1
            continue
        out_chars.append(ch)
        i += 1
    return "".join(out_chars)


def _parse_array_literal(text: str) -> list[str] | None:
    """If `text` is a `{ a, b, c }` array initialiser, split into the
    inner pieces (each piece is a raw token — caller decides whether to
    treat them as strings or ident references). Otherwise None.

    `'{GET, POST}'`           → `['GET', 'POST']`
    `'{"a", "b"}'`            → `['"a"', '"b"']`
    `'{RequestMethod.GET}'`   → `['RequestMethod.GET']`
    `'GET'`                   → None
    """
    if not text:
        return None
    s = text.strip()
    if not s.startswith("{") or not s.endswith("}"):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    return _split_top_level_commas(inner)


def parse_annotation_args(annotation_text: str) -> dict[str, Any]:
    """Parse the argument list of a Spring annotation into a dict.

    Returns:
        {} for marker annotations / no-arg annotations.
        {"value": "..."} for the single-positional-string sugar form.
        {key: value, ...} for `key = value` named-arg form, where each
            value is one of:
              - str (unquoted from a string literal)
              - list[str] (parsed from `{a, b}` array — each entry is the
                raw inner token, which may itself be a quoted string or
                an enum reference like `RequestMethod.GET`)
              - the raw verbatim token for unrecognised value shapes
                (idents like `RequestMethod.GET`, numeric literals, etc.)

    Defensive: any parse failure logs a warning and returns whatever
    pieces were successfully extracted (possibly an empty dict). The
    extractor never raises — a malformed annotation just produces
    fewer routes, never a crash.
    """
    args_text = _annotation_args_text(annotation_text)
    if not args_text:
        return {}

    out: dict[str, Any] = {}
    try:
        pieces = _split_top_level_commas(args_text)
        if not pieces:
            return {}

        # Single-positional-string sugar: `@GetMapping("/x")` — the
        # whole args body is one quoted string. Map to `value=`.
        if len(pieces) == 1 and "=" not in _strip_strings_for_eq_check(pieces[0]):
            sole = pieces[0].strip()
            unq = _unquote_string(sole)
            if unq is not None:
                out["value"] = unq
                return out
            arr = _parse_array_literal(sole)
            if arr is not None:
                # `@RequestMapping({"/x", "/y"})` — multi-path positional.
                # Map to `value=` as a list.
                out["value"] = [
                    _unquote_string(p) if _unquote_string(p) is not None else p
                    for p in arr
                ]
                return out
            # Bare ident / number as the only positional — shove it
            # into `value` verbatim. Useful so `@GetMapping(SOME_CONST)`
            # at least surfaces something for debug.
            out["value"] = sole
            return out

        # Named-arg form: each piece is `key = value`.
        for piece in pieces:
            eq = _find_top_level_eq(piece)
            if eq < 0:
                # Mixed positional + named is not strictly legal in
                # Spring's annotations, but handle gracefully by treating
                # the bare token as `value=`.
                if "value" not in out:
                    unq = _unquote_string(piece.strip())
                    out["value"] = unq if unq is not None else piece.strip()
                continue
            key = piece[:eq].strip()
            value_text = piece[eq + 1 :].strip()
            if not key:
                continue
            unq = _unquote_string(value_text)
            if unq is not None:
                out[key] = unq
                continue
            arr = _parse_array_literal(value_text)
            if arr is not None:
                # Each array entry: try unquoting; otherwise keep raw.
                out[key] = [
                    _unquote_string(e) if _unquote_string(e) is not None else e
                    for e in arr
                ]
                continue
            # Verbatim token (ident / enum ref / numeric).
            out[key] = value_text
    except Exception as exc:  # pragma: no cover — defensive guard.
        _LOG.warning(
            "routes_java: failed to parse annotation args %r (%s); returning %r",
            annotation_text, exc, out,
        )
    return out


def _strip_strings_for_eq_check(text: str) -> str:
    """Return `text` with quoted-string contents replaced by `_`s so an
    `=` sign INSIDE a string doesn't trick the named-arg detector.

    Only used by `parse_annotation_args` for the
    "is this single piece a key=value or a bare positional?" check.
    """
    out_chars: list[str] = []
    in_string = False
    quote = ""
    for ch in text:
        if in_string:
            if ch == "\\":
                out_chars.append("_")
                continue
            if ch == quote:
                in_string = False
                out_chars.append(ch)
                continue
            out_chars.append("_")
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
        out_chars.append(ch)
    return "".join(out_chars)


def _find_top_level_eq(text: str) -> int:
    """Return the index of the first `=` in `text` that is NOT inside
    a quoted string, brace, or paren. Returns -1 if none found.
    """
    depth = 0
    in_string = False
    quote = ""
    for i, ch in enumerate(text):
        if in_string:
            if ch == "\\":
                continue
            if ch == quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
            continue
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return i
    return -1


def _normalise_method_token(token: str) -> str:
    """Normalise a Spring `method=` token to an uppercase HTTP method.

    Accepts both fully-qualified and bare forms — Spring lets you write
    `RequestMethod.GET` (the canonical form) or just `GET` if you've
    statically imported it.

    `'RequestMethod.GET'`  → `'GET'`
    `'GET'`                → `'GET'`
    `'org.springframework.web.bind.annotation.RequestMethod.POST'`
                           → `'POST'`
    `'"GET"'`              → `'GET'` (defensive — string form is
                                       unusual but harmless)
    """
    s = token.strip()
    unq = _unquote_string(s)
    if unq is not None:
        s = unq
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.upper()


def _extract_class_prefix(class_annotations: tuple[str, ...]) -> str:
    """Return the path prefix declared by a class-level
    `@RequestMapping(...)` annotation, or `""` if absent.

    Multi-value `@RequestMapping({"/a", "/b"})` is rare on classes;
    we take the first entry deterministically.
    """
    for ann in class_annotations:
        if _annotation_name(ann) != SPRING_REQUEST_MAPPING_NAME:
            continue
        parsed = parse_annotation_args(ann)
        value = parsed.get("value")
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first
        # path= is an alias for value= in Spring's annotations.
        path = parsed.get("path")
        if isinstance(path, str):
            return path
        if isinstance(path, list) and path:
            first = path[0]
            if isinstance(first, str):
                return first
        return ""
    return ""


def _is_spring_controller(class_annotations: tuple[str, ...]) -> bool:
    """True when the class carries `@RestController` or `@Controller`.
    Strict v1 gating — `@RequestMapping`-only classes don't count.
    """
    for ann in class_annotations:
        if _annotation_name(ann) in SPRING_CONTROLLER_MARKERS:
            return True
    return False


def _compose_path(prefix: str, suffix: str) -> str:
    """Join class-level prefix and method-level suffix into a single
    raw path (pre-normalisation). Both sides are tolerant of leading /
    trailing slashes.

    `("/api", "/users")`     → `"/api/users"`
    `("/api/", "/users")`    → `"/api/users"`  (normalize collapses //)
    `("", "/users")`         → `"/users"`
    `("/api", "")`           → `"/api"`
    `("", "")`               → `""`            → normalize_path → "/"
    """
    if not prefix and not suffix:
        return ""
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    # `normalize_path` collapses repeated slashes — safe to just concat
    # with a separator slash. Avoid trailing-slash on prefix turning into
    # `//` is also fine (collapsed downstream).
    if prefix.endswith("/") or suffix.startswith("/"):
        return prefix + suffix
    return prefix + "/" + suffix


def _method_paths_and_methods(
    method_annotation: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """For ONE method-level annotation, return the list of
    (raw_path_suffix, http_method, parsed_args) tuples it implies.

    Most annotations imply exactly one entry. The exception is the
    long form `@RequestMapping(method = {GET, POST})` which fans out.

    Returns [] if the annotation is not a routing annotation.
    """
    name = _annotation_name(method_annotation)
    parsed = parse_annotation_args(method_annotation)

    if name in SPRING_VERB_MAPPINGS:
        http_method = SPRING_VERB_MAPPINGS[name]
        return _expand_paths(parsed, http_method, parsed)

    if name == SPRING_REQUEST_MAPPING_NAME:
        # Long form. `method=` may be a single ident, a string, or a
        # `{...}` array. Default (absent `method=`) → GET in v1.
        method_arg = parsed.get("method")
        if method_arg is None:
            http_methods = ["GET"]
        elif isinstance(method_arg, list):
            http_methods = [_normalise_method_token(m) for m in method_arg]
            http_methods = [m for m in http_methods if m]
        elif isinstance(method_arg, str):
            normalised = _normalise_method_token(method_arg)
            http_methods = [normalised] if normalised else ["GET"]
        else:
            http_methods = ["GET"]
        out: list[tuple[str, str, dict[str, Any]]] = []
        for hm in http_methods:
            out.extend(_expand_paths(parsed, hm, parsed))
        return out

    return []


def _expand_paths(
    parsed: dict[str, Any],
    http_method: str,
    full_parsed: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Given the parsed args of a routing annotation and a single HTTP
    method, fan out across the value/path entries. Most annotations
    have exactly one path; `@GetMapping({"/a", "/b"})` fans to two.

    Returns [(raw_path_suffix, http_method, parsed_args), ...].
    """
    # `value=` and `path=` are aliases in Spring; check both.
    raw_paths_field = parsed.get("value")
    if raw_paths_field is None:
        raw_paths_field = parsed.get("path")

    if raw_paths_field is None:
        # No path → effectively the class prefix only. Emit one entry
        # with empty suffix; the composer + normalize_path produces the
        # right thing (`/api` from class prefix + "" from method).
        return [("", http_method, full_parsed)]

    if isinstance(raw_paths_field, str):
        return [(raw_paths_field, http_method, full_parsed)]

    if isinstance(raw_paths_field, list):
        out: list[tuple[str, str, dict[str, Any]]] = []
        for entry in raw_paths_field:
            if isinstance(entry, str):
                out.append((entry, http_method, full_parsed))
        if out:
            return out
        return [("", http_method, full_parsed)]

    # Unknown shape — treat as "path unknown", still emit one entry
    # so the route surfaces (path=class prefix only).
    return [("", http_method, full_parsed)]


def extract_routes_for_controller(
    class_annotations: tuple[str, ...],
    methods: list[tuple[str, tuple[str, ...], int]],
    file_path: str,
    repo_name: str,
    tenant_id: str,
) -> list[tuple[RouteNode, RouteEdge]]:
    """Return [(RouteNode, RouteEdge)] for every Spring-mapped method
    on a controller class.

    Args:
        class_annotations: tuple of raw annotation strings on the class.
        methods: list of (method_qn, method_annotations, line_start).
        file_path: repo-relative source path.
        repo_name, tenant_id: from the IndexBatch's repo.

    If the class is NOT a Spring controller (`@RestController` /
    `@Controller` absent), returns []. Strict v1 gating.
    """
    if not _is_spring_controller(class_annotations):
        return []

    prefix = _extract_class_prefix(class_annotations)

    out: list[tuple[RouteNode, RouteEdge]] = []
    for method_qn, method_annotations, line_start in methods:
        for ann in method_annotations:
            entries = _method_paths_and_methods(ann)
            for raw_suffix, http_method, parsed in entries:
                raw_path = _compose_path(prefix, raw_suffix)
                framework_specifics = _framework_specifics(parsed)
                out.append(_make_route(
                    repo_name=repo_name,
                    tenant_id=tenant_id,
                    raw_path=raw_path,
                    http_method=http_method,
                    handler_qn=method_qn,
                    file_path=file_path,
                    line_start=line_start,
                    framework_specifics=framework_specifics,
                ))
    return out


def _framework_specifics(parsed: dict[str, Any]) -> dict[str, Any]:
    """Carry through the cheap bits of a routing annotation's args for
    debug/inspection. Currently captured: `consumes`, `produces`,
    `headers`, `params`, `name`. The RouteNode dataclass doesn't yet
    have a framework_specifics field (per actions.py inspection); this
    helper returns the dict but the caller currently discards it.

    Kept as a separate function so when RouteNode gains the field
    (planned per the prompt's "framework_specifics" mention), wiring
    is a one-line change at `_make_route`.
    """
    out: dict[str, Any] = {}
    for key in ("consumes", "produces", "headers", "params", "name"):
        if key in parsed:
            out[key] = parsed[key]
    return out


def _make_route(
    repo_name: str,
    tenant_id: str,
    raw_path: str,
    http_method: str,
    handler_qn: str,
    file_path: str,
    line_start: int,
    framework_specifics: dict[str, Any],
) -> tuple[RouteNode, RouteEdge]:
    """Build a (RouteNode, RouteEdge) pair from extracted fields.

    `framework_specifics` is currently unused (RouteNode dataclass
    doesn't carry the field) but accepted so the call site doesn't
    have to change when it's added. See `_framework_specifics`.
    """
    del framework_specifics  # Reserved for a future RouteNode field.
    path = normalize_path(raw_path)
    return (
        RouteNode(
            repo=repo_name,
            tenant_id=tenant_id,
            path=path,
            method=http_method,
            framework="spring",
            handler_qn=handler_qn,
            file_path=file_path,
            line_start=line_start,
            raw_path=raw_path,
        ),
        RouteEdge(
            repo=repo_name,
            tenant_id=tenant_id,
            path=path,
            method=http_method,
            handler_qn=handler_qn,
        ),
    )


__all__ = [
    "SPRING_CONTROLLER_MARKERS",
    "SPRING_VERB_MAPPINGS",
    "SPRING_REQUEST_MAPPING_NAME",
    "extract_routes_for_controller",
    "parse_annotation_args",
]
