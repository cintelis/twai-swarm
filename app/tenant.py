"""
Tenant identity — lightweight scaffolding for the multi-tenant future.

Today every request runs under `DEFAULT_TENANT_ID = "default"`. When the
auth middleware lands, JWT claims will resolve to a real tenant_id and
this module's helpers will validate/normalise it. Until then, the value
is always "default" and every function just takes it as a parameter with
a default value.

The point of this module is to give us ONE canonical constant + validation
rule, not to build auth. That comes later.

Valid tenant_id format (matches GitHub org-name rules so one App install
per-tenant on GitHub maps 1:1 to a tenant):
  - Lowercase alphanumerics + hyphens + underscores
  - 1-60 characters
  - Doesn't start or end with a hyphen
  - Not the reserved words 'public', 'pg_*', 'information_schema'
    (reserved for when schema-per-tenant lands)
"""
from __future__ import annotations

import re

DEFAULT_TENANT_ID = "default"

# Reserved names that mustn't be used as tenant_id because they collide
# with Postgres schema names or pg internals.
_RESERVED_TENANT_IDS = {"public", "information_schema", "pg_catalog", "pg_toast"}

_TENANT_ID_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9_-]{0,58}[a-z0-9])?$")


class InvalidTenantIdError(ValueError):
    """Raised when a tenant_id fails validation."""


def validate_tenant_id(tenant_id: str) -> str:
    """Check tenant_id is well-formed. Returns the id on success.

    Used at the trust boundary — wherever tenant_id enters the system
    (API, workflow start, CLI tools). Callers that receive tenant_id
    from already-trusted internal code (activities, agent functions)
    don't need to re-validate.
    """
    if not isinstance(tenant_id, str):
        raise InvalidTenantIdError(f"tenant_id must be a string, got {type(tenant_id).__name__}")
    if tenant_id in _RESERVED_TENANT_IDS:
        raise InvalidTenantIdError(f"tenant_id {tenant_id!r} is reserved")
    if tenant_id.startswith("pg_"):
        raise InvalidTenantIdError(f"tenant_id {tenant_id!r} starts with reserved prefix 'pg_'")
    if not _TENANT_ID_PATTERN.match(tenant_id):
        raise InvalidTenantIdError(
            f"tenant_id {tenant_id!r} must be 1-60 lowercase alphanumerics, hyphens, or underscores "
            f"(not starting/ending with hyphen or underscore)"
        )
    return tenant_id
