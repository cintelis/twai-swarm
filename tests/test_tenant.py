"""Tenant ID validation rules."""
from __future__ import annotations

import pytest

from app import tenant


def test_default_tenant_id_constant():
    assert tenant.DEFAULT_TENANT_ID == "default"
    # Default must itself pass validation.
    assert tenant.validate_tenant_id(tenant.DEFAULT_TENANT_ID) == "default"


@pytest.mark.parametrize("valid_id", [
    "default",
    "acme",
    "acme-corp",
    "acme_corp",
    "tenant-123",
    "a",                                # single char
    "a" * 60,                           # max length
    "365soft-labs",                     # starts with digit (valid)
])
def test_valid_tenant_ids(valid_id):
    assert tenant.validate_tenant_id(valid_id) == valid_id


@pytest.mark.parametrize("invalid_id,reason_keyword", [
    ("Acme", "lowercase"),              # uppercase not allowed
    ("a" * 61, "lowercase"),            # too long (61 chars)
    ("-acme", "lowercase"),             # leading hyphen
    ("acme-", "lowercase"),             # trailing hyphen
    ("acme corp", "lowercase"),         # space
    ("acme/corp", "lowercase"),         # slash
    ("public", "reserved"),             # reserved
    ("information_schema", "reserved"), # reserved
    ("pg_catalog", "reserved"),         # reserved
    ("pg_anything", "pg_"),             # pg_ prefix
    ("", "lowercase"),                  # empty
])
def test_invalid_tenant_ids_raise(invalid_id, reason_keyword):
    with pytest.raises(tenant.InvalidTenantIdError):
        tenant.validate_tenant_id(invalid_id)


def test_non_string_raises():
    with pytest.raises(tenant.InvalidTenantIdError, match="string"):
        tenant.validate_tenant_id(123)
    with pytest.raises(tenant.InvalidTenantIdError, match="string"):
        tenant.validate_tenant_id(None)
