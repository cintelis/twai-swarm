import types

from app.api import _salvage_files_from_truncated

# We import dynamically to ensure the helpers exist even if unexported
from app import api as api_mod


def test_validate_branch_name_accepts_common_names():
    assert api_mod._validate_branch_name("feature/x")
    assert api_mod._validate_branch_name("v1.2.3")
    assert api_mod._validate_branch_name("hotfix-01")
    assert api_mod._validate_branch_name("foo/bar/baz")


def test_validate_branch_name_rejects_bad_names():
    bad = ["..", "foo//bar", "/bad", "bad/", "a b", "\tctrl", "name~", "", "a" * 201]
    for name in bad:
        assert not api_mod._validate_branch_name(name)


def test_approval_awaiting_true_false_and_safety():
    tasks = [
        {"role": "architect", "status": "done"},
        {"role": "ba", "status": "done"},
    ]
    # Awaiting when running, architect done, no SE, and neither approved nor rejected
    assert api_mod._approval_awaiting("RUNNING", tasks, {"approved": False, "rejected": False}) is True
    # Not awaiting when approved
    assert api_mod._approval_awaiting("RUNNING", tasks, {"approved": True, "rejected": False}) is False
    # Not awaiting when rejected
    assert api_mod._approval_awaiting("RUNNING", tasks, {"approved": False, "rejected": True}) is False
    # Not awaiting when SE present
    tasks_with_se = tasks + [{"role": "se", "status": "queued"}]
    assert api_mod._approval_awaiting("RUNNING", tasks_with_se, {"approved": False, "rejected": False}) is False
    # Non-dict approval_state should not blow up and should be treated as not approved/rejected
    assert api_mod._approval_awaiting("RUNNING", tasks, None) is True
    class Obj:
        def __init__(self):
            self.approved = False
            self.rejected = False
    assert api_mod._approval_awaiting("RUNNING", tasks, Obj()) is True


def test_salvage_files_from_truncated_extracts_complete_pairs():
    # Construct a truncated JSON-like string with two complete file objects and one partial
    raw = (
        '{"files": [\n'
        '{"path": "a/b.txt", "content": "one"},\n'
        '{"path": "c.txt", "content": "two"},\n'
        '{"path": "d.txt", "content": "three"'  # truncated before closing braces
        ']\n}'
    )
    files = _salvage_files_from_truncated(raw)
    paths = [f.get("path") for f in files]
    assert "a/b.txt" in paths and "c.txt" in paths
    # The truncated third should not be included
    assert "d.txt" not in paths
