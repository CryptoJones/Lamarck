# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aaron K. Clark
"""Test helper: build a minimum-valid JSON document for a given schema.

Used by Tier 3 integration tests to drive perfect-inference mocks
against the real on-disk 100-problem held-out set. The helper
walks the schema's ``required`` keys + their typed entries in
``properties`` and emits the cheapest valid value per type.

Honours the simple patterns the L14 generator actually emits
(``^DEV-\\d{4}$``, ``^P-\\d+$``, ``format=email``). For everything
else it emits the lowest-cost-passing value: empty(ish) string of
the right minLength, integer at the declared minimum, etc.
"""

from __future__ import annotations

import json
from typing import Any


def minimal_for(spec: dict[str, Any]) -> Any:
    t = spec.get("type")
    if t == "object":
        req = spec.get("required", [])
        props = spec.get("properties", {})
        return {k: minimal_for(props.get(k, {"type": "string"})) for k in req}
    if t == "array":
        inner = spec.get("items", {"type": "string"})
        min_n = spec.get("minItems", 0) or 1
        return [minimal_for(inner) for _ in range(min_n)]
    if t == "string":
        if spec.get("format") == "email":
            return "user@example.com"
        pattern = spec.get("pattern", "")
        if pattern == r"^DEV-\d{4}$":
            return "DEV-0001"
        if pattern == r"^P-\d+$":
            return "P-1"
        return "x" * max(spec.get("minLength", 1), 1)
    if t == "integer":
        return int(max(spec.get("minimum", 0), 0))
    if t == "number":
        return 0.0 if spec.get("minimum", -1) <= 0 else float(spec["minimum"])
    if t == "boolean":
        return True
    if "enum" in spec and spec["enum"]:
        return spec["enum"][0]
    return None


def minimal_valid_json(schema: dict[str, Any]) -> str:
    """Return a JSON string that minimally satisfies ``schema``."""
    return json.dumps(minimal_for(schema))
