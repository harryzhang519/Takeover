"""JSON Schema validation with zero required dependencies.

Uses the ``jsonschema`` package (Draft7Validator) when it is installed.
Otherwise falls back to a minimal built-in draft-07 validator that covers
every keyword used in ``schemas/market_scan_schema.json``: type (including
type lists), required, properties, additionalProperties (bool), items,
enum, minimum, maximum, minLength, pattern and local ``$ref``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def load_schema(path: str | Path) -> dict:
    """Load a JSON schema from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(instance: Any, schema: dict) -> tuple[list[str], str]:
    """Validate ``instance`` against ``schema``.

    Returns:
        (errors, backend): a list of human-readable error strings (empty when
        valid) and the name of the validator backend that was used.
    """
    try:
        from jsonschema import Draft7Validator  # type: ignore
    except ImportError:
        errors: list[str] = []
        _validate_node(instance, schema, schema, "$", errors)
        return errors, "builtin-draft07"

    validator = Draft7Validator(schema)
    errors = [
        f"{'$' + ''.join(f'[{p!r}]' for p in e.absolute_path)}: {e.message}"
        for e in validator.iter_errors(instance)
    ]
    return errors, "jsonschema"


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local $ref supported, got {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _validate_node(value: Any, schema: dict, root: dict, path: str, errors: list[str]) -> None:
    if "$ref" in schema:
        schema = _resolve(schema["$ref"], root)

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS[t](value) for t in types):
            errors.append(f"{path}: expected type {types}, got {type(value).__name__}")
            return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")

    if _TYPE_CHECKS["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property '{key}'")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: additional property '{key}' not allowed")
        for key, sub in props.items():
            if key in value:
                _validate_node(value[key], sub, root, f"{path}.{key}", errors)

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _validate_node(item, schema["items"], root, f"{path}[{i}]", errors)
