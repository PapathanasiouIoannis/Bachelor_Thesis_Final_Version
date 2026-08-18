"""Canonicalization and validation shared by experiment configuration parsers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, is_dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


_EXPERIMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or ambiguous."""


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ConfigurationError("Configuration numbers must be finite.")
    if value.is_zero():
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        # JSON has no decimal type. A canonical string retains the exact value
        # and produces the same hash for 60, 60.0, and 6e1.
        return _decimal_text(value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError("Configuration numbers must be finite.")
    return value


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest of configuration-like data."""

    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decimal_amplitude_grid(
    start: Decimal,
    stop: Decimal,
    step: Decimal,
) -> tuple[Decimal, ...]:
    """Return the exact inclusive amplitude grid ``start, ..., stop``.

    The endpoint must lie exactly on the requested step and the undeformed
    control, ``A = 0``, must be present. No endpoint or near-zero replacement
    is performed.
    """

    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ConfigurationError(f"Amplitude {name} must be a finite decimal.")
    if start >= stop:
        raise ConfigurationError("Amplitude start must be smaller than amplitude stop.")
    if step <= 0:
        raise ConfigurationError("Amplitude step must be strictly positive.")

    quotient = (stop - start) / step
    if quotient != quotient.to_integral_value():
        raise ConfigurationError(
            "Amplitude stop is not exactly aligned with amplitude start and step; "
            "choose values whose interval is an integer number of steps."
        )
    intervals = int(quotient)
    if intervals > 100_000:
        raise ConfigurationError(
            "Amplitude grid exceeds 100,001 values; choose a larger step."
        )
    values = tuple(start + step * index for index in range(intervals + 1))
    if Decimal("0") not in values:
        raise ConfigurationError(
            "Amplitude grid must contain the undeformed control A = 0 exactly."
        )
    return values


def _runtimeize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {key: _runtimeize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_runtimeize(item) for item in value]
    return value


def _require_table(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be a TOML table.")
    return value


def _require_exact_keys(
    table: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(table)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    messages = []
    if missing:
        messages.append(f"missing fields {missing}")
    if unknown:
        messages.append(f"unknown fields {unknown}")
    if messages:
        raise ConfigurationError(f"{context}: " + "; ".join(messages) + ".")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be a non-empty string.")
    return value.strip()


def _integer(value: Any, context: str) -> int:
    if type(value) is not int:  # bool must not be accepted as an integer
        raise ConfigurationError(f"{context} must be an integer.")
    return value


def _boolean(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ConfigurationError(f"{context} must be true or false.")
    return value


def _decimal(value: Any, context: str) -> Decimal:
    if isinstance(value, bool):
        raise ConfigurationError(f"{context} must be a finite number.")
    if isinstance(value, Decimal):
        result = value
    elif type(value) is int:
        result = Decimal(value)
    else:
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ConfigurationError(f"{context} must be a finite number.") from None
    if not result.is_finite():
        raise ConfigurationError(f"{context} must be a finite number.")
    return result


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{context} must be a non-empty TOML array.")
    result = tuple(_string(item, f"{context} item") for item in value)
    if len(set(result)) != len(result):
        raise ConfigurationError(f"{context} must not contain duplicate names.")
    return result


def _validate_common_header(root: Mapping[str, Any]) -> tuple[int, str, str, str]:
    schema_version = _integer(root["schema_version"], "schema_version")
    if schema_version != 1:
        raise ConfigurationError(
            f"schema_version = {schema_version} is unsupported; use schema_version = 1."
        )
    experiment_name = _string(root["experiment_name"], "experiment_name")
    if not _EXPERIMENT_NAME.fullmatch(experiment_name):
        raise ConfigurationError(
            "experiment_name must contain 3-64 lower-case letters, digits, hyphens, "
            "or underscores, and must start with a letter or digit."
        )
    workflow = _string(root["workflow"], "workflow")
    mode = _string(root["mode"], "mode")
    return schema_version, experiment_name, workflow, mode
