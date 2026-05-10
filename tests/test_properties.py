"""Hypothesis property tests for stable invariants.

Skipped if ``hypothesis`` isn't installed (CI installs it; the minimal local
venv may not). Each property is intentionally small and shrinkable.
"""

from __future__ import annotations

import math
import random

import pytest

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
from hypothesis import HealthCheck, given, settings  # noqa: E402

from mcp_wandb import _cursor  # noqa: E402
from mcp_wandb._util import flatten_config, parse_since  # noqa: E402
from mcp_wandb.importance import compute_importance  # noqa: E402
from mcp_wandb.tools.analysis import _has_distinct  # noqa: E402

# ---------------------------------------------------------------------------
# parse_since
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=10000),
    unit=st.sampled_from(["s", "m", "h", "d", "w"]),
)
def test_parse_since_relative_never_raises(n: int, unit: str) -> None:
    parsed = parse_since(f"{n}{unit}")
    assert parsed is not None


# ---------------------------------------------------------------------------
# flatten_config
# ---------------------------------------------------------------------------


@st.composite
def _nested_config(draw):  # type: ignore[no-untyped-def]
    leaf = st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(min_size=0, max_size=5), st.booleans())
    return draw(
        st.dictionaries(
            keys=st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=4),
            values=st.recursive(leaf, lambda children: st.dictionaries(keys=st.text(alphabet="abcd", min_size=1, max_size=3), values=children, max_size=3), max_leaves=10),
            max_size=4,
        )
    )


@given(cfg=_nested_config())
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
def test_flatten_config_keys_no_underscore_prefix(cfg) -> None:  # type: ignore[no-untyped-def]
    flat = flatten_config(cfg)
    for key in flat:
        # No segment should start with underscore (those are W&B internals).
        for segment in key.split("."):
            assert not segment.startswith("_") or segment == ""


# ---------------------------------------------------------------------------
# cursor round-trip
# ---------------------------------------------------------------------------


@given(offset=st.integers(min_value=0, max_value=10**6), seed=st.integers(min_value=0, max_value=1000))
def test_cursor_round_trip(offset: int, seed: int) -> None:
    rng = random.Random(seed)
    query = {f"k{i}": rng.random() for i in range(rng.randint(1, 5))}
    cursor = _cursor.encode(offset=offset, query=query)
    assert _cursor.decode(cursor, query) == offset


# ---------------------------------------------------------------------------
# _has_distinct
# ---------------------------------------------------------------------------


@given(value=st.one_of(st.integers(), st.text(), st.floats(allow_nan=False)))
def test_has_distinct_on_identical_singleton_is_false(value) -> None:  # type: ignore[no-untyped-def]
    assert _has_distinct([value, value, value]) is False


def test_has_distinct_treats_nan_as_equal() -> None:
    # Hand-rolled because hypothesis can't easily build NaN-only lists.
    assert _has_distinct([math.nan, math.nan]) is False


# ---------------------------------------------------------------------------
# compute_importance shape invariants
# ---------------------------------------------------------------------------


@st.composite
def _importance_rows(draw):  # type: ignore[no-untyped-def]
    n = draw(st.integers(min_value=4, max_value=40))
    rng = random.Random(draw(st.integers(min_value=0, max_value=1000)))
    return [
        {
            "config_flat": {
                "lr": rng.uniform(1e-5, 1e-1),
                "bs": rng.choice([16, 32, 64, 128]),
            },
            "metric": rng.gauss(0.5, 0.1),
        }
        for _ in range(n)
    ]


@given(rows=_importance_rows())
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture], max_examples=10, deadline=10000)
def test_compute_importance_invariants(rows) -> None:  # type: ignore[no-untyped-def]
    top_k = 5
    result = compute_importance(rows, target_metric="val", method="rf", top_k=top_k)
    assert result.n_runs == len(rows)
    assert len(result.ranking) <= top_k
    assert all(0.0 <= e.importance <= 1.0 + 1e-6 for e in result.ranking)
    # OOB R² is bounded above by 1.0 but can be arbitrarily negative when
    # the model is worse than the mean predictor (hypothesis can construct
    # such cases trivially with few samples). The upper bound is the actual
    # mathematical invariant.
    assert result.model_r2 <= 1.0 + 1e-6
