"""Friendly error catalog: maps wandb exception strings to actionable codes.

Every tool path funnels W&B exceptions through ``map_wandb_exception``, which
returns one of the five subclasses below with a human-actionable message and
a stable ``error_code`` the agent can match on.

Stability: ``error_code`` strings are SemVer-protected starting v1.0.
"""

from __future__ import annotations

from typing import Final


class WandbApiError(RuntimeError):
    """Base for every error we surface to the LLM caller."""

    error_code: str = "wandb.unknown"


class WandbAuthError(WandbApiError):
    error_code: str = "auth.bad_key"


class WandbPermissionError(WandbApiError):
    error_code: str = "permission.denied"


class WandbNotFoundError(WandbApiError):
    error_code: str = "not_found.resource"


class WandbRateLimitError(WandbApiError):
    error_code: str = "quota.rate_limit"


class WandbTransientError(WandbApiError):
    error_code: str = "transient.try_again"


_AUTH_SIGNS: Final = ("401", "unauthorized", "invalid api key", "not authenticated")
_PERM_SIGNS: Final = ("403", "forbidden", "permission denied", "access denied")
_NF_SIGNS: Final = ("404", "not found", "no such project", "no such run", "no such sweep", "no project", "no run named")
_RL_SIGNS: Final = ("429", "rate limit", "too many requests")
_TRANSIENT_SIGNS: Final = ("502", "503", "504", "timeout", "temporarily unavailable", "connection reset", "connection refused")


_FRIENDLY_MESSAGES: Final[dict[str, str]] = {
    "auth.bad_key": (
        "W&B rejected the API key. Run `wandb login`, set the WANDB_API_KEY "
        "environment variable, or use `mcp-wandb auth store` to provide one."
    ),
    "permission.denied": (
        "The W&B user attached to this API key cannot access that resource. "
        "Check the entity/team membership or whether the project is private."
    ),
    "not_found.resource": (
        "W&B reports the resource does not exist. Verify the entity/project/run-id "
        "spelling; remember that paths look like 'entity/project/run_id'."
    ),
    "quota.rate_limit": (
        "W&B throttled the request. Retry in a moment; if this is persistent, "
        "lower MCP_WANDB_RATE_LIMIT below 60 to back off harder."
    ),
    "transient.try_again": (
        "W&B returned a transient server-side error. Retry the request; if "
        "this persists, check status.wandb.ai."
    ),
}


def map_wandb_exception(exc: BaseException) -> WandbApiError:
    """Translate any exception from the wandb SDK into our catalog.

    The classification is best-effort string matching because the wandb
    SDK does not have a fully consistent exception hierarchy across
    its release history.
    """
    msg = str(exc).lower()
    cls = WandbApiError
    code: str = "wandb.unknown"

    if _any_in(msg, _AUTH_SIGNS):
        cls, code = WandbAuthError, "auth.bad_key"
    elif _any_in(msg, _PERM_SIGNS):
        cls, code = WandbPermissionError, "permission.denied"
    elif _any_in(msg, _NF_SIGNS):
        cls, code = WandbNotFoundError, "not_found.resource"
    elif _any_in(msg, _RL_SIGNS):
        cls, code = WandbRateLimitError, "quota.rate_limit"
    elif _any_in(msg, _TRANSIENT_SIGNS):
        cls, code = WandbTransientError, "transient.try_again"

    friendly = _FRIENDLY_MESSAGES.get(code)
    new = cls(f"{friendly} (original: {exc})") if friendly else cls(str(exc))
    new.error_code = code
    new.__cause__ = exc
    return new


def is_retryable(exc: BaseException) -> bool:
    """A pre-mapping shortcut used by client.py's retry decision."""
    msg = str(exc).lower()
    return _any_in(msg, _RL_SIGNS + _TRANSIENT_SIGNS)


def _any_in(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)
