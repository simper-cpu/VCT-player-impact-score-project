"""Agent-to-role mapping used by both dataset construction and inference.

The mapping is deliberately conservative.  A role inferred from an agent is
an input signal, not ground truth; callers should keep ``role_confidence`` and
allow ``Flex/Unknown`` for new or ambiguous agents.
"""

from __future__ import annotations

import re

import pandas as pd


ROLES = ("Duelist", "Initiator", "Controller", "Sentinel", "Flex/Unknown")

_AGENT_ROLE_MAP = {
    "jett": ("Duelist", 0.98), "raze": ("Duelist", 0.98),
    "neon": ("Duelist", 0.98), "phoenix": ("Duelist", 0.98),
    "reyna": ("Duelist", 0.98), "yoru": ("Duelist", 0.98),
    "iso": ("Duelist", 0.98), "waylay": ("Duelist", 0.95),
    "sova": ("Initiator", 0.98), "breach": ("Initiator", 0.98),
    "skye": ("Initiator", 0.98), "kayo": ("Initiator", 0.98),
    "fade": ("Initiator", 0.98), "gekko": ("Initiator", 0.98),
    "tejo": ("Initiator", 0.95),
    "brimstone": ("Controller", 0.98), "omen": ("Controller", 0.98),
    "viper": ("Controller", 0.98), "astra": ("Controller", 0.98),
    "harbor": ("Controller", 0.95), "clove": ("Controller", 0.95),
    "sage": ("Sentinel", 0.98), "cypher": ("Sentinel", 0.98),
    "killjoy": ("Sentinel", 0.98), "chamber": ("Sentinel", 0.95),
    "deadlock": ("Sentinel", 0.95), "vyse": ("Sentinel", 0.95),
}


def normalize_agent(agent: object) -> str:
    value = "" if agent is None else str(agent).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def infer_role(agent: object) -> tuple[str, float]:
    """Return ``(role, confidence)`` without pretending unknown is certain."""
    normalized = normalize_agent(agent)
    return _AGENT_ROLE_MAP.get(normalized, ("Flex/Unknown", 0.0))


def map_agent_to_role(agent: object) -> str:
    """Convenience API for callers that need only the categorical role."""
    return infer_role(agent)[0]


def add_role_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add role and confidence while preserving explicit user-provided roles."""
    result = frame.copy()
    inferred = result.get("agent", pd.Series("", index=result.index)).map(infer_role)
    inferred_role = inferred.map(lambda item: item[0])
    inferred_confidence = inferred.map(lambda item: item[1]).astype(float)

    if "role" not in result:
        result["role"] = inferred_role
    else:
        explicit = result["role"].astype("string").str.strip()
        result["role"] = explicit.where(explicit.notna() & explicit.ne(""), inferred_role)
    if "role_confidence" not in result:
        result["role_confidence"] = inferred_confidence
    else:
        explicit_confidence = pd.to_numeric(result["role_confidence"], errors="coerce")
        result["role_confidence"] = explicit_confidence.fillna(inferred_confidence)
    result["role"] = result["role"].fillna("Flex/Unknown").astype(str)
    result["role_confidence"] = result["role_confidence"].clip(0.0, 1.0)
    return result


AGENT_ROLE_MAP = dict(_AGENT_ROLE_MAP)
AGENT_TO_ROLE = {agent: role for agent, (role, _) in _AGENT_ROLE_MAP.items()}
