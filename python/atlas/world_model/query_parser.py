"""Natural language query parser for spatial scene queries.

Parses human queries into structured spatial operations that can be
resolved against the semantic label store and spatial relationship graph.
All parsing is rule-based — fast, deterministic, and dependency-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class QueryType(Enum):
    """Category of a spatial query."""

    LOCATION = "location"
    """'Where is the X?' — find the position of an object."""

    PROPERTY = "property"
    """'What is X made of?' — retrieve a physical property."""

    SPATIAL_RELATION = "spatial_relation"
    """'What is on top of X?' — ask about object relationships."""

    RISK = "risk"
    """'What is the most unstable object?' — rank by risk score."""

    UNKNOWN = "unknown"
    """Fallback when no pattern matches."""


@dataclass
class ParsedQuery:
    """Structured representation of a natural language scene query.

    Args:
        raw: Original query string.
        query_type: Classified query category.
        subject: Primary object being asked about (e.g. ``"glass"``).
        predicate: Attribute to retrieve (e.g. ``"position"``, ``"material"``).
        relation: Spatial-relation keyword if applicable (e.g. ``"on_top_of"``).
        reference_object: Secondary object for spatial-relation queries.

    Example::

        parser = QueryParser()
        q = parser.parse("Where is the glass?")
        assert q.query_type == QueryType.LOCATION
        assert q.subject == "glass"
    """

    raw: str
    query_type: QueryType
    subject: str
    predicate: str
    relation: str = ""
    reference_object: str = ""


# ── Compiled pattern sets ─────────────────────────────────────────────────────

_RISK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bmost\s+(unstable|dangerous|risky|likely\s+to\s+fall|fragile)\b"),
    re.compile(r"\b(unstable|dangerous|risky|at\s+risk)\s+object\b"),
    re.compile(r"\brisks?\b"),
    re.compile(r"\bhazards?\b"),
]

_LOCATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhere\s+is\b"),
    re.compile(r"\blocation\s+of\b"),
    re.compile(r"\bfind\b"),
    re.compile(r"\bposition\s+of\b"),
]

_PROPERTY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+is\s+.+\s+made\s+of\b"),
    re.compile(r"\bmaterial\s+of\b"),
    re.compile(r"\bweight\s+of\b"),
    re.compile(r"\bhow\s+(heavy|fragile|slippery)\b"),
    re.compile(r"\bmass\s+of\b"),
]

_SPATIAL_RELATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+is\s+(on\s+top\s+of|above|below|inside|next\s+to|near|adjacent\s+to)\b"),
    re.compile(r"\bon\s+top\s+of\b"),
    re.compile(r"\bsupport(ing|s)\b"),
    re.compile(r"\bleaning\b"),
]

# Ordered longest-first so "on top of" matches before "on".
_RELATION_KEYWORDS: list[tuple[str, str]] = sorted(
    [
        ("on top of", "on_top_of"),
        ("above", "on_top_of"),
        ("below", "below"),
        ("inside", "inside"),
        ("in", "inside"),
        ("next to", "adjacent_to"),
        ("near", "adjacent_to"),
        ("adjacent to", "adjacent_to"),
        ("leaning", "leaning"),
        ("supporting", "supporting"),
        ("on", "on_top_of"),
    ],
    key=lambda pair: -len(pair[0]),
)

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?!.,;]+$")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip(text: str) -> str:
    """Strip trailing punctuation and lower-case *text*."""
    return _TRAILING_PUNCT_RE.sub("", text.strip().lower())


def _extract_object_name(query: str) -> str:
    """Heuristically extract the primary object name from *query*.

    Tries several common question patterns before falling back to the
    full cleaned query string.

    Args:
        query: Raw natural language query.

    Returns:
        Best-guess object name, lowercased. Empty string if none found.
    """
    cleaned = _strip(query)

    # "where is [the] OBJECT" / "find [the] OBJECT" / "position of [the] OBJECT"
    m = re.search(
        r"(?:where\s+is|find|location\s+of|position\s+of)\s+(?:the\s+)?(.+)$",
        cleaned,
    )
    if m:
        return m.group(1).strip()

    # "what is [the] OBJECT made of"
    m = re.search(r"what\s+is\s+(?:the\s+)?(.+?)\s+made\s+of", cleaned)
    if m:
        return m.group(1).strip()

    # "what is [the] OBJECT's PROPERTY"
    m = re.search(r"what\s+is\s+(?:the\s+)?(.+?)(?:'s|\s+of\b)", cleaned)
    if m:
        return m.group(1).strip()

    # "… on top of [the] OBJECT"
    for phrase, _ in _RELATION_KEYWORDS:
        if phrase in cleaned:
            idx = cleaned.find(phrase) + len(phrase)
            remainder = _ARTICLE_RE.sub("", cleaned[idx:].strip())
            if remainder:
                return remainder
            break

    return ""


def _extract_relation(query: str) -> tuple[str, str]:
    """Extract the relation keyword and reference object from *query*.

    Args:
        query: Raw natural language query.

    Returns:
        ``(relation_type, reference_object)`` — both empty strings when no
        relation phrase is found.
    """
    lower = query.lower()
    for phrase, rel_type in _RELATION_KEYWORDS:
        if phrase in lower:
            idx = lower.find(phrase) + len(phrase)
            remainder = _ARTICLE_RE.sub("", lower[idx:].strip())
            remainder = _TRAILING_PUNCT_RE.sub("", remainder)
            return rel_type, remainder
    return "", ""


def _property_predicate(query: str) -> str:
    """Map property-related keywords to attribute names.

    Args:
        query: Lowercased query string.

    Returns:
        Attribute name to look up (e.g. ``"mass_kg"``).
    """
    if re.search(r"\b(heavy|mass|weight)\b", query):
        return "mass_kg"
    if re.search(r"\bfragile\b", query):
        return "fragility"
    if re.search(r"\b(friction|slippery|grippy)\b", query):
        return "friction"
    return "material"


# ── Public API ────────────────────────────────────────────────────────────────


class QueryParser:
    """Parses natural language spatial queries into structured operations.

    Uses rule-based pattern matching — no external ML dependencies.
    Priority order: RISK > SPATIAL_RELATION > PROPERTY > LOCATION > UNKNOWN.

    Example::

        parser = QueryParser()

        q1 = parser.parse("Where is the glass?")
        assert q1.query_type == QueryType.LOCATION
        assert q1.subject == "glass"

        q2 = parser.parse("What is the most unstable object?")
        assert q2.query_type == QueryType.RISK

        q3 = parser.parse("What is the laptop made of?")
        assert q3.query_type == QueryType.PROPERTY
        assert q3.subject == "laptop"
        assert q3.predicate == "material"
    """

    def parse(self, query: str) -> ParsedQuery:
        """Parse *query* into a :class:`ParsedQuery`.

        Args:
            query: Natural language query string.

        Returns:
            Structured :class:`ParsedQuery` with classified type and extracted fields.
        """
        lower = query.lower()

        # 1. Risk — highest priority
        if any(p.search(lower) for p in _RISK_PATTERNS):
            logger.debug("query_parsed", query_type="risk", raw=query)
            return ParsedQuery(
                raw=query,
                query_type=QueryType.RISK,
                subject="",
                predicate="risk_rank",
            )

        # 2. Spatial relation
        if any(p.search(lower) for p in _SPATIAL_RELATION_PATTERNS):
            relation, ref_obj = _extract_relation(query)
            subject = _extract_object_name(query)
            logger.debug(
                "query_parsed",
                query_type="spatial_relation",
                relation=relation,
                reference=ref_obj,
                raw=query,
            )
            return ParsedQuery(
                raw=query,
                query_type=QueryType.SPATIAL_RELATION,
                subject=subject,
                predicate="spatial_relation",
                relation=relation,
                reference_object=ref_obj,
            )

        # 3. Property
        if any(p.search(lower) for p in _PROPERTY_PATTERNS):
            subject = _extract_object_name(query)
            predicate = _property_predicate(lower)
            logger.debug(
                "query_parsed",
                query_type="property",
                predicate=predicate,
                raw=query,
            )
            return ParsedQuery(
                raw=query,
                query_type=QueryType.PROPERTY,
                subject=subject,
                predicate=predicate,
            )

        # 4. Location
        if any(p.search(lower) for p in _LOCATION_PATTERNS):
            subject = _extract_object_name(query)
            logger.debug("query_parsed", query_type="location", subject=subject, raw=query)
            return ParsedQuery(
                raw=query,
                query_type=QueryType.LOCATION,
                subject=subject,
                predicate="position",
            )

        # 5. Fallback
        logger.debug("query_parsed", query_type="unknown", raw=query)
        return ParsedQuery(
            raw=query,
            query_type=QueryType.UNKNOWN,
            subject=_strip(query),
            predicate="unknown",
        )
