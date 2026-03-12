# src/athena/jurisdiction/__init__.py
"""Jurisdiction registry — multi-jurisdiction support for ATHENA."""

from athena.jurisdiction.registry import (
    JurisdictionConfig,
    register_jurisdiction,
    get_jurisdiction,
    get_jurisdiction_for_case,
    list_jurisdictions,
)

# Import jurisdiction modules to trigger registration
import athena.jurisdiction.it  # noqa: F401
import athena.jurisdiction.ch  # noqa: F401

__all__ = [
    "JurisdictionConfig",
    "register_jurisdiction",
    "get_jurisdiction",
    "get_jurisdiction_for_case",
    "list_jurisdictions",
]
