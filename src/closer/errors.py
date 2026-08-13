"""Explicit failure types. Core stages must raise these instead of silently degrading."""

from __future__ import annotations


class CloserError(Exception):
    """Base class for all CLOSER failures."""


class BenchmarkParseError(CloserError):
    """ProteinGym-LLM prompt could not be parsed without guessing."""


class MutationCompilationError(CloserError):
    """WT/mutant sequence diff failed or failed reconstruction."""


class AssayInterpretationError(CloserError):
    """Assay contract could not be produced from a real LLM call."""


class EvidenceAnalysisError(CloserError):
    """Mechanistic evidence analysis failed for a variant batch."""


class ComparisonError(CloserError):
    """Comparative reasoning failed."""


class GraphConnectivityError(CloserError):
    """Preference graph could not be made weakly connected."""


class RankingSolverError(CloserError):
    """Global rank solver failed."""


class AuditError(CloserError):
    """Adaptive auditor failed."""


class BudgetExceededError(CloserError):
    """Token or call budget was exceeded and hard_stop_on_exceed is set."""


class FinalRankingValidationError(CloserError):
    """Final ranking is incomplete, duplicated, or contains unknown ids."""


class ProviderError(CloserError):
    """Inner LLM provider call failed after retries."""


class StructuredOutputError(CloserError):
    """Structured output could not be parsed or repaired."""


class ConfigError(CloserError):
    """Invalid configuration, including recursive CLOSER_BASE_URL."""
