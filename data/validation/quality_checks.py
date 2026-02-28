# data/validation/quality_checks.py
"""
Quality filters for the dataset.
In production, these thresholds are empirically tuned.
We keep them configurable to demonstrate scalability awareness.
"""
from dataclasses import dataclass
from loguru import logger
from data.validation.schema import Conversation


@dataclass
class QualityConfig:
    """Configuration for quality filters."""
    min_chars: int = 50          # Very short conversations are not useful
    max_chars: int = 8000        # Prevents exceeding model context window
    min_turns: int = 1           # At least 1 user/assistant turn
    max_turns: int = 20          # Very long conversations are rare and noisy
    min_quality_score: float = 0.5


@dataclass
class QualityReport:
    """Report describing dataset cleanliness after filtering."""
    total_samples: int = 0
    passed: int = 0
    failed_min_chars: int = 0
    failed_max_chars: int = 0
    failed_min_turns: int = 0
    failed_max_turns: int = 0
    failed_quality_score: int = 0
    failed_validation: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.passed / self.total_samples

    def summary(self) -> str:
        return (
            f"Quality Report:\n"
            f"  Total:   {self.total_samples:,}\n"
            f"  Passed:  {self.passed:,} ({self.pass_rate:.1%})\n"
            f"  Failed (too short):    {self.failed_min_chars:,}\n"
            f"  Failed (too long):     {self.failed_max_chars:,}\n"
            f"  Failed (too few turns): {self.failed_min_turns:,}\n"
            f"  Failed (too many turns): {self.failed_max_turns:,}\n"
            f"  Failed (low quality):  {self.failed_quality_score:,}\n"
            f"  Failed (schema error): {self.failed_validation:,}"
        )


def run_quality_checks(
    conversations: list[Conversation],
    config: QualityConfig | None = None,
) -> tuple[list[Conversation], QualityReport]:
    """
    Filters out conversations that do not pass quality controls.

    Returns:
        Tuple of (clean_conversations, report)
    """
    if config is None:
        config = QualityConfig()

    report = QualityReport(total_samples=len(conversations))
    clean = []

    for conv in conversations:
        # Filter 1: Minimum length
        if conv.total_chars < config.min_chars:
            report.failed_min_chars += 1
            continue

        # Filter 2: Maximum length
        if conv.total_chars > config.max_chars:
            report.failed_max_chars += 1
            continue

        # Filter 3: Number of turns
        if conv.num_turns < config.min_turns:
            report.failed_min_turns += 1
            continue

        if conv.num_turns > config.max_turns:
            report.failed_max_turns += 1
            continue

        # Filter 4: Quality score
        if conv.quality_score < config.min_quality_score:
            report.failed_quality_score += 1
            continue

        report.passed += 1
        clean.append(conv)

    logger.info(report.summary())
    return clean, report