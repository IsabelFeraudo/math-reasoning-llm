# data/ingestion/sources.py
"""
Data source configuration.
Separating configuration from logic is a key design principle
that demonstrates experience in scalable production systems.
"""
from dataclasses import dataclass, field


@dataclass
class DataSource:
    """Configuration for a dataset hosted on the Hugging Face Hub."""
    name: str                          # Dataset name on HF Hub
    split: str = "train"               # Split to download
    max_samples: int | None = None     # None = load all data
    subset: str | None = None          # Dataset subset (if applicable)
    trust_remote_code: bool = False    # Security: set True only if you trust the source


# Datasets used in this project
SOURCES: dict[str, DataSource] = {
    # High-quality general instruction tuning
    "openhermes": DataSource(
        name="teknium/OpenHermes-2.5",
        split="train",
        max_samples=50_000,   # Limit to 50k to avoid exhausting Colab resources
    ),
    # Mathematical reasoning — foundation for later RL training
    "gsm8k": DataSource(
        name="openai/gsm8k",
        subset="main",
        split="train",
        max_samples=None,     # GSM8K has ~7.5k samples; we use all of them
    ),
    # Small dataset for quick testing
    "gsm8k_test": DataSource(
        name="openai/gsm8k",
        subset="main",
        split="test",
        max_samples=500,
    ),
}