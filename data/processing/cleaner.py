# data/processing/cleaner.py
"""
Dataset cleaning and deduplication utilities.

Deduplication is critical: repeated data leads to memorization
instead of generalization.
"""
import hashlib
from loguru import logger
from data.validation.schema import Conversation


def compute_hash(conv: Conversation) -> str:
    """
    Generates a unique hash based on the full conversation content.
    We use the complete text to detect exact duplicates.
    """
    content = "".join(m.content for m in conv.messages)
    return hashlib.md5(content.encode()).hexdigest()


def deduplicate(conversations: list[Conversation]) -> list[Conversation]:
    """
    Removes duplicate conversations based on a content hash.
    Keeps the first occurrence of each unique conversation.
    """
    seen_hashes: set[str] = set()
    unique = []

    for conv in conversations:
        h = compute_hash(conv)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(conv)

    removed = len(conversations) - len(unique)
    logger.info(
        f"Deduplication: {len(unique):,} unique / "
        f"{len(conversations):,} total ({removed:,} removed)"
    )
    return unique


def clean_text(text: str) -> str:
    """
    Basic text normalization.
    In production, this would be more comprehensive
    (HTML tag removal, unicode normalization, etc.).
    """
    import re

    # Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_conversations(
    conversations: list[Conversation],
) -> list[Conversation]:
    """Applies text cleaning to all messages in each conversation."""
    cleaned = []

    for conv in conversations:
        try:
            clean_messages = [
                conv.messages[i].model_copy(
                    update={"content": clean_text(conv.messages[i].content)}
                )
                for i in range(len(conv.messages))
            ]

            cleaned.append(
                conv.model_copy(update={"messages": clean_messages})
            )

        except Exception:
            continue

    return cleaned