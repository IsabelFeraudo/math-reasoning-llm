# data/processing/formatter.py
"""
Converts different datasets into the standard Llama-3 chat format.

Each dataset has its own schema. This module acts as the "adapter"
that unifies them. This is the most critical part of the ETL pipeline,
as formatting quality directly impacts model performance.

Target format (ChatML):
[
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]
"""
from datasets import Dataset
from loguru import logger
from data.validation.schema import Conversation, Message, Role

# Default system prompt — in production this should be configurable
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and thoughtful assistant. "
    "When solving math problems, show your reasoning step by step "
    "before giving the final answer."
)


def format_openhermes(dataset: Dataset) -> list[Conversation]:
    """
    Converts OpenHermes-2.5 into the standard conversation format.

    OpenHermes contains a 'conversations' field with objects {from, value}
    where 'from' can be: 'system', 'human', 'gpt'.
    """
    conversations = []
    skipped = 0

    # Map OpenHermes roles to our internal schema
    role_map = {
        "system": Role.SYSTEM,
        "human": Role.USER,
        "gpt": Role.ASSISTANT,
    }

    for sample in dataset:
        try:
            raw_messages = sample.get("conversations", [])
            messages = []

            for msg in raw_messages:
                raw_role = msg.get("from", "").lower()
                content = msg.get("value", "")

                if raw_role not in role_map:
                    # Unknown role: skip this message
                    continue

                messages.append(Message(
                    role=role_map[raw_role],
                    content=content,
                ))

            # If no system prompt exists, insert the default one at the beginning
            if messages and messages[0].role != Role.SYSTEM:
                messages.insert(0, Message(
                    role=Role.SYSTEM,
                    content=DEFAULT_SYSTEM_PROMPT,
                ))

            conv = Conversation(messages=messages, source="openhermes")
            conversations.append(conv)

        except Exception:
            skipped += 1
            continue

    logger.info(
        f"OpenHermes: {len(conversations):,} converted, {skipped:,} skipped"
    )
    return conversations


def format_gsm8k(dataset: Dataset) -> list[Conversation]:
    """
    Converts GSM8K into the standard conversation format.

    GSM8K contains two fields: 'question' and 'answer'.
    The 'answer' includes step-by-step reasoning and ends with ####
    followed by the final numeric answer.

    Example:
        question: "Janet has 3 ducks..."
        answer:   "Janet has 3 ducks...\n#### 18"
    """
    conversations = []
    skipped = 0

    for sample in dataset:
        try:
            question = sample.get("question", "").strip()
            answer = sample.get("answer", "").strip()

            if not question or not answer:
                skipped += 1
                continue

            messages = [
                Message(role=Role.SYSTEM, content=DEFAULT_SYSTEM_PROMPT),
                Message(role=Role.USER, content=question),
                Message(role=Role.ASSISTANT, content=answer),
            ]

            conv = Conversation(messages=messages, source="gsm8k")
            conversations.append(conv)

        except Exception:
            skipped += 1
            continue

    logger.info(
        f"GSM8K: {len(conversations):,} converted, {skipped:,} skipped"
    )
    return conversations


# Formatter registry — pattern that makes adding new datasets easy
FORMATTERS = {
    "openhermes": format_openhermes,
    "gsm8k": format_gsm8k,
    "gsm8k_test": format_gsm8k,   # Same format, different split
}