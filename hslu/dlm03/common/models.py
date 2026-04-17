"""Enum registries of model name strings for each supported LLM provider."""

import enum


class Anthropic(enum.Enum):
    """Anthropic Claude model identifiers."""

    CLAUDE_OPUS_4 = "claude-opus-4-6"
    CLAUDE_SONNET_4 = "claude-sonnet-4-6"
    CLAUDE_HAIKU_4 = "claude-haiku-4-5-20251001"
    CLAUDE_3p5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3p5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"


class Google(enum.Enum):
    """Google Gemini and Gemma model identifiers (Google AI Studio)."""

    GEMINI_3p1_PRO = "gemini-3.1-pro-preview"
    GEMINI_3p1_FLASH_LITE = "gemini-3.1-flash-lite-preview"
    GEMINI_3_FLASH = "gemini-3-flash-preview"
    GEMINI_2p5_PRO = "gemini-2.5-pro"
    GEMINI_2p5_FLASH = "gemini-2.5-flash"
    GEMINI_2p5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2p0_FLASH = "gemini-2.0-flash"
    GEMINI_2p0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_1p5_PRO = "gemini-1.5-pro"
    GEMINI_1p5_FLASH = "gemini-1.5-flash"
    GEMINI_1p5_FLASH_8B = "gemini-1.5-flash-8b"


class Gemma(enum.Enum):
    """Google Gemma model identifiers (served via the Gemini API)."""

    GEMMA_3_27B = "gemma-3-27b-it"
    GEMMA_3_12B = "gemma-3-12b-it"
    GEMMA_3_4B = "gemma-3-4b-it"
    GEMMA_4_26B = "gemma-4-26b-a4b-it"
    GEMMA_4_31B = "gemma-4-31b-it"


class OpenAI(enum.Enum):
    """OpenAI model identifiers."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"
