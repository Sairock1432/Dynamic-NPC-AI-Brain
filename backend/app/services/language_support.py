from types import MappingProxyType
from typing import Literal, Mapping, cast

LanguageCode = Literal["en", "te", "hi"]

DEFAULT_LANGUAGE: LanguageCode = "en"
LANGUAGE_NAMES: Mapping[LanguageCode, str] = MappingProxyType(
    {
        "en": "English",
        "te": "Telugu (తెలుగు)",
        "hi": "Hindi (हिन्दी)",
    }
)
LANGUAGE_INSTRUCTIONS: Mapping[LanguageCode, str] = MappingProxyType(
    {
        "en": (
            "Respond to the player in English. Understand their message in its original "
            "language, but keep the reply in English unless they explicitly ask for a translation."
        ),
        "te": (
            "Respond to the player in natural Telugu (తెలుగు), using Telugu script. "
            "Understand the player's message in its original language and do not translate "
            "it unless asked. Keep NPC personality, decisions, memories, names, and game facts unchanged."
        ),
        "hi": (
            "Respond to the player in natural Hindi (हिन्दी), using Devanagari script. "
            "Understand the player's message in its original language and do not translate "
            "it unless asked. Keep NPC personality, decisions, memories, names, and game facts unchanged."
        ),
    }
)


def normalize_language(value: object) -> LanguageCode:
    if not isinstance(value, str):
        return DEFAULT_LANGUAGE
    language_code = value.strip().casefold().replace("_", "-").split("-", 1)[0]
    if language_code in LANGUAGE_NAMES:
        return cast(LanguageCode, language_code)
    return DEFAULT_LANGUAGE


def language_instruction(value: object) -> str:
    return LANGUAGE_INSTRUCTIONS[normalize_language(value)]