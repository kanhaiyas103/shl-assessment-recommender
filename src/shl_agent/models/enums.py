"""Shared domain enumerations."""

from enum import StrEnum


class ConversationRole(StrEnum):
    """Roles accepted from assignment clients."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationIntent(StrEnum):
    """Supported conversational behaviors."""

    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"
    CLOSE = "close"


class DecisionAction(StrEnum):
    """Next action selected by the conversation policy."""

    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFUSE = "refuse"
    CLOSE = "close"


class TestType(StrEnum):
    """SHL catalog test-type codes."""

    ABILITY_AND_APTITUDE = "A"
    BIODATA_AND_SITUATIONAL_JUDGEMENT = "B"
    COMPETENCIES = "C"
    DEVELOPMENT_AND_360 = "D"
    ASSESSMENT_EXERCISES = "E"
    KNOWLEDGE_AND_SKILLS = "K"
    PERSONALITY_AND_BEHAVIOR = "P"
    SIMULATIONS = "S"


class LlmProvider(StrEnum):
    """Supported language-model provider configurations."""

    DISABLED = "disabled"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
