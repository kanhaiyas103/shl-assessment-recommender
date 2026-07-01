"""Constants used by the offline SHL catalog pipeline."""

from shl_agent.models.enums import TestType

ASSIGNMENT_CATALOG_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)
CATALOG_VERSION = "2026-06-30"
SHL_HOST = "www.shl.com"

TEST_TYPE_LABELS: dict[str, TestType] = {
    "ability & aptitude": TestType.ABILITY_AND_APTITUDE,
    "assessment exercises": TestType.ASSESSMENT_EXERCISES,
    "biodata & situational judgment": TestType.BIODATA_AND_SITUATIONAL_JUDGEMENT,
    "biodata & situational judgement": TestType.BIODATA_AND_SITUATIONAL_JUDGEMENT,
    "competencies": TestType.COMPETENCIES,
    "development & 360": TestType.DEVELOPMENT_AND_360,
    "knowledge & skills": TestType.KNOWLEDGE_AND_SKILLS,
    "personality & behavior": TestType.PERSONALITY_AND_BEHAVIOR,
    "personality & behaviour": TestType.PERSONALITY_AND_BEHAVIOR,
    "simulations": TestType.SIMULATIONS,
}

TEST_TYPE_CODES: dict[str, TestType] = {item.value: item for item in TestType}
