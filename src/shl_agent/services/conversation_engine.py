"""Deterministic conversation and decision engine for stateless chat turns."""

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from shl_agent.api.models.chat import ChatMessage, ChatResponse, RecommendationResponse
from shl_agent.models.assessment import Assessment
from shl_agent.models.enums import ConversationIntent, DecisionAction, TestType
from shl_agent.models.recommendation import RecommendationReadiness, ResolvedRequirements
from shl_agent.models.retrieval import RetrievalEvidence

_DURATION_RE = re.compile(r"(?P<minutes>\d{1,3})\s*(?:min|mins|minutes)?", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#. -]{1,40}")
_TOKEN_RE = re.compile(r"[a-z0-9+#.]{2,}")

SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "nodejs": "node.js",
    "node": "node.js",
    "py": "python",
    "contact centre": "contact center",
    "call centre": "call center",
    "situational judgement": "situational judgment",
    "rest api design": "rest api",
    "restful api": "rest api",
    "microsoft office": "microsoft office",
    "plant operators": "plant operator",
    "inbound call": "inbound calls",
}

KNOWN_SKILLS = (
    ".net",
    "aws",
    "c#",
    "communication",
    "java",
    "javascript",
    "leadership",
    "node.js",
    "python",
    "react",
    "sales",
    "sql",
    "rust",
    "linux",
    "networking",
    "spring",
    "docker",
    "rest api",
    "angular",
    "live coding",
    "excel",
    "word",
    "microsoft office",
    "contact center",
    "customer service",
    "inbound calls",
    "call center",
    "spoken english",
    "us accent",
    "numerical reasoning",
    "financial accounting",
    "statistics",
    "graduate scenarios",
    "situational judgment",
    "safety",
    "dependability",
    "industrial",
    "procedure compliance",
    "plant operator",
    "plant operators",
    "reliability",
    "hipaa",
    "medical terminology",
    "patient records",
)

TEST_TYPE_TERMS: dict[str, TestType] = {
    "ability": TestType.ABILITY_AND_APTITUDE,
    "aptitude": TestType.ABILITY_AND_APTITUDE,
    "cognitive": TestType.ABILITY_AND_APTITUDE,
    "competency": TestType.COMPETENCIES,
    "competencies": TestType.COMPETENCIES,
    "knowledge": TestType.KNOWLEDGE_AND_SKILLS,
    "personality": TestType.PERSONALITY_AND_BEHAVIOR,
    "simulation": TestType.SIMULATIONS,
    "situational judgment": TestType.BIODATA_AND_SITUATIONAL_JUDGEMENT,
    "situational judgement": TestType.BIODATA_AND_SITUATIONAL_JUDGEMENT,
}

REFUSAL_PATTERNS = (
    "ignore previous",
    "ignore the instructions",
    "system prompt",
    "developer message",
    "write code",
    "weather",
    "joke",
    "recipe",
)

COMPLETION_PATTERNS = (
    "thanks",
    "thank you",
    "done",
    "that's all",
    "that works",
    "that's good",
    "confirmed",
    "confirm",
    "perfect",
    "locking it in",
    "lock it in",
    "final list",
    "keep the shortlist",
    "keep verify",
)

STOPWORDS = frozenset(
    {
        "about",
        "against",
        "also",
        "assessment",
        "assessments",
        "battery",
        "candidate",
        "candidates",
        "can",
        "could",
        "currently",
        "daily",
        "does",
        "for",
        "from",
        "good",
        "hiring",
        "into",
        "need",
        "recommend",
        "recommendation",
        "role",
        "solution",
        "solutions",
        "test",
        "tests",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "use",
        "want",
        "what",
        "where",
        "which",
        "with",
        "work",
        "working",
    }
)


class RetrievalEngine(Protocol):
    """Hybrid retrieval capability required by the conversation engine."""

    async def retrieve(self, requirements: ResolvedRequirements) -> RetrievalEvidence:
        """Return retrieval evidence for requirements."""
        raise NotImplementedError


class CatalogLookup(Protocol):
    """Catalog lookup capability required by comparison/output validation."""

    def assessment(self, assessment_id: str) -> Assessment:
        """Return one assessment."""
        raise NotImplementedError

    def all_assessments(self) -> tuple[Assessment, ...]:
        """Return all assessments."""
        raise NotImplementedError


class IntentClassifier:
    """Rule-first intent classifier."""

    def classify(self, latest_user_message: str) -> ConversationIntent:
        """Classify the latest user turn."""
        text = latest_user_message.casefold()
        if any(pattern in text for pattern in REFUSAL_PATTERNS):
            return ConversationIntent.REFUSE
        if any(_contains_phrase(text, term) for term in COMPLETION_PATTERNS):
            return ConversationIntent.CLOSE
        if any(
            _contains_phrase(text, term)
            for term in ("compare", "vs", "versus", "difference between")
        ):
            return ConversationIntent.COMPARE
        if any(
            _contains_phrase(text, term)
            for term in ("instead", "change", "more", "less", "under", "remove", "add", "drop")
        ):
            return ConversationIntent.REFINE
        if any(
            _contains_phrase(text, term)
            for term in ("recommend", "need", "looking for", "assessment", "test")
        ):
            return ConversationIntent.RECOMMEND
        return ConversationIntent.CLARIFY


class RefinementEngine:
    """Merge corrections and refinements into previous requirements."""

    def merge(
        self,
        previous: ResolvedRequirements,
        update: ResolvedRequirements,
    ) -> ResolvedRequirements:
        """Merge updated constraints without discarding valid previous context."""
        remove_skills = set(update.excluded_assessment_ids)
        skills = tuple(skill for skill in previous.skills if skill not in remove_skills)
        skills = self._unique((*skills, *update.skills))
        competencies = self._unique((*previous.competencies, *update.competencies))
        test_types = tuple(dict.fromkeys((*previous.test_types, *update.test_types)))
        return ResolvedRequirements(
            intent=update.intent,
            role=update.role or previous.role,
            seniority=update.seniority or previous.seniority,
            skills=skills,
            competencies=competencies,
            test_types=test_types,
            max_duration_minutes=update.max_duration_minutes or previous.max_duration_minutes,
            languages=self._unique((*previous.languages, *update.languages)),
            assessment_names=self._unique((*previous.assessment_names, *update.assessment_names)),
            excluded_assessment_ids=previous.excluded_assessment_ids,
            unresolved_questions=update.unresolved_questions,
        )

    @staticmethod
    def _unique(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value for value in values if value))


class ConversationHistoryResolver:
    """Reconstruct current requirements from complete stateless message history."""

    def __init__(
        self,
        intent_classifier: IntentClassifier | None = None,
        refinement_engine: RefinementEngine | None = None,
    ) -> None:
        self._intent_classifier = intent_classifier or IntentClassifier()
        self._refinement_engine = refinement_engine or RefinementEngine()

    def resolve(self, messages: Sequence[ChatMessage]) -> ResolvedRequirements:
        """Merge all user turns into current requirements."""
        current = ResolvedRequirements(intent=ConversationIntent.CLARIFY)
        for message in messages:
            if message.role != "user":
                continue
            update = self._extract_user_turn(message.content)
            current = self._refinement_engine.merge(current, update)
        latest_intent = self._intent_classifier.classify(messages[-1].content)
        return replace(current, intent=latest_intent)

    def _extract_user_turn(self, content: str) -> ResolvedRequirements:
        text = content.casefold()
        intent = self._intent_classifier.classify(content)
        return ResolvedRequirements(
            intent=intent,
            role=self._role(text),
            seniority=self._seniority(text),
            skills=self._skills(text),
            competencies=self._competencies(text),
            test_types=self._test_types(text),
            max_duration_minutes=self._duration(text),
            languages=self._languages(text),
            assessment_names=self._assessment_names(content),
            excluded_assessment_ids=self._removed_skills(text),
        )

    @staticmethod
    def _role(text: str) -> str | None:
        for role in (
            "developer",
            "engineer",
            "manager",
            "sales",
            "graduate",
            "analyst",
            "admin",
            "operator",
            "operators",
        ):
            if _contains_phrase(text, role):
                return "operator" if role == "operators" else role
        return None

    @staticmethod
    def _seniority(text: str) -> str | None:
        for seniority in ("entry-level", "entry level", "junior", "mid", "senior", "graduate"):
            if _contains_phrase(text, seniority):
                return seniority
        return None

    @staticmethod
    def _skills(text: str) -> tuple[str, ...]:
        found = [
            SKILL_ALIASES.get(skill, skill)
            for skill in KNOWN_SKILLS
            if _contains_phrase(text, skill)
        ]
        found.extend(ConversationHistoryResolver._unknown_terms(text, found))
        return tuple(dict.fromkeys(found))

    @staticmethod
    def _competencies(text: str) -> tuple[str, ...]:
        return tuple(
            term
            for term in ("leadership", "communication", "dependability", "reliability")
            if _contains_phrase(text, term)
        )

    @staticmethod
    def _test_types(text: str) -> tuple[TestType, ...]:
        return tuple(
            dict.fromkeys(
                value for term, value in TEST_TYPE_TERMS.items() if _contains_phrase(text, term)
            )
        )

    @staticmethod
    def _duration(text: str) -> int | None:
        if "under" not in text and "less than" not in text and "max" not in text:
            return None
        match = _DURATION_RE.search(text)
        return int(match.group("minutes")) if match else None

    @staticmethod
    def _languages(text: str) -> tuple[str, ...]:
        return tuple(
            language
            for language in ("english", "spanish", "french")
            if _contains_phrase(text, language)
        )

    @staticmethod
    def _assessment_names(content: str) -> tuple[str, ...]:
        quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", content)
        if quoted:
            return tuple(item.strip() for item in quoted)
        return tuple(
            match.group(0).strip()
            for match in _WORD_RE.finditer(content)
            if " test" in match.group(0).casefold()
        )

    @staticmethod
    def _removed_skills(text: str) -> tuple[str, ...]:
        if not _contains_phrase(text, "remove") and not _contains_phrase(text, "not"):
            return ()
        return tuple(skill for skill in KNOWN_SKILLS if _contains_phrase(text, skill))

    @staticmethod
    def _unknown_terms(text: str, known_terms: Sequence[str]) -> tuple[str, ...]:
        known_tokens = {
            token for term in known_terms for token in _TOKEN_RE.findall(term.casefold())
        }
        terms = [
            token
            for token in _TOKEN_RE.findall(text)
            if (
                token not in STOPWORDS
                and token not in known_tokens
                and len(token) >= 3
                and not token.isdigit()
            )
        ]
        return tuple(dict.fromkeys(terms[:12]))


class ConversationPolicy:
    """Conversation-level limits and clarification controls."""

    def __init__(self, max_messages: int = 8, max_clarifications: int = 1) -> None:
        self._max_messages = max_messages
        self._max_clarifications = max_clarifications

    def remaining_turns(self, messages: Sequence[ChatMessage]) -> int:
        """Return remaining messages under the assignment cap."""
        return max(0, self._max_messages - len(messages))

    def clarification_count(self, messages: Sequence[ChatMessage]) -> int:
        """Count assistant clarification questions already asked."""
        return sum(
            1
            for message in messages
            if message.role == "assistant"
            and "?" in message.content
            and not message.content.startswith("I recommend")
        )

    def can_clarify(self, messages: Sequence[ChatMessage]) -> bool:
        """Return whether asking another clarification is allowed."""
        return (
            self.remaining_turns(messages) >= 2
            and self.clarification_count(messages) < self._max_clarifications
        )

    def should_complete(self, messages: Sequence[ChatMessage], intent: ConversationIntent) -> bool:
        """Return whether a stateless response should end the conversation."""
        return intent is ConversationIntent.CLOSE or self.remaining_turns(messages) <= 0


class RecommendationReadinessPolicy:
    """Decide whether to clarify or recommend."""

    def __init__(self, min_confidence: float = 0.35, min_coverage: float = 0.25) -> None:
        self._min_confidence = min_confidence
        self._min_coverage = min_coverage

    def decide(
        self,
        requirements: ResolvedRequirements,
        evidence: RetrievalEvidence,
        *,
        can_clarify: bool,
    ) -> RecommendationReadiness:
        """Return an auditable clarify/recommend decision."""
        if not requirements.has_actionable_context and can_clarify:
            return RecommendationReadiness(
                DecisionAction.CLARIFY,
                "The request lacks a role, skill, assessment name, or assessment type.",
                "What role or skill should the assessment focus on?",
            )
        material_question = self._material_clarification_question(requirements)
        if can_clarify and material_question is not None:
            return RecommendationReadiness(
                DecisionAction.CLARIFY,
                "One missing user fact could materially improve retrieval.",
                material_question,
            )
        if (
            can_clarify
            and evidence.retrieval_confidence < self._min_confidence
            and evidence.required_skill_coverage < self._min_coverage
        ):
            return RecommendationReadiness(
                DecisionAction.CLARIFY,
                "Retrieval confidence is low and more detail would improve recall.",
                "Could you share the main skill, role, or assessment type you need?",
            )
        return RecommendationReadiness(
            DecisionAction.RECOMMEND,
            "Enough grounded evidence to respond.",
        )

    @classmethod
    def _material_clarification_question(
        cls,
        requirements: ResolvedRequirements,
    ) -> str | None:
        skills = set(requirements.skills)
        if cls._needs_software_focus(skills):
            return (
                "Should I prioritize backend, frontend, or balanced full-stack coverage "
                "for this technical role?"
            )
        if "rust" in skills:
            return (
                "SHL does not have a Rust-specific test; should I build a shortlist using "
                "live coding plus adjacent systems and infrastructure assessments?"
            )
        if cls._needs_contact_center_detail(requirements, skills):
            return (
                "What caller language or accent should the contact-center assessment "
                "be calibrated for?"
            )
        if cls._needs_healthcare_language_strategy(requirements, skills):
            return (
                "Are candidates able to complete healthcare knowledge tests in English, "
                "or do you need a Spanish-only assessment approach?"
            )
        if cls._needs_leadership_decision_context(requirements, skills):
            return (
                "Is this leadership assessment for selection against a benchmark or "
                "developmental feedback?"
            )
        return None

    @staticmethod
    def _needs_software_focus(skills: set[str]) -> bool:
        technical_terms = {
            "angular",
            "aws",
            "docker",
            "java",
            "linux",
            "networking",
            "rest api",
            "spring",
            "sql",
        }
        focus_terms = {"backend", "frontend", "balanced"}
        return len(skills.intersection(technical_terms)) >= 4 and not skills.intersection(
            focus_terms
        )

    @staticmethod
    def _needs_contact_center_detail(
        requirements: ResolvedRequirements,
        skills: set[str],
    ) -> bool:
        contact_terms = {
            "call center",
            "contact center",
            "customer service",
            "inbound calls",
            "spoken english",
        }
        if not skills.intersection(contact_terms):
            return False
        return not requirements.languages or (
            "english" in requirements.languages and "us accent" not in skills
        )

    @staticmethod
    def _needs_healthcare_language_strategy(
        requirements: ResolvedRequirements,
        skills: set[str],
    ) -> bool:
        healthcare_terms = {"hipaa", "medical terminology", "patient records"}
        return (
            bool(skills.intersection(healthcare_terms))
            and "spanish" in requirements.languages
            and "english" not in requirements.languages
        )

    @staticmethod
    def _needs_leadership_decision_context(
        requirements: ResolvedRequirements,
        skills: set[str],
    ) -> bool:
        if "leadership" not in skills and "leadership" not in requirements.competencies:
            return False
        decision_terms = {"benchmark", "selection", "development", "feedback"}
        return not skills.intersection(decision_terms) and not requirements.test_types


class ComparisonEngine:
    """Compare assessments using only grounded catalog fields."""

    def __init__(self, catalog: CatalogLookup) -> None:
        self._catalog = catalog

    def compare(self, names: Sequence[str]) -> tuple[Assessment, ...]:
        """Resolve assessment names to catalog entries."""
        normalized = [name.casefold() for name in names]
        matches = [
            assessment
            for assessment in self._catalog.all_assessments()
            if any(name in assessment.name.casefold() for name in normalized)
        ]
        return tuple(matches[:4])

    @staticmethod
    def compose(assessments: Sequence[Assessment]) -> str:
        """Return grounded comparison prose."""
        if len(assessments) < 2:
            return (
                "I could not resolve two catalog assessments to compare. "
                "Please provide exact assessment names."
            )
        lines = ["Here is a grounded comparison using catalog fields only:"]
        for assessment in assessments:
            duration = (
                f"{assessment.duration_minutes} minutes"
                if assessment.duration_minutes
                else "duration not listed"
            )
            test_types = ", ".join(item.value for item in assessment.test_types)
            lines.append(
                f"- {assessment.name}: type {test_types}, {duration}, "
                f"remote testing={assessment.remote_testing}, "
                f"adaptive/IRT={assessment.adaptive_irt}."
            )
        return "\n".join(lines)


class GroundedResponseComposer:
    """Compose conversational text from trusted requirements and retrieved records only."""

    async def compose_recommendation(
        self,
        requirements: ResolvedRequirements,
        evidence: RetrievalEvidence,
        recommendations: Sequence[Assessment],
    ) -> str:
        """Compose recommendation prose without creating recommendation objects."""
        if not recommendations:
            return "I could not find a grounded SHL catalog match for those requirements."
        focus = (
            ", ".join((*requirements.skills, *requirements.competencies))
            or requirements.role
            or "your request"
        )
        names = ", ".join(assessment.name for assessment in recommendations[:3])
        return (
            f"I found SHL assessments for {focus}. Top matches: {names}. "
            f"Retrieval confidence is {evidence.retrieval_confidence:.2f}."
        )

    @staticmethod
    def compose_clarification(readiness: RecommendationReadiness) -> str:
        """Return a clarification question selected by policy."""
        return (
            readiness.clarification_question
            or "Could you share more detail about the role or skill?"
        )

    @staticmethod
    def compose_refusal() -> str:
        """Return an off-topic/prompt-injection refusal."""
        return "I can only help recommend SHL Individual Test Solutions from the catalog."

    @staticmethod
    def compose_close() -> str:
        """Return a concise closing response."""
        return "Glad I could help. I will end the conversation here."


class OutputValidator:
    """Construct ChatResponse only from trusted catalog Assessment objects."""

    def build_response(
        self,
        *,
        reply: str,
        recommendations: Sequence[Assessment],
        end_of_conversation: bool,
    ) -> ChatResponse:
        """Return schema-valid output or raise on unsafe data."""
        seen: set[str] = set()
        items: list[RecommendationResponse] = []
        for assessment in recommendations[:10]:
            if assessment.assessment_id in seen:
                continue
            if not assessment.url.startswith("https://www.shl.com/"):
                raise ValueError("recommendation URL must come from SHL catalog")
            seen.add(assessment.assessment_id)
            items.append(
                RecommendationResponse(
                    name=assessment.name,
                    url=assessment.url,
                    test_type=assessment.test_types[0],
                )
            )
        return ChatResponse(
            reply=reply,
            recommendations=items,
            end_of_conversation=end_of_conversation,
        )


class ConversationEngine:
    """Complete stateless conversation intelligence layer."""

    def __init__(
        self,
        *,
        history_resolver: ConversationHistoryResolver,
        retrieval_engine: RetrievalEngine,
        readiness_policy: RecommendationReadinessPolicy,
        conversation_policy: ConversationPolicy,
        comparison_engine: ComparisonEngine,
        composer: GroundedResponseComposer,
        output_validator: OutputValidator,
    ) -> None:
        self._history_resolver = history_resolver
        self._retrieval_engine = retrieval_engine
        self._readiness_policy = readiness_policy
        self._conversation_policy = conversation_policy
        self._comparison_engine = comparison_engine
        self._composer = composer
        self._output_validator = output_validator

    async def respond(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        """Return a validated chat response for one stateless request."""
        requirements = self._history_resolver.resolve(messages)
        if requirements.intent is ConversationIntent.REFUSE:
            return self._output_validator.build_response(
                reply=self._composer.compose_refusal(),
                recommendations=(),
                end_of_conversation=False,
            )
        if requirements.intent is ConversationIntent.CLOSE:
            return self._output_validator.build_response(
                reply=self._composer.compose_close(),
                recommendations=(),
                end_of_conversation=True,
            )
        if requirements.intent is ConversationIntent.COMPARE:
            assessments = self._comparison_engine.compare(
                requirements.assessment_names or requirements.skills
            )
            return self._output_validator.build_response(
                reply=self._comparison_engine.compose(assessments),
                recommendations=assessments,
                end_of_conversation=False,
            )

        evidence = await self._retrieval_engine.retrieve(requirements)
        readiness = self._readiness_policy.decide(
            requirements,
            evidence,
            can_clarify=self._conversation_policy.can_clarify(messages),
        )
        if readiness.action is DecisionAction.CLARIFY:
            return self._output_validator.build_response(
                reply=self._composer.compose_clarification(readiness),
                recommendations=(),
                end_of_conversation=False,
            )
        recommendations = tuple(result.assessment for result in evidence.results[:10])
        reply = await self._composer.compose_recommendation(requirements, evidence, recommendations)
        return self._output_validator.build_response(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=self._conversation_policy.should_complete(
                messages,
                requirements.intent,
            ),
        )


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.casefold()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
