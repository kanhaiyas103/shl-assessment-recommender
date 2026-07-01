"""Catalog assessment domain model."""

from dataclasses import dataclass

from shl_agent.models.enums import TestType


@dataclass(frozen=True, slots=True)
class Assessment:
    """One canonical SHL Individual Test Solution."""

    assessment_id: str
    name: str
    url: str
    test_types: tuple[TestType, ...]
    description: str
    duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    job_levels: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Enforce invariants relied on by retrieval and API mapping."""
        if not self.assessment_id.strip():
            raise ValueError("assessment_id must not be blank")
        if not self.name.strip():
            raise ValueError("assessment name must not be blank")
        if not self.url.startswith("https://www.shl.com/"):
            raise ValueError("assessment URL must be an HTTPS SHL catalog URL")
        if not self.test_types:
            raise ValueError("assessment must have at least one test type")
        if self.duration_minutes is not None and self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive when present")
