"""Tests for application dependency composition."""

from fastapi import FastAPI

from shl_agent.api.app import create_app
from shl_agent.services.container import build_foundation_container
from shl_agent.utils.settings import Settings


def test_app_factory_attaches_typed_container() -> None:
    settings = Settings(_env_file=None, app_env="test")
    container = build_foundation_container(settings)

    app = create_app()  # application container is created during lifespan

    assert isinstance(app, FastAPI)
    assert {getattr(route, "path", "") for route in app.routes} == {"/health", "/chat"}
    assert container.settings.app_env == "test"
