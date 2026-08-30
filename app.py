"""Compatibility import for the packaged InspectFlow ASGI application."""

from inspectflow.webapp import RunArtifact, RunStore, app, build_run_artifact, create_app

__all__ = ["RunArtifact", "RunStore", "app", "build_run_artifact", "create_app"]
