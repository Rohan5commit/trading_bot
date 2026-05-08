"""Cerebrium entrypoint for the trained trading model service.

This intentionally reuses the repository's existing FastAPI runtime so the
model, adapter loading, prompts, and response schema stay identical across
Lightning and Cerebrium deployments.
"""

from trained_model_service_runtime import app
