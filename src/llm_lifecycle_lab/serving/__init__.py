"""Local, dependency-light model serving interfaces."""

from llm_lifecycle_lab.serving.openai import (
    OpenAIService,
    create_openai_server,
)

__all__ = ["OpenAIService", "create_openai_server"]
