"""Project-specific exceptions with actionable failure messages."""


class LLMLabError(Exception):
    """Base exception for expected, user-actionable failures."""


class ConfigError(LLMLabError):
    """Raised when an experiment configuration is invalid."""


class ContractError(LLMLabError):
    """Raised when a persisted contract is invalid or incompatible."""


class ArtifactError(LLMLabError):
    """Raised when run artifacts cannot be created or verified safely."""


class DataValidationError(LLMLabError):
    """Raised when input data violates the declared record contract."""
