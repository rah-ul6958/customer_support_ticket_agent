class ComponentNotReadyError(RuntimeError):
    """Raised when a required model, workflow, or retrieval component is unavailable."""


class AgentProcessingError(RuntimeError):
    """Raised for recoverable failures while processing a customer turn."""
